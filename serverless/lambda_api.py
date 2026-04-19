"""AWS Lambda handler for the serverless chunking API.

Routes (Lambda Function URL mode, event v2):
  GET  /                   -> serve UI
  GET  /api/upload-url     -> presigned S3 POST for client-side upload
  POST /api/run            -> RunTask on Fargate for an uploaded job
  GET  /api/status?job=ID  -> current status.json from S3
  GET  /api/download?job=ID -> 302 to presigned S3 GET for chunks.zip

Environment vars (required):
  BUCKET           - S3 bucket used by the Fargate task (chunking-jobs-...)
  CLUSTER          - ECS cluster name (default)
  TASK_DEFINITION  - ECS task-def family or ARN (chunking-worker)
  SUBNETS          - comma-separated subnet IDs for the task ENI
  SECURITY_GROUPS  - comma-separated security group IDs

Everything is boto3 + stdlib — no extra deps needed so the Lambda zip stays tiny.
"""
from __future__ import annotations

import base64
import json
import os
import secrets
import time
from pathlib import Path

import boto3
from botocore.config import Config

REGION = os.environ.get("AWS_REGION", "ap-south-1")
BUCKET = os.environ["BUCKET"]
CLUSTER = os.environ.get("CLUSTER", "default")
TASK_DEFINITION = os.environ["TASK_DEFINITION"]
SUBNETS = [s.strip() for s in os.environ["SUBNETS"].split(",") if s.strip()]
SECURITY_GROUPS = [s.strip() for s in os.environ["SECURITY_GROUPS"].split(",") if s.strip()]

# Use regional S3 endpoint + virtual-hosted style so presigned POST URLs point
# directly at the regional endpoint (default is the global s3.amazonaws.com
# which 307-redirects browsers on POST — they drop the body).
s3 = boto3.client(
    "s3", region_name=REGION,
    endpoint_url=f"https://s3.{REGION}.amazonaws.com",
    config=Config(signature_version="s3v4",
                  s3={"addressing_style": "virtual"}),
)
ecs = boto3.client("ecs", region_name=REGION)
logs = boto3.client("logs", region_name=REGION)

WORKER_LOG_GROUP = "/ecs/chunking-worker"

CHUNK_SOURCES = {
    "elements":   "elements.jsonl",
    "structural": "structural.jsonl",
    "coarse":     "coarse.jsonl",
}

HERE = Path(__file__).resolve().parent
UI_HTML = (HERE / "index.html").read_text(encoding="utf-8") if (HERE / "index.html").exists() else "<h1>UI missing</h1>"


# ---------- helpers ----------

def _ok(body, ctype="application/json", extra_headers=None, status=200):
    if ctype == "application/json" and not isinstance(body, (str, bytes)):
        body = json.dumps(body, default=str)
    headers = {"Content-Type": ctype, "Cache-Control": "no-store"}
    if extra_headers:
        headers.update(extra_headers)
    return {"statusCode": status, "headers": headers, "body": body,
            "isBase64Encoded": False}


def _err(msg, status=400):
    return _ok({"error": msg}, status=status)


def _qs(event: dict) -> dict:
    """Lambda Function URL v2 puts single-value params in queryStringParameters."""
    return event.get("queryStringParameters") or {}


def _job_id() -> str:
    ts = time.strftime("%Y%m%d-%H%M%S")
    return f"{ts}-{secrets.token_hex(4)}"


# ---------- route handlers ----------

def serve_ui(event):
    return _ok(UI_HTML, ctype="text/html; charset=utf-8")


def generate_upload_url(event):
    qs = _qs(event)
    filename = qs.get("filename", "input.pdf")
    if not filename.lower().endswith(".pdf"):
        return _err("only .pdf accepted")
    job_id = _job_id()
    in_key = f"jobs/{job_id}/input.pdf"
    meta_key = f"jobs/{job_id}/meta.json"

    # Store a tiny meta.json with the original filename so the task can use it.
    s3.put_object(Bucket=BUCKET, Key=meta_key,
                  Body=json.dumps({"original_name": filename}).encode("utf-8"),
                  ContentType="application/json")

    post = s3.generate_presigned_post(
        Bucket=BUCKET, Key=in_key,
        Fields={"Content-Type": "application/pdf"},
        Conditions=[
            {"Content-Type": "application/pdf"},
            ["content-length-range", 1, 50 * 1024 * 1024],  # up to 50 MB
        ],
        ExpiresIn=600,
    )
    return _ok({"job_id": job_id, "upload": post})


def run_task(event):
    body = event.get("body") or "{}"
    if event.get("isBase64Encoded"):
        body = base64.b64decode(body).decode()
    data = json.loads(body)
    job_id = data.get("job_id")
    if not job_id:
        return _err("job_id required")

    # Confirm the PDF is actually uploaded before spending money on a task.
    in_key = f"jobs/{job_id}/input.pdf"
    try:
        s3.head_object(Bucket=BUCKET, Key=in_key)
    except Exception:
        return _err(f"input not found at s3://{BUCKET}/{in_key}", 404)

    resp = ecs.run_task(
        cluster=CLUSTER,
        taskDefinition=TASK_DEFINITION,
        launchType="FARGATE",
        count=1,
        networkConfiguration={"awsvpcConfiguration": {
            "subnets": SUBNETS,
            "securityGroups": SECURITY_GROUPS,
            "assignPublicIp": "ENABLED",
        }},
        overrides={"containerOverrides": [{
            "name": "worker",
            "environment": [
                {"name": "JOB_ID", "value": job_id},
                {"name": "BUCKET", "value": BUCKET},
                {"name": "REGION", "value": REGION},
            ],
        }]},
        tags=[
            {"key": "Project", "value": "chunking"},
            {"key": "JobId", "value": job_id},
        ],
    )
    failures = resp.get("failures") or []
    if failures:
        return _err({"ecs_failures": failures, "job_id": job_id}, 500)

    tasks = resp.get("tasks") or []
    task_arn = tasks[0]["taskArn"] if tasks else None

    # Seed initial status so the UI sees "queued" immediately.
    s3.put_object(
        Bucket=BUCKET, Key=f"jobs/{job_id}/status.json",
        Body=json.dumps({"job_id": job_id, "status": "queued",
                         "queued_at": time.time(),
                         "task_arn": task_arn}).encode("utf-8"),
        ContentType="application/json", CacheControl="no-store",
    )
    return _ok({"job_id": job_id, "task_arn": task_arn, "status": "queued"})


def read_status(event):
    qs = _qs(event)
    job_id = qs.get("job")
    if not job_id:
        return _err("job query param required")
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=f"jobs/{job_id}/status.json")
        data = json.loads(obj["Body"].read())
    except s3.exceptions.NoSuchKey:
        return _ok({"job_id": job_id, "status": "unknown"})
    except Exception as e:
        return _err(str(e), 500)
    return _ok(data)


def read_log(event):
    """Tail CloudWatch Logs for the task ARN recorded in the job's status.json."""
    qs = _qs(event)
    job_id = qs.get("job")
    if not job_id:
        return _err("job query param required")
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=f"jobs/{job_id}/status.json")
        status = json.loads(obj["Body"].read())
    except Exception as e:
        return _err(f"job {job_id} not found: {e}", 404)

    task_arn = status.get("task_arn")
    if not task_arn:
        return _ok({"lines": [], "note": "no task_arn in status yet"})

    task_id = task_arn.rsplit("/", 1)[-1]
    # awslogs-stream-prefix=task + container name 'worker' + task id
    stream = f"task/worker/{task_id}"

    try:
        resp = logs.get_log_events(
            logGroupName=WORKER_LOG_GROUP,
            logStreamName=stream,
            startFromHead=True,
            limit=500,
        )
        events = [{"t": e["timestamp"], "m": e["message"]} for e in resp.get("events", [])]
    except logs.exceptions.ResourceNotFoundException:
        return _ok({"lines": [], "note": "log stream not created yet"})
    except Exception as e:
        return _err(f"logs error: {e}", 500)
    return _ok({"lines": events, "stream": stream})


def read_metrics(event):
    qs = _qs(event)
    job_id = qs.get("job")
    if not job_id:
        return _err("job query param required")
    prefix = f"jobs/{job_id}/metrics/"
    out = {}
    try:
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
            for obj in page.get("Contents") or []:
                name = obj["Key"].rsplit("/", 1)[-1]
                body = s3.get_object(Bucket=BUCKET, Key=obj["Key"])["Body"].read()
                try:
                    out[name] = json.loads(body)
                except Exception:
                    out[name] = {"_raw": body.decode("utf-8", errors="replace")[:2000]}
    except Exception as e:
        return _err(str(e), 500)
    return _ok(out)


def read_chunks(event):
    """Paginated view of elements/structural/coarse JSONL lines."""
    qs = _qs(event)
    job_id = qs.get("job")
    src = qs.get("src", "coarse")
    if not job_id:
        return _err("job query param required")
    if src not in CHUNK_SOURCES:
        return _err(f"unknown src; valid: {list(CHUNK_SOURCES)}", 400)

    key = f"jobs/{job_id}/{CHUNK_SOURCES[src]}"
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=key)
        body = obj["Body"].read().decode("utf-8", errors="replace")
    except s3.exceptions.NoSuchKey:
        return _ok({"src": src, "exists": False, "total": 0, "rows": []})
    except Exception as e:
        return _err(f"read error: {e}", 500)

    offset = int(qs.get("offset", "0"))
    limit = min(int(qs.get("limit", "50")), 500)
    full_id = qs.get("id")

    records = []
    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except Exception:
            continue
    total = len(records)

    if full_id is not None:
        try:
            want = int(full_id)
        except ValueError:
            want = full_id
        match = next((r for r in records
                      if r.get("id", r.get("idx")) == want), None)
        return _ok({"src": src, "record": match})

    slice_ = records[offset:offset + limit]
    rows = []
    for r in slice_:
        text = (r.get("text") or "")
        row = {
            "id": r.get("id", r.get("idx")),
            "n_chars": r.get("n_chars", len(text)),
            "preview": text.replace("\n", " ")[:160],
        }
        if src == "elements":
            row["type"] = r.get("type")
            row["page"] = r.get("page")
            row["heading_level"] = r.get("heading_level")
        else:
            row["heading"] = r.get("heading") or r.get("heading_first")
            row["heading_path"] = r.get("heading_path")
            row["pages"] = r.get("pages")
            row["contains_table"] = r.get("contains_table", False)
        rows.append(row)
    return _ok({"src": src, "exists": True, "total": total,
                "offset": offset, "limit": limit, "rows": rows})


def list_figures(event):
    qs = _qs(event)
    job_id = qs.get("job")
    if not job_id:
        return _err("job query param required")
    prefix = f"jobs/{job_id}/figures/"
    figs = []
    try:
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
            for obj in page.get("Contents") or []:
                name = obj["Key"].rsplit("/", 1)[-1]
                figs.append({"name": name, "size": obj["Size"]})
    except Exception as e:
        return _err(str(e), 500)
    return _ok({"figures": figs})


def get_figure(event):
    qs = _qs(event)
    job_id = qs.get("job")
    name = qs.get("name", "")
    if not job_id or not name or "/" in name:
        return _err("job + safe name required")
    key = f"jobs/{job_id}/figures/{name}"
    try:
        url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": BUCKET, "Key": key},
            ExpiresIn=600,
        )
    except Exception as e:
        return _err(str(e), 500)
    return {"statusCode": 302,
            "headers": {"Location": url, "Cache-Control": "no-store"},
            "body": ""}


def download_redirect(event):
    qs = _qs(event)
    job_id = qs.get("job")
    if not job_id:
        return _err("job query param required")
    key = f"jobs/{job_id}/chunks.zip"
    try:
        s3.head_object(Bucket=BUCKET, Key=key)
    except Exception:
        return _err("chunks.zip not available yet", 404)
    url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": BUCKET, "Key": key,
                "ResponseContentDisposition":
                f'attachment; filename="{job_id}_coarse_chunks.zip"'},
        ExpiresIn=600,
    )
    return {"statusCode": 302,
            "headers": {"Location": url, "Cache-Control": "no-store"},
            "body": ""}


# ---------- dispatcher ----------

ROUTES = {
    ("GET", "/"):                 serve_ui,
    ("GET", "/index.html"):       serve_ui,
    ("GET", "/api/upload-url"):   generate_upload_url,
    ("POST", "/api/run"):         run_task,
    ("GET", "/api/status"):       read_status,
    ("GET", "/api/log"):          read_log,
    ("GET", "/api/metrics"):      read_metrics,
    ("GET", "/api/chunks"):       read_chunks,
    ("GET", "/api/figures"):      list_figures,
    ("GET", "/api/figure"):       get_figure,
    ("GET", "/api/download"):     download_redirect,
}


def lambda_handler(event, context):
    # Lambda Function URL v2 event shape
    rc = event.get("requestContext") or {}
    method = (rc.get("http") or {}).get("method") or event.get("httpMethod", "GET")
    path = event.get("rawPath") or event.get("path") or "/"
    fn = ROUTES.get((method, path))
    if fn is None:
        return _err(f"no route for {method} {path}", 404)
    try:
        return fn(event)
    except Exception as e:
        import traceback
        return _err({"error": f"{type(e).__name__}: {e}",
                     "traceback": traceback.format_exc()[-600:]}, 500)
