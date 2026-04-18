"""AWS Lambda handler: shut down the ECS Express 'chunking' service when idle.

Idleness signal: the CloudWatch log group `LOG_GROUP` had no events for
`IDLE_MINUTES` minutes. Access logs from the HTTP server keep the stream
fresh as long as a user's browser tab is open.

Env vars (all required):
    LOG_GROUP     - e.g. /ecs/chunking
    SERVICE_ARN   - arn:aws:ecs:<region>:<acct>:service/default/chunking
    CLUSTER       - default (ECS cluster name)
    IDLE_MINUTES  - threshold (defaults to 15 if unset)

Triggered every 5 minutes by an EventBridge schedule.
"""
from __future__ import annotations

import os
import time

import boto3

IDLE_MINUTES = int(os.environ.get("IDLE_MINUTES", "15"))
LOG_GROUP = os.environ["LOG_GROUP"]
SERVICE_ARN = os.environ["SERVICE_ARN"]
CLUSTER = os.environ.get("CLUSTER", "default")

logs = boto3.client("logs")
ecs = boto3.client("ecs")


def lambda_handler(event, context):
    # 1. Find the newest log stream in the group
    try:
        r = logs.describe_log_streams(
            logGroupName=LOG_GROUP,
            orderBy="LastEventTime",
            descending=True,
            limit=1,
        )
    except logs.exceptions.ResourceNotFoundException:
        print(f"log group {LOG_GROUP} not found — service likely already deleted")
        return {"action": "noop", "reason": "log_group_missing"}

    streams = r.get("logStreams", [])
    if not streams:
        print("no log streams yet — service may be warming up")
        return {"action": "noop", "reason": "no_streams"}

    last_ms = streams[0].get("lastEventTimestamp", 0)
    now_ms = int(time.time() * 1000)
    idle_min = (now_ms - last_ms) / 60000.0
    print(f"last log event {idle_min:.1f} min ago (threshold {IDLE_MINUTES} min)")

    if idle_min < IDLE_MINUTES:
        return {"action": "noop", "idle_minutes": round(idle_min, 1)}

    # 2. Confirm service still exists
    service_name = SERVICE_ARN.rsplit("/", 1)[-1]
    try:
        desc = ecs.describe_services(cluster=CLUSTER, services=[service_name])
        live = [s for s in desc.get("services", [])
                if s.get("status") == "ACTIVE"]
        if not live:
            print("service already gone / inactive — nothing to do")
            return {"action": "noop", "reason": "already_gone"}
    except Exception as e:
        print(f"describe_services failed: {e}")
        return {"action": "error", "reason": str(e)}

    # 3. Delete — this tears down tasks + ALB. Recreate via scripts/wake.sh.
    print(f"idle {idle_min:.1f} min → deleting {SERVICE_ARN}")
    try:
        ecs.delete_express_gateway_service(service=SERVICE_ARN)
        return {"action": "deleted",
                "idle_minutes": round(idle_min, 1),
                "service_arn": SERVICE_ARN}
    except Exception as e:
        print(f"delete failed: {e}")
        return {"action": "error", "reason": str(e)}
