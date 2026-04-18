#!/usr/bin/env bash
# Recreate the ECS Express 'chunking' service after the idle-watcher Lambda
# deleted it. Prints the new URL when ready.
#
# Prereq: AWS CLI configured with creds that can create ECS Express services
# in ap-south-1. Run from project root.
set -euo pipefail

REGION="${AWS_DEFAULT_REGION:-ap-south-1}"
ACCT="${AWS_ACCOUNT_ID:-902451183446}"
SERVICE_NAME="${SERVICE_NAME:-chunking}"
CLUSTER="${CLUSTER:-default}"
IMAGE="${IMAGE:-${ACCT}.dkr.ecr.${REGION}.amazonaws.com/sop-chunker:latest}"

if aws ecs describe-services --cluster "$CLUSTER" --services "$SERVICE_NAME" --region "$REGION" \
     --query 'services[0].status' --output text 2>/dev/null | grep -q ACTIVE; then
    echo "Service already ACTIVE."
    aws ecs describe-express-gateway-service \
      --service "arn:aws:ecs:${REGION}:${ACCT}:service/${CLUSTER}/${SERVICE_NAME}" \
      --region "$REGION" \
      --query 'service.activeConfigurations[0].ingressPaths[0].endpoint' --output text
    exit 0
fi

PRIMARY=$(mktemp)
SCALE=$(mktemp)
cat > "$PRIMARY" <<EOF
{
  "image": "${IMAGE}",
  "containerPort": 17842,
  "awsLogsConfiguration": {"logGroup": "/ecs/chunking", "logStreamPrefix": "app"},
  "environment": [{"name": "DASH_PORT", "value": "17842"}]
}
EOF
cat > "$SCALE" <<EOF
{"minTaskCount": 1, "maxTaskCount": 1,
 "autoScalingMetric": "REQUEST_COUNT_PER_TARGET", "autoScalingTargetValue": 60}
EOF

echo "Creating Express service in ${REGION}..."
aws ecs create-express-gateway-service \
  --service-name "$SERVICE_NAME" \
  --cluster "$CLUSTER" \
  --region "$REGION" \
  --execution-role-arn "arn:aws:iam::${ACCT}:role/ecsTaskExecutionRole" \
  --infrastructure-role-arn "arn:aws:iam::${ACCT}:role/ecsExpressInfrastructureRole" \
  --primary-container "file://$PRIMARY" \
  --cpu 1024 --memory 2048 \
  --health-check-path / \
  --scaling-target "file://$SCALE" \
  --tags 'key=Project,value=chunking' 'key=WokeBy,value=scripts-wake-sh' \
  --query 'service.activeConfigurations[0].ingressPaths[0].endpoint' --output text

rm -f "$PRIMARY" "$SCALE"

echo
echo "Service created. Initial task spin-up takes ~2-3 minutes before the URL"
echo "responds with HTTP 200. Tail logs with:"
echo "  aws logs tail /ecs/chunking --follow --region ${REGION}"
