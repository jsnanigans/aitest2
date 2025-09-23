#!/bin/bash

# Start SAM local without AWS credentials for local development
echo "🚀 Starting local development server (no AWS auth required)..."
echo ""

# Export dummy credentials to bypass AWS auth
export AWS_ACCESS_KEY_ID=local
export AWS_SECRET_ACCESS_KEY=local
export AWS_SESSION_TOKEN=local
export AWS_DEFAULT_REGION=us-east-1
export SAM_CLI_TELEMETRY=0

# Build first with container
echo "🔨 Building Lambda package..."
sam build --use-container --template template-local.yaml

echo ""
echo "📝 API Endpoints (No Auth Required):"
echo "  GET  http://localhost:5448/api/v1/health              - Health check"
echo "  POST http://localhost:5448/api/v1/process/{userId}    - Process measurements"
echo "  POST http://localhost:5448/api/v1/replay/{userId}     - Replay measurements"
echo "  POST http://localhost:5448/api/v1/cleanup/{userId}    - Cleanup with reset"
echo "  GET  http://localhost:5448/api/v1/state/{userId}      - Get user state"
echo ""
echo "Press Ctrl+C to stop..."
echo ""

# Start SAM local with skip-pull-image to speed up startup
sam local start-api \
    --port 5448 \
    --docker-network bridge \
    --template template-local.yaml \
    --skip-pull-image \
    --warm-containers EAGER \
    --debug-port 5858 \
    --parameter-overrides "Environment=local"