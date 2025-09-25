# Local Development Setup

## Quick Start (No AWS Auth Required)

The local development environment has been configured to run WITHOUT AWS credentials or MFA authentication.

### Method 1: Using the start script (Recommended)
```bash
./start_local.sh
```

### Method 2: Using Make commands
```bash
make docker-run     # Start local API on port 5448
```

### Method 3: Manual commands
```bash
# Set dummy credentials to bypass AWS auth
export AWS_ACCESS_KEY_ID=local
export AWS_SECRET_ACCESS_KEY=local
export AWS_SESSION_TOKEN=local

# Build and run
sam build --use-container --template template-local.yaml
sam local start-api --port 5448 --docker-network bridge --template template-local.yaml --skip-pull-image --warm-containers EAGER
```

## Testing the API

Once the server is running, test it with:

```bash
./test_local_api.sh
```

Or manually:
```bash
# Health check
curl http://localhost:5448/api/v1/health

# Process measurements
curl -X POST http://localhost:5448/api/v1/process/test-user \
    -H "Content-Type: application/json" \
    -d '{"measurements": [{"weight": 75.5, "unit": "kg", "effectiveDateTime": "2024-01-01T10:00:00Z", "source": "patient-device"}]}'
```

## Available Endpoints (No Auth Required)

- `GET  /api/v1/health` - Health check
- `POST /api/v1/process/{userId}` - Process measurements
- `POST /api/v1/replay/{userId}` - Replay measurements
- `POST /api/v1/cleanup/{userId}` - Cleanup with reset
- `GET  /api/v1/state/{userId}` - Get user state
- `DELETE /api/v1/state/{userId}` - Delete user state

## Performance Optimizations

The following optimizations have been applied to speed up local development:

1. **Skip Docker image pulls**: `--skip-pull-image` flag prevents re-downloading images
2. **Warm containers**: `--warm-containers EAGER` keeps containers running between invocations
3. **Dummy AWS credentials**: Bypasses AWS authentication entirely for local testing
4. **Local template**: Uses `template-local.yaml` which has no authentication requirements

## Troubleshooting

### If you still get MFA prompts
Make sure you're using one of the methods above that sets the dummy AWS credentials.

### If Docker is slow to start
1. Ensure Docker Desktop has enough memory allocated (4GB+ recommended)
2. Use the `--skip-pull-image` flag to avoid re-downloading images
3. Consider pruning old Docker images: `docker system prune -a`

### If endpoints return errors
Check that the local server is running on port 5448:
```bash
lsof -i :5448
```

## Environment Variables

The following environment variables are set automatically for local development:
- `AWS_ACCESS_KEY_ID=local`
- `AWS_SECRET_ACCESS_KEY=local`
- `AWS_SESSION_TOKEN=local`
- `AWS_DEFAULT_REGION=us-east-1`
- `SAM_CLI_TELEMETRY=0`