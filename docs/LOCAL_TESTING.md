# Local Testing with Docker

## Quick Start

No Python installation needed! Just Docker and SAM CLI.

**🔓 No Authentication Required for Local Testing!**

All endpoints are open when running locally - no API keys needed.

### Prerequisites

```bash
# Install Docker Desktop
# https://www.docker.com/products/docker-desktop

# Install SAM CLI
brew install aws-sam-cli
```

### Run Locally

```bash
# 1. Build and start the API
make docker-run

# 2. In another terminal, test the endpoints
make docker-test         # Test process endpoint
make docker-test-replay  # Test replay endpoint

# 3. View logs
make docker-logs
```

## Available Commands

### Basic Workflow

```bash
make docker-build  # Build Lambda package with Docker
make docker-run    # Start local API Gateway (http://localhost:5448)
make docker-test   # Test with sample data
make docker-stop   # Stop all containers
make docker-clean  # Clean up everything
```

### Testing Endpoints

The API runs at `http://localhost:5448` with these endpoints (no auth required locally):

#### 0. Health Check
```bash
curl http://localhost:5448/api/v1/health

# Or with make
make docker-health
```

Returns system status including:
- Overall health status
- Database connectivity
- Configuration status
- Enabled features (Kalman, quality scoring, etc.)
- Runtime information (region, memory, version)

#### 1. Process Measurements
```bash
curl -X POST http://localhost:5448/api/v1/process/user123 \
  -H "Content-Type: application/json" \
  -d '{
    "measurements": [{
      "uuid": "550e8400-e29b-41d4-a716-446655440000",
      "weight": 75.5,
      "unit": "kg",
      "effectiveDateTime": "2024-01-01T10:00:00Z",
      "source": "patient-device"
    }]
  }'
```

#### 2. Run Replay
```bash
curl -X POST http://localhost:5448/api/v1/replay/user123 \
  -H "Content-Type: application/json" \
  -d '{
    "replay_from_timestamp": "2024-01-01T00:00:00Z",
    "measurements": [{
      "uuid": "550e8400-e29b-41d4-a716-446655440001",
      "weight": 76.0,
      "unit": "kg",
      "effectiveDateTime": "2024-01-02T10:00:00Z",
      "source": "patient-device"
    }]
  }'
```

#### 3. Cleanup (Process with Reset)
```bash
curl -X POST http://localhost:5448/api/v1/cleanup/user123 \
  -H "Content-Type: application/json" \
  -d '{
    "measurements": [...],
    "options": {
      "reset_state": true
    }
  }'
```

#### 4. Get User State
```bash
curl http://localhost:5448/api/v1/state/user123
```

### Direct Lambda Invocation

Test with event files:

```bash
# Use existing test events
make docker-invoke

# Use specific event file
make docker-invoke-file EVENT=test_events/process_measurements.json

# Generate new test event
make docker-generate-event
```

### Debugging

```bash
# View container logs
make docker-logs

# Tail logs in real-time (while API is running)
make docker-tail-logs

# Stop everything if something goes wrong
make docker-stop

# Full cleanup
make docker-clean
```

## Test Events

Test events are in `test_events/`:

- `process_measurements.json` - Standard processing
- `historical_conflict.json` - Test conflict handling
- `cleanup_user.json` - Full cleanup/reset
- `get_state.json` - Retrieve user state

## Environment Variables

The local environment uses these defaults (from `template.yaml`):

```yaml
LOG_LEVEL: INFO
DB_BACKEND: dynamodb  # Uses local DynamoDB
KALMAN_ENABLED: true
QUALITY_SCORING_ENABLED: true
OUTLIER_DETECTION_ENABLED: true
```

## Troubleshooting

### Port Already in Use
```bash
# Find what's using port 5448
lsof -i :5448

# Kill it
kill -9 <PID>

# Or use a different port
sam local start-api --port 5449
```

### Container Issues
```bash
# Clean everything and start fresh
make docker-clean
make docker-run
```

### Slow First Run
The first run downloads Docker images (~500MB). Subsequent runs are faster.

### Memory Issues
If Lambda runs out of memory, increase in `template.yaml`:
```yaml
MemorySize: 1024  # Increase to 2048 if needed
```

## Production Deployment

When ready to deploy to AWS:

```bash
# Configure AWS credentials
aws configure

# Deploy to dev
sam deploy --guided

# Deploy to staging/prod
make deploy-staging
make deploy-prod
```