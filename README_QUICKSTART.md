# Weight Processor - Quick Start Guide

## Prerequisites

- **Docker Desktop** installed and running
- **Python 3.11+** with `uv` package manager

## Instant Start

```bash
# Run with default data (handles everything automatically)
./run.sh

# Run with specific CSV file
./run.sh data/weights.csv

# Run with options
./run.sh data/weights.csv --max-users 100 --no-viz
```

That's it! The script automatically:
- ✅ Installs required dependencies (boto3)
- ✅ Starts DynamoDB Local
- ✅ Creates tables as needed
- ✅ Sets up environment variables
- ✅ Runs the processor

## Alternative: Using Make

```bash
# Run with default settings
make run

# Run with specific file
make run-file FILE=data/weights.csv

# Check setup status
./scripts/check-setup.sh
```

## View Your Data

Open the DynamoDB Admin UI to inspect data:
```bash
open http://localhost:8001
```

## Troubleshooting

### "Docker is not running"
→ Start Docker Desktop

### "Cannot connect to DynamoDB"
```bash
# Restart DynamoDB Local
docker-compose down
docker-compose up -d dynamodb-local
```

### "Module 'boto3' not found"
```bash
uv pip install boto3
```

## Manual Setup (if needed)

```bash
# 1. Install dependencies
uv pip install boto3

# 2. Start DynamoDB Local
docker-compose up -d dynamodb-local dynamodb-admin

# 3. Set environment variables
export DYNAMODB_ENDPOINT=http://localhost:8000
export DYNAMODB_TABLE_NAME=weight-processor-state
export AWS_ACCESS_KEY_ID=local
export AWS_SECRET_ACCESS_KEY=local

# 4. Run
uv run python main.py data/weights.csv
```

## Production Deployment

For AWS deployment, see [README_DYNAMODB.md](README_DYNAMODB.md)