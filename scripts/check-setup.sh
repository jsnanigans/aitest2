#!/bin/bash
# Check if the environment is properly set up for running the weight processor

set -e

echo "Checking environment setup..."
echo ""

# Check Docker
echo -n "Docker: "
if docker info > /dev/null 2>&1; then
    echo "✓ Running"
else
    echo "✗ Not running"
    echo "  Please start Docker Desktop first"
    exit 1
fi

# Check boto3
echo -n "boto3: "
if uv run python -c "import boto3" 2>/dev/null; then
    echo "✓ Installed"
else
    echo "✗ Not installed"
    echo "  Installing boto3..."
    uv pip install boto3
fi

# Check DynamoDB Local
echo -n "DynamoDB Local: "
if docker ps | grep -q weight-processor-dynamodb; then
    echo "✓ Running"
else
    echo "✗ Not running"
    echo "  Starting DynamoDB Local..."
    docker-compose up -d dynamodb-local dynamodb-admin

    # Wait for it to be ready
    for i in {1..10}; do
        if curl -s http://localhost:8000 > /dev/null 2>&1; then
            echo "  ✓ DynamoDB Local is ready"
            break
        fi
        if [ $i -eq 10 ]; then
            echo "  ✗ Failed to start"
            exit 1
        fi
        sleep 1
    done
fi

# Check if DynamoDB is accessible
echo -n "DynamoDB Connection: "
if curl -s http://localhost:8000 > /dev/null 2>&1; then
    echo "✓ Accessible"
else
    echo "✗ Not accessible"
    echo "  Check Docker logs: docker-compose logs dynamodb-local"
    exit 1
fi

# Check environment variables
echo ""
echo "Environment variables:"
if [ -z "$DYNAMODB_ENDPOINT" ]; then
    echo "  DYNAMODB_ENDPOINT: Not set (will use http://localhost:8000)"
    export DYNAMODB_ENDPOINT=http://localhost:8000
else
    echo "  DYNAMODB_ENDPOINT: $DYNAMODB_ENDPOINT"
fi

if [ -z "$DYNAMODB_TABLE_NAME" ]; then
    echo "  DYNAMODB_TABLE_NAME: Not set (will use weight-processor-state)"
    export DYNAMODB_TABLE_NAME=weight-processor-state
else
    echo "  DYNAMODB_TABLE_NAME: $DYNAMODB_TABLE_NAME"
fi

echo ""
echo "✓ All checks passed! You can now run:"
echo "  make run                    # Run with default data"
echo "  make run-file FILE=data.csv # Run with specific file"
echo ""
echo "DynamoDB Admin UI: http://localhost:8001"