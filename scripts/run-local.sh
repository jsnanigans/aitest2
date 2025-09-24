#!/bin/bash
# Script to run the weight processor locally with DynamoDB Local (REQUIRED)

set -e

echo "Starting local development environment with DynamoDB..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker is not running. Please start Docker first."
    echo "DynamoDB Local requires Docker to run."
    exit 1
fi

# Check if boto3 is installed
if ! uv run python -c "import boto3" 2>/dev/null; then
    echo "Installing boto3..."
    uv pip install boto3
fi

# Start DynamoDB Local
echo "Starting DynamoDB Local..."
docker-compose up -d dynamodb-local dynamodb-admin

# Wait for DynamoDB to be ready
echo "Waiting for DynamoDB Local to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:8000 > /dev/null 2>&1; then
        echo "✓ DynamoDB Local is ready!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "Error: DynamoDB Local failed to start"
        echo "Check Docker logs: docker-compose logs dynamodb-local"
        exit 1
    fi
    sleep 1
done

# Export environment variables
export $(grep -v '^#' .env.local | xargs)

echo ""
echo "Environment configured:"
echo "  DYNAMODB_ENDPOINT=$DYNAMODB_ENDPOINT"
echo "  DYNAMODB_TABLE_NAME=$DYNAMODB_TABLE_NAME"
echo "  AWS_REGION=$AWS_DEFAULT_REGION"
echo ""
echo "DynamoDB Admin UI available at: http://localhost:8001"
echo ""

# Run the main script with arguments
if [ $# -eq 0 ]; then
    echo "Usage: $0 <csv_file> [options]"
    echo "Example: $0 data/weights.csv --max-users 10"
    exit 1
fi

echo "Running weight processor with DynamoDB..."
uv run python main.py "$@"