#!/bin/bash
# Simple run script that handles DynamoDB Local setup automatically

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Weight Processor - Starting...${NC}"
echo ""

# Check Docker
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}Error: Docker is not running${NC}"
    echo "Please start Docker Desktop and try again"
    exit 1
fi

# Install boto3 if needed
if ! uv run python -c "import boto3" 2>/dev/null; then
    echo -e "${YELLOW}Installing boto3...${NC}"
    uv pip install boto3
fi

# Start DynamoDB Local if not running
if ! docker ps | grep -q weight-processor-dynamodb; then
    echo -e "${YELLOW}Starting DynamoDB Local...${NC}"
    docker-compose up -d dynamodb-local dynamodb-admin

    # Wait for it to be ready
    echo -n "Waiting for DynamoDB to be ready"
    for i in {1..30}; do
        if curl -s http://localhost:8000 > /dev/null 2>&1; then
            echo -e " ${GREEN}✓${NC}"
            break
        fi
        echo -n "."
        if [ $i -eq 30 ]; then
            echo -e " ${RED}✗${NC}"
            echo -e "${RED}Error: DynamoDB Local failed to start${NC}"
            echo "Check logs: docker-compose logs dynamodb-local"
            exit 1
        fi
        sleep 1
    done
else
    echo -e "${GREEN}✓${NC} DynamoDB Local is already running"
fi

echo -e "${GREEN}✓${NC} Environment ready"
echo ""
echo "DynamoDB Admin UI: http://localhost:8001"
echo ""

# Set environment variables
export DYNAMODB_ENDPOINT=http://localhost:8000
export DYNAMODB_TABLE_NAME=weight-processor-state
export AWS_ACCESS_KEY_ID=local
export AWS_SECRET_ACCESS_KEY=local
export AWS_DEFAULT_REGION=us-east-1

# Initialize DynamoDB table
echo -e "${GREEN}Initializing DynamoDB table...${NC}"
uv run python scripts/init-dynamodb.py
if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to initialize DynamoDB table${NC}"
    exit 1
fi

# Run the processor
echo -e "${GREEN}Running weight processor...${NC}"
echo "----------------------------------------"
uv run python main.py "$@"