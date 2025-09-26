#!/bin/bash

# Weight Processor API Test Runner
# Ensures environment is configured and runs comprehensive tests

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DYNAMODB_PORT=8000
API_PORT=3080
BASE_URL="http://localhost:${API_PORT}"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Weight Processor API Test Runner${NC}"
echo -e "${BLUE}========================================${NC}"

# Function to check if a service is running
check_service() {
    local port=$1
    local service=$2

    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null ; then
        echo -e "${GREEN}✓${NC} $service is running on port $port"
        return 0
    else
        echo -e "${RED}✗${NC} $service is not running on port $port"
        return 1
    fi
}

# Function to wait for service
wait_for_service() {
    local port=$1
    local service=$2
    local max_attempts=30
    local attempt=0

    echo -e "${YELLOW}Waiting for $service on port $port...${NC}"

    while [ $attempt -lt $max_attempts ]; do
        if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null ; then
            echo -e "${GREEN}✓${NC} $service is ready"
            return 0
        fi
        attempt=$((attempt+1))
        sleep 1
    done

    echo -e "${RED}✗${NC} Timeout waiting for $service"
    return 1
}

# Check prerequisites
echo -e "\n${BLUE}Checking prerequisites...${NC}"

# Check if Python is installed
if command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
elif command -v python &> /dev/null; then
    PYTHON_CMD=python
else
    echo -e "${RED}✗${NC} Python is not installed"
    exit 1
fi
echo -e "${GREEN}✓${NC} Python found: $PYTHON_CMD"

# Check if requests library is installed
if $PYTHON_CMD -c "import requests" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} Python requests library is installed"
else
    echo -e "${YELLOW}!${NC} Installing requests library..."
    if command -v uv &> /dev/null; then
        uv pip install requests
    else
        $PYTHON_CMD -m pip install requests
    fi
fi

# Set environment variables
export DYNAMODB_ENDPOINT="http://localhost:${DYNAMODB_PORT}"
export AWS_REGION=us-east-1
export AWS_ACCESS_KEY_ID=test
export AWS_SECRET_ACCESS_KEY=test

echo -e "\n${BLUE}Environment Configuration:${NC}"
echo "  DYNAMODB_ENDPOINT: $DYNAMODB_ENDPOINT"
echo "  AWS_REGION: $AWS_REGION"
echo "  API_BASE_URL: $BASE_URL"

# Check services
echo -e "\n${BLUE}Checking services...${NC}"

# Check DynamoDB
if ! check_service $DYNAMODB_PORT "DynamoDB Local"; then
    echo -e "${YELLOW}!${NC} Attempting to start DynamoDB Local..."
    # Check if docker is available
    if command -v docker &> /dev/null; then
        # Try to start DynamoDB using docker
        docker run -d -p ${DYNAMODB_PORT}:8000 \
            --name dynamodb-local \
            amazon/dynamodb-local \
            -jar DynamoDBLocal.jar -sharedDb -inMemory \
            2>/dev/null || echo -e "${YELLOW}!${NC} DynamoDB container may already exist"

        wait_for_service $DYNAMODB_PORT "DynamoDB Local"
    else
        echo -e "${RED}✗${NC} Docker not found. Please start DynamoDB Local manually on port $DYNAMODB_PORT"
        echo "   You can run: docker run -p 8000:8000 amazon/dynamodb-local"
        exit 1
    fi
fi

# Check SAM Local API
if ! check_service $API_PORT "SAM Local API"; then
    echo -e "${RED}✗${NC} SAM Local API is not running on port $API_PORT"
    echo -e "${YELLOW}!${NC} Please start the SAM Local API in another terminal:"
    echo "   sam local start-api --port $API_PORT"
    echo ""
    read -p "Press Enter once the API is running, or Ctrl+C to exit..."

    if ! check_service $API_PORT "SAM Local API"; then
        echo -e "${RED}✗${NC} SAM Local API still not detected"
        exit 1
    fi
fi

# Run tests
echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}Running Test Suite${NC}"
echo -e "${BLUE}========================================${NC}\n"

# Run the Python test script
$PYTHON_CMD test_lambda_api.py --base-url "$BASE_URL" "$@"

TEST_EXIT_CODE=$?

# Print final status
echo ""
if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}All tests passed successfully!${NC}"
    echo -e "${GREEN}========================================${NC}"
else
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}Some tests failed. Check the output above.${NC}"
    echo -e "${RED}========================================${NC}"
fi

exit $TEST_EXIT_CODE