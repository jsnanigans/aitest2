#!/bin/bash
# Test Lambda function locally using SAM

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Testing Lambda Function Locally${NC}"
echo "================================"

# Check for SAM CLI
if ! command -v sam &> /dev/null; then
    echo -e "${RED}Error: SAM CLI is not installed${NC}"
    echo "Install it with: pip install aws-sam-cli"
    exit 1
fi

# Build if needed
if [ ! -d ".aws-sam" ]; then
    echo -e "${YELLOW}Building Lambda package...${NC}"
    sam build --use-container
fi

# Create test event if it doesn't exist
TEST_EVENT_FILE="test_events/process_measurement.json"
if [ ! -f "$TEST_EVENT_FILE" ]; then
    echo -e "${YELLOW}Creating test event...${NC}"
    mkdir -p test_events

    cat > "$TEST_EVENT_FILE" << 'EOF'
{
  "resource": "/api/v1/process/{userId}",
  "path": "/api/v1/process/test-user-123",
  "httpMethod": "POST",
  "headers": {
    "Content-Type": "application/json",
    "x-api-key": "test-key"
  },
  "pathParameters": {
    "userId": "test-user-123"
  },
  "body": "{\"measurements\":[{\"uuid\":\"550e8400-e29b-41d4-a716-446655440000\",\"weight\":75.5,\"unit\":\"kg\",\"effectiveDateTime\":\"2024-01-15T10:30:00Z\",\"source\":\"patient-device\"},{\"uuid\":\"550e8400-e29b-41d4-a716-446655440001\",\"weight\":75.8,\"unit\":\"kg\",\"effectiveDateTime\":\"2024-01-16T10:30:00Z\",\"source\":\"patient-device\"}]}"
}
EOF
    echo "Created test event at: $TEST_EVENT_FILE"
fi

# Start local API
echo ""
echo -e "${YELLOW}Starting local API...${NC}"
echo "API will be available at: http://localhost:3080"
echo ""

# Set environment variables for local testing
export DB_BACKEND=memory
export LOG_LEVEL=INFO

# Start SAM local API
sam local start-api \
    --env-vars env.json \
    --docker-network host \
    --warm-containers EAGER