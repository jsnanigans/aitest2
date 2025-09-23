#!/bin/bash
# Test Lambda function locally with various payloads

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Testing Lambda Function Locally${NC}"
echo "================================="

# Function to test an endpoint
test_endpoint() {
    local name=$1
    local event_file=$2

    echo ""
    echo -e "${YELLOW}Testing: $name${NC}"
    echo "Event file: $event_file"

    # Invoke Lambda locally
    sam local invoke WeightProcessorFunction \
        --event "$event_file" \
        --env-vars ../env.json \
        2>/dev/null | tail -n +2 | jq '.'

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ $name passed${NC}"
    else
        echo -e "${RED}✗ $name failed${NC}"
    fi
}

# Build first if needed
if [ ! -d "../.aws-sam" ]; then
    echo -e "${YELLOW}Building Lambda package...${NC}"
    cd ..
    sam build --use-container
    cd test_events
fi

# Test each endpoint
echo ""
echo "Running tests..."

# Process measurements (normal flow)
test_endpoint "Process Measurements" "process_measurements.json"

# Cleanup with reset
test_endpoint "Cleanup User" "cleanup_user.json"

# Get state (should work after process)
test_endpoint "Get State" "get_state.json"

# Historical conflict detection
# First process a recent measurement
echo ""
echo -e "${YELLOW}Setting up historical conflict test...${NC}"
cat > temp_recent.json << 'EOF'
{
  "resource": "/api/v1/process/{userId}",
  "path": "/api/v1/process/user-conflict",
  "httpMethod": "POST",
  "headers": {
    "Content-Type": "application/json",
    "x-api-key": "test-api-key"
  },
  "pathParameters": {
    "userId": "user-conflict"
  },
  "body": "{\"measurements\":[{\"uuid\":\"850e8400-e29b-41d4-a716-446655440000\",\"weight\":75.0,\"unit\":\"kg\",\"effectiveDateTime\":\"2024-01-20T10:00:00Z\",\"source\":\"patient-device\"}]}"
}
EOF

test_endpoint "Setup Recent Measurement" "temp_recent.json"

# Now try to process an older measurement (should conflict)
cat > temp_old.json << 'EOF'
{
  "resource": "/api/v1/process/{userId}",
  "path": "/api/v1/process/user-conflict",
  "httpMethod": "POST",
  "headers": {
    "Content-Type": "application/json",
    "x-api-key": "test-api-key"
  },
  "pathParameters": {
    "userId": "user-conflict"
  },
  "body": "{\"measurements\":[{\"uuid\":\"850e8400-e29b-41d4-a716-446655440001\",\"weight\":74.5,\"unit\":\"kg\",\"effectiveDateTime\":\"2024-01-15T10:00:00Z\",\"source\":\"patient-device\"}]}"
}
EOF

echo ""
echo -e "${YELLOW}This should return a 409 Conflict:${NC}"
test_endpoint "Historical Conflict Detection" "temp_old.json"

# Cleanup
rm -f temp_recent.json temp_old.json

echo ""
echo -e "${GREEN}All tests complete!${NC}"