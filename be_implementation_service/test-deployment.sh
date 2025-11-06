#!/bin/bash
# Weight Processor Lambda - Deployment Test Script
# Tests all operations documented in DEPLOYMENT_USAGE.md

set -e

# Configuration
STACK_NAME="weight-processor-dev"
FUNCTION_NAME="weight-processor-dev-us"
REGION="us-east-1"
TEST_USER_ID="test-user-$(date +%s)"
OUTPUT_DIR="test-results"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0

# Setup
mkdir -p "$OUTPUT_DIR"
echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}Weight Processor Deployment Test${NC}"
echo -e "${BLUE}======================================${NC}"

# Get stack outputs
echo -e "${YELLOW}Getting stack outputs...${NC}"
INVOKER_ROLE_ARN=$(aws cloudformation describe-stacks \
    --stack-name "$STACK_NAME" \
    --region "$REGION" \
    --query 'Stacks[0].Outputs[?OutputKey==`InvokerRoleArn`].OutputValue' \
    --output text 2>/dev/null || echo "")

EXTERNAL_ID=$(aws cloudformation describe-stacks \
    --stack-name "$STACK_NAME" \
    --region "$REGION" \
    --query 'Stacks[0].Outputs[?OutputKey==`ExternalId`].OutputValue' \
    --output text 2>/dev/null || echo "")

if [ -z "$INVOKER_ROLE_ARN" ] || [ -z "$EXTERNAL_ID" ]; then
    echo -e "${RED}ERROR: Could not get stack outputs. Is the stack deployed?${NC}"
    echo -e "Expected stack: ${STACK_NAME}"
    exit 1
fi

# Assume the invoker role
echo -e "${YELLOW}Assuming invoker role...${NC}"
ROLE_SESSION="weight-processor-test-$(date +%s)"
CREDENTIALS=$(aws sts assume-role \
    --role-arn "$INVOKER_ROLE_ARN" \
    --role-session-name "$ROLE_SESSION" \
    --external-id "$EXTERNAL_ID" \
    --region "$REGION" \
    --duration-seconds 3600 \
    --output json)

if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: Failed to assume role${NC}"
    echo -e "Role ARN: ${INVOKER_ROLE_ARN}"
    echo -e "External ID: ${EXTERNAL_ID}"
    echo -e ""
    echo -e "Make sure your AWS credentials have permission to assume this role:"
    echo -e "  {\"Effect\": \"Allow\", \"Action\": \"sts:AssumeRole\", \"Resource\": \"${INVOKER_ROLE_ARN}\"}"
    exit 1
fi

# Export temporary credentials
export AWS_ACCESS_KEY_ID=$(echo "$CREDENTIALS" | jq -r '.Credentials.AccessKeyId')
export AWS_SECRET_ACCESS_KEY=$(echo "$CREDENTIALS" | jq -r '.Credentials.SecretAccessKey')
export AWS_SESSION_TOKEN=$(echo "$CREDENTIALS" | jq -r '.Credentials.SessionToken')

echo -e "${GREEN}✓ Successfully assumed role${NC}"
echo -e "Function: ${FUNCTION_NAME}"
echo -e "Region: ${REGION}"
echo -e "Test User: ${TEST_USER_ID}"
echo -e "Output Dir: ${OUTPUT_DIR}"
echo ""

# Helper function to run test
run_test() {
    local test_name="$1"
    local payload="$2"
    local expected_success="$3"
    local output_file="${OUTPUT_DIR}/$(echo "$test_name" | tr ' ' '_' | tr '[:upper:]' '[:lower:]').json"

    TESTS_RUN=$((TESTS_RUN + 1))
    echo -e "${YELLOW}Test ${TESTS_RUN}: ${test_name}${NC}"

    # Invoke Lambda
    if aws lambda invoke \
        --function-name "$FUNCTION_NAME" \
        --region "$REGION" \
        --payload "$payload" \
        --cli-binary-format raw-in-base64-out \
        "$output_file" \
        --log-type Tail \
        --query 'LogResult' \
        --output text 2>/dev/null | base64 --decode > "${output_file}.log" 2>&1; then

        # Check response (parse body if it's an API Gateway response)
        local success=$(cat "$output_file" | jq -r 'if .body then (.body | fromjson | .success) else .success end // false' 2>/dev/null || echo "false")

        if [ "$success" = "$expected_success" ]; then
            TESTS_PASSED=$((TESTS_PASSED + 1))
            echo -e "${GREEN}✓ PASSED${NC}"

            # Show relevant data (parse body if it's an API Gateway response)
            if [ "$success" = "true" ]; then
                echo -e "  Response: $(cat "$output_file" | jq -c 'if .body then (.body | fromjson | .data) else .data end' 2>/dev/null || echo 'N/A')"
            else
                echo -e "  Error: $(cat "$output_file" | jq -c 'if .body then (.body | fromjson | .error.message) else .error.message end' 2>/dev/null || echo 'N/A')"
            fi
        else
            TESTS_FAILED=$((TESTS_FAILED + 1))
            echo -e "${RED}✗ FAILED${NC}"
            echo -e "  Expected success=$expected_success, got success=$success"
            echo -e "  Response: $(cat "$output_file" | jq 'if .body then (.body | fromjson) else . end' 2>/dev/null || cat "$output_file")"
        fi
    else
        TESTS_FAILED=$((TESTS_FAILED + 1))
        echo -e "${RED}✗ FAILED (Lambda invocation error)${NC}"
        cat "${output_file}.log" 2>/dev/null || echo "No logs available"
    fi

    echo ""
}

# Test 1: Health Check
run_test "Health Check" \
'{"action": "health"}' \
"true"

# Test 2: Get State (should not exist yet)
run_test "Get State - No State" \
"{\"action\": \"get_state\", \"user_id\": \"${TEST_USER_ID}\"}" \
"false"

# Test 3: Process Single Measurement
run_test "Process Single Measurement" \
"{
  \"action\": \"process\",
  \"user_id\": \"${TEST_USER_ID}\",
  \"body\": {
    \"measurements\": [{
      \"uuid\": \"measurement-001\",
      \"weight\": 185.5,
      \"unit\": \"lb\",
      \"effectiveDateTime\": \"2025-10-01T10:00:00Z\",
      \"source\": \"smart_scale\"
    }]
  }
}" \
"true"

# Test 4: Get State (should exist now)
run_test "Get State - After First Measurement" \
"{\"action\": \"get_state\", \"user_id\": \"${TEST_USER_ID}\"}" \
"true"

# Test 5: Process Multiple Measurements
run_test "Process Multiple Measurements" \
"{
  \"action\": \"process\",
  \"user_id\": \"${TEST_USER_ID}\",
  \"body\": {
    \"measurements\": [
      {
        \"uuid\": \"measurement-002\",
        \"weight\": 184.8,
        \"unit\": \"lb\",
        \"effectiveDateTime\": \"2025-10-01T11:00:00Z\",
        \"source\": \"smart_scale\"
      },
      {
        \"uuid\": \"measurement-003\",
        \"weight\": 184.5,
        \"unit\": \"lb\",
        \"effectiveDateTime\": \"2025-10-01T12:00:00Z\",
        \"source\": \"smart_scale\"
      }
    ]
  }
}" \
"true"

# Test 6: Process with Different Units
run_test "Process with Kilograms" \
"{
  \"action\": \"process\",
  \"user_id\": \"${TEST_USER_ID}\",
  \"body\": {
    \"measurements\": [{
      \"uuid\": \"measurement-004\",
      \"weight\": 84.0,
      \"unit\": \"kg\",
      \"effectiveDateTime\": \"2025-10-01T13:00:00Z\",
      \"source\": \"smart_scale\"
    }]
  }
}" \
"true"

# Test 7: Process with User Height
run_test "Process with User Height" \
"{
  \"action\": \"process\",
  \"user_id\": \"${TEST_USER_ID}\",
  \"body\": {
    \"measurements\": [{
      \"uuid\": \"measurement-005\",
      \"weight\": 83.8,
      \"unit\": \"kg\",
      \"effectiveDateTime\": \"2025-10-01T14:00:00Z\",
      \"source\": \"smart_scale\"
    }],
    \"user_height_m\": 1.75
  }
}" \
"true"

# Test 8: Invalid Unit (should fail)
run_test "Invalid Unit - Should Fail" \
"{
  \"action\": \"process\",
  \"user_id\": \"${TEST_USER_ID}\",
  \"body\": {
    \"measurements\": [{
      \"uuid\": \"measurement-invalid\",
      \"weight\": 185.5,
      \"unit\": \"pounds\",
      \"effectiveDateTime\": \"2025-10-01T15:00:00Z\",
      \"source\": \"smart_scale\"
    }]
  }
}" \
"false"

# Test 9: Missing Required Field (should fail)
run_test "Missing Required Field - Should Fail" \
"{
  \"action\": \"process\",
  \"user_id\": \"${TEST_USER_ID}\",
  \"body\": {
    \"measurements\": [{
      \"uuid\": \"measurement-invalid2\",
      \"weight\": 185.5,
      \"effectiveDateTime\": \"2025-10-01T16:00:00Z\",
      \"source\": \"smart_scale\"
    }]
  }
}" \
"false"

# Test 10: Replay Measurements
run_test "Replay Measurements" \
"{
  \"action\": \"replay\",
  \"user_id\": \"${TEST_USER_ID}\",
  \"body\": {
    \"replay_from_timestamp\": \"2025-10-01T10:00:00Z\",
    \"measurements\": [
      {
        \"uuid\": \"measurement-001\",
        \"weight\": 185.5,
        \"unit\": \"lb\",
        \"effectiveDateTime\": \"2025-10-01T10:00:00Z\",
        \"source\": \"smart_scale\"
      },
      {
        \"uuid\": \"measurement-002\",
        \"weight\": 184.8,
        \"unit\": \"lb\",
        \"effectiveDateTime\": \"2025-10-01T11:00:00Z\",
        \"source\": \"smart_scale\"
      },
      {
        \"uuid\": \"measurement-003\",
        \"weight\": 184.5,
        \"unit\": \"lb\",
        \"effectiveDateTime\": \"2025-10-01T12:00:00Z\",
        \"source\": \"smart_scale\"
      },
      {
        \"uuid\": \"measurement-004\",
        \"weight\": 84.0,
        \"unit\": \"kg\",
        \"effectiveDateTime\": \"2025-10-01T13:00:00Z\",
        \"source\": \"smart_scale\"
      },
      {
        \"uuid\": \"measurement-005\",
        \"weight\": 83.8,
        \"unit\": \"kg\",
        \"effectiveDateTime\": \"2025-10-01T14:00:00Z\",
        \"source\": \"smart_scale\"
      }
    ],
    \"options\": {
      \"validate_order\": true,
      \"stop_on_error\": false
    },
    \"user_height_m\": 1.75
  }
}" \
"true"

# Test 11: Cleanup State - Reset Adaptive
run_test "Cleanup - Reset Adaptive" \
"{
  \"action\": \"cleanup\",
  \"user_id\": \"${TEST_USER_ID}\",
  \"body\": {
    \"cleanup_type\": \"reset_adaptive\"
  }
}" \
"true"

# Test 12: Get State After Cleanup
run_test "Get State - After Cleanup" \
"{\"action\": \"get_state\", \"user_id\": \"${TEST_USER_ID}\"}" \
"true"

# Test 13: Process with Options
run_test "Process with Options" \
"{
  \"action\": \"process\",
  \"user_id\": \"${TEST_USER_ID}\",
  \"body\": {
    \"measurements\": [{
      \"uuid\": \"measurement-006\",
      \"weight\": 83.5,
      \"unit\": \"kg\",
      \"effectiveDateTime\": \"2025-10-01T15:00:00Z\",
      \"source\": \"smart_scale\"
    }],
    \"options\": {
      \"force_replay\": false,
      \"fail_on_conflict\": true,
      \"include_debug_info\": true
    }
  }
}" \
"true"

# Test 14: Process with Metadata
run_test "Process with Metadata" \
"{
  \"action\": \"process\",
  \"user_id\": \"${TEST_USER_ID}\",
  \"body\": {
    \"measurements\": [{
      \"uuid\": \"measurement-007\",
      \"weight\": 83.4,
      \"unit\": \"kg\",
      \"effectiveDateTime\": \"2025-10-01T16:00:00Z\",
      \"source\": \"smart_scale\",
      \"metadata\": {
        \"device_id\": \"scale-123\",
        \"battery_level\": 85
      }
    }]
  }
}" \
"true"

# Test 15: Delete State
run_test "Delete State" \
"{\"action\": \"delete_state\", \"user_id\": \"${TEST_USER_ID}\"}" \
"true"

# Test 16: Get State After Delete (should not exist)
run_test "Get State - After Delete" \
"{\"action\": \"get_state\", \"user_id\": \"${TEST_USER_ID}\"}" \
"false"

# Test 17: Process with Different Sources
run_test "Process with Different Sources" \
"{
  \"action\": \"process\",
  \"user_id\": \"${TEST_USER_ID}-sources\",
  \"body\": {
    \"measurements\": [
      {
        \"uuid\": \"measurement-s1\",
        \"weight\": 185.5,
        \"unit\": \"lb\",
        \"effectiveDateTime\": \"2025-10-01T10:00:00Z\",
        \"source\": \"smart_scale\"
      },
      {
        \"uuid\": \"measurement-s2\",
        \"weight\": 185.2,
        \"unit\": \"lb\",
        \"effectiveDateTime\": \"2025-10-01T11:00:00Z\",
        \"source\": \"manual_entry\"
      },
      {
        \"uuid\": \"measurement-s3\",
        \"weight\": 184.8,
        \"unit\": \"lb\",
        \"effectiveDateTime\": \"2025-10-01T12:00:00Z\",
        \"source\": \"questionnaire\"
      }
    ]
  }
}" \
"true"

# Test 18: Cleanup - Clear All
run_test "Cleanup - Clear All" \
"{
  \"action\": \"cleanup\",
  \"user_id\": \"${TEST_USER_ID}-sources\",
  \"body\": {
    \"cleanup_type\": \"clear_all\"
  }
}" \
"true"

# Cleanup temporary credentials
unset AWS_ACCESS_KEY_ID
unset AWS_SECRET_ACCESS_KEY
unset AWS_SESSION_TOKEN

# Summary
echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}Test Summary${NC}"
echo -e "${BLUE}======================================${NC}"
echo -e "Total Tests: ${TESTS_RUN}"
echo -e "${GREEN}Passed: ${TESTS_PASSED}${NC}"
echo -e "${RED}Failed: ${TESTS_FAILED}${NC}"
echo -e "Results saved to: ${OUTPUT_DIR}/"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}✗ Some tests failed. Check ${OUTPUT_DIR}/ for details.${NC}"
    exit 1
fi
