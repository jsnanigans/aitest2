#!/bin/bash
# LocalStack initialization script
# This runs when LocalStack container starts

set -e

echo "Initializing LocalStack..."

# Wait for LocalStack to be ready
sleep 2

# Create DynamoDB table
echo "Creating DynamoDB table..."
awslocal dynamodb create-table \
    --table-name weight-processor-state \
    --attribute-definitions \
        AttributeName=userId,AttributeType=S \
        AttributeName=stateType,AttributeType=S \
    --key-schema \
        AttributeName=userId,KeyType=HASH \
        AttributeName=stateType,KeyType=RANGE \
    --billing-mode PAY_PER_REQUEST \
    --region us-east-1 || echo "Table already exists"

# Create S3 bucket for Lambda deployments (if needed)
echo "Creating S3 bucket..."
awslocal s3 mb s3://weight-processor-lambda-deployments || echo "Bucket already exists"

# Create Lambda function placeholder
echo "Creating Lambda function..."
awslocal lambda create-function \
    --function-name weight-processor \
    --runtime python3.11 \
    --role arn:aws:iam::000000000000:role/lambda-role \
    --handler src.lambda_handler.handler \
    --zip-file fileb:///dev/null \
    --environment Variables="{DB_BACKEND=dynamodb,DYNAMODB_TABLE_NAME=weight-processor-state}" \
    --timeout 60 \
    --memory-size 1024 || echo "Function already exists"

# Create API Gateway
echo "Creating API Gateway..."
API_ID=$(awslocal apigateway create-rest-api \
    --name weight-processor-api \
    --description "Weight Processor API" \
    --query 'id' \
    --output text || echo "existing")

if [ "$API_ID" != "existing" ]; then
    echo "API Gateway created with ID: $API_ID"

    # Get root resource ID
    ROOT_ID=$(awslocal apigateway get-resources \
        --rest-api-id $API_ID \
        --query 'items[?path==`/`].id' \
        --output text)

    # Create /api resource
    API_RESOURCE=$(awslocal apigateway create-resource \
        --rest-api-id $API_ID \
        --parent-id $ROOT_ID \
        --path-part api \
        --query 'id' \
        --output text)

    # Create /api/v1 resource
    V1_RESOURCE=$(awslocal apigateway create-resource \
        --rest-api-id $API_ID \
        --parent-id $API_RESOURCE \
        --path-part v1 \
        --query 'id' \
        --output text)

    # Create /api/v1/process resource
    PROCESS_RESOURCE=$(awslocal apigateway create-resource \
        --rest-api-id $API_ID \
        --parent-id $V1_RESOURCE \
        --path-part process \
        --query 'id' \
        --output text)

    # Create /api/v1/process/{userId} resource
    PROCESS_USER_RESOURCE=$(awslocal apigateway create-resource \
        --rest-api-id $API_ID \
        --parent-id $PROCESS_RESOURCE \
        --path-part '{userId}' \
        --query 'id' \
        --output text)

    # Create POST method for /api/v1/process/{userId}
    awslocal apigateway put-method \
        --rest-api-id $API_ID \
        --resource-id $PROCESS_USER_RESOURCE \
        --http-method POST \
        --authorization-type NONE

    # Create Lambda integration
    awslocal apigateway put-integration \
        --rest-api-id $API_ID \
        --resource-id $PROCESS_USER_RESOURCE \
        --http-method POST \
        --type AWS_PROXY \
        --integration-http-method POST \
        --uri "arn:aws:apigateway:us-east-1:lambda:path/2015-03-31/functions/arn:aws:lambda:us-east-1:000000000000:function:weight-processor/invocations"

    # Deploy API
    awslocal apigateway create-deployment \
        --rest-api-id $API_ID \
        --stage-name local

    echo "API Gateway configured successfully"
    echo "API available at: http://localhost:4566/restapis/$API_ID/local/_user_request_/api/v1"
fi

# Seed some test data
echo "Seeding test data..."
awslocal dynamodb put-item \
    --table-name weight-processor-state \
    --item '{
        "userId": {"S": "test-user-001"},
        "stateType": {"S": "current"},
        "lastTimestamp": {"S": "2024-01-01T00:00:00Z"},
        "lastRawWeight": {"N": "75.0"},
        "measurementCount": {"N": "10"}
    }' || echo "Test data already exists"

echo "LocalStack initialization complete!"
echo ""
echo "Services available at:"
echo "  DynamoDB: http://localhost:4566"
echo "  Lambda:   http://localhost:4566"
echo "  API Gateway: http://localhost:4566/restapis/$API_ID/local/_user_request_"
echo ""
echo "DynamoDB Admin UI: http://localhost:8001"