#!/bin/bash
# Script to run the weight processor with AWS DynamoDB (Production)

set -e

echo "Running with AWS DynamoDB (Production)..."

# Ensure AWS credentials are configured
if ! aws sts get-caller-identity > /dev/null 2>&1; then
    echo "Error: AWS credentials not configured. Please run 'aws configure' first."
    exit 1
fi

# Check if boto3 is installed
if ! uv run python -c "import boto3" 2>/dev/null; then
    echo "Installing boto3..."
    uv pip install boto3
fi

# Set environment variables for AWS
export DYNAMODB_TABLE_NAME=${DYNAMODB_TABLE_NAME:-weight-processor-state}
export AWS_REGION=${AWS_REGION:-us-east-1}
unset DYNAMODB_ENDPOINT  # Clear local endpoint to use AWS

echo ""
echo "Environment configured:"
echo "  DYNAMODB_TABLE_NAME=$DYNAMODB_TABLE_NAME"
echo "  AWS_REGION=$AWS_REGION"
echo "  Using AWS DynamoDB (production)"
echo ""

# Verify DynamoDB access
echo "Verifying AWS DynamoDB access..."
if aws dynamodb describe-limits --region $AWS_REGION > /dev/null 2>&1; then
    echo "✓ AWS DynamoDB access confirmed"
else
    echo "Warning: Could not verify DynamoDB access. Table will be created if needed."
fi

# Run the main script with arguments
if [ $# -eq 0 ]; then
    echo "Usage: $0 <csv_file> [options]"
    echo "Example: $0 data/weights.csv --max-users 10"
    exit 1
fi

echo "Running weight processor..."
uv run python main.py "$@"