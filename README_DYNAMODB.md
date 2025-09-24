# Weight Processor - DynamoDB Setup

## Overview

This weight processor uses **DynamoDB exclusively** for data storage to ensure consistency between local development and AWS production environments.

## Prerequisites

1. **Docker** - Required for local development (to run DynamoDB Local)
2. **Python with uv** - Package manager
3. **AWS CLI** - For production deployment
4. **boto3** - AWS SDK for Python (installed automatically)

## Quick Start

### Local Development

```bash
# 1. Start DynamoDB Local (required)
docker-compose up -d dynamodb-local dynamodb-admin

# 2. Run the processor
./scripts/run-local.sh data/weights.csv

# 3. View data in DynamoDB Admin UI
open http://localhost:8001
```

### Production (AWS)

```bash
# 1. Configure AWS credentials
aws configure

# 2. Run the processor
./scripts/run-aws.sh data/weights.csv
```

## Database Architecture

### Why DynamoDB Only?

- **Consistency**: Same database in development and production
- **Scalability**: Auto-scaling for production workloads
- **Serverless**: Perfect for Lambda deployments
- **No SQLite**: Eliminates file-system dependencies and concurrency issues

### Table Structure

- **Table Name**: `weight-processor-state`
- **Partition Key**: `userId` (String)
- **Sort Key**: `stateType` (String)
- **Billing**: Pay-per-request (on-demand)

### State Types

- `current` - Active user state
- `snapshot_TIMESTAMP` - Point-in-time snapshots for replay

## Environment Configuration

### Local Development (.env.local)

```bash
# DynamoDB Local configuration
DYNAMODB_ENDPOINT=http://localhost:8000
DYNAMODB_TABLE_NAME=weight-processor-state
AWS_ACCESS_KEY_ID=local
AWS_SECRET_ACCESS_KEY=local
AWS_DEFAULT_REGION=us-east-1
```

### AWS Lambda

Set these environment variables in your Lambda function:

```bash
DYNAMODB_TABLE_NAME=weight-processor-state
AWS_REGION=us-east-1
```

## Docker Services

### Start Services

```bash
# Start DynamoDB Local and Admin UI
docker-compose up -d dynamodb-local dynamodb-admin

# View logs
docker-compose logs -f dynamodb-local

# Stop services
docker-compose down
```

### Available Services

- **DynamoDB Local**: http://localhost:8000
- **DynamoDB Admin**: http://localhost:8001

## Development Workflow

### 1. Initial Setup

```bash
# Clone repository
git clone <repository>
cd weight-processor

# Install dependencies
uv pip install boto3

# Start DynamoDB Local
docker-compose up -d dynamodb-local
```

### 2. Run Processing

```bash
# Use convenience script (starts DynamoDB automatically)
./scripts/run-local.sh data/weights.csv --max-users 10

# Or manually with environment variables
export DYNAMODB_ENDPOINT=http://localhost:8000
export DYNAMODB_TABLE_NAME=weight-processor-state
uv run python main.py data/weights.csv
```

### 3. View Data

Open http://localhost:8001 to see:
- Table structure
- User states
- Query data

## Production Deployment

### AWS Lambda Setup

1. **Create Lambda function**
```bash
aws lambda create-function \
  --function-name weight-processor \
  --runtime python3.11 \
  --handler lambda_handler.lambda_handler \
  --memory-size 512 \
  --timeout 300
```

2. **Set environment variables**
```bash
aws lambda update-function-configuration \
  --function-name weight-processor \
  --environment Variables="{DYNAMODB_TABLE_NAME=weight-processor-state}"
```

3. **Add IAM permissions**
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "dynamodb:GetItem",
        "dynamodb:PutItem",
        "dynamodb:DeleteItem",
        "dynamodb:Query",
        "dynamodb:Scan",
        "dynamodb:CreateTable",
        "dynamodb:DescribeTable"
      ],
      "Resource": "arn:aws:dynamodb:*:*:table/weight-processor-state*"
    }
  ]
}
```

### Direct AWS Execution

```bash
# Ensure AWS credentials are configured
aws configure

# Run with production DynamoDB
./scripts/run-aws.sh data/weights.csv
```

## Troubleshooting

### DynamoDB Local Won't Start

```bash
# Check if port 8000 is in use
lsof -i :8000

# Restart Docker services
docker-compose down
docker-compose up -d dynamodb-local

# Check logs
docker-compose logs dynamodb-local
```

### boto3 Not Found

```bash
# Install boto3
uv pip install boto3

# Or use pip directly
pip install boto3
```

### AWS Access Denied

1. Check IAM permissions
2. Verify AWS credentials: `aws sts get-caller-identity`
3. Ensure correct region: `echo $AWS_REGION`

### Table Not Created

The table is created automatically on first use. If it fails:

```bash
# Create manually with AWS CLI
aws dynamodb create-table \
  --table-name weight-processor-state \
  --attribute-definitions \
    AttributeName=userId,AttributeType=S \
    AttributeName=stateType,AttributeType=S \
  --key-schema \
    AttributeName=userId,KeyType=HASH \
    AttributeName=stateType,KeyType=RANGE \
  --billing-mode PAY_PER_REQUEST
```

## Performance Tips

### Local Development

- DynamoDB Local stores data in `/data` volume
- Data persists between container restarts
- Use Admin UI to monitor queries

### Production

- Table auto-scales with on-demand billing
- Consider provisioned capacity for predictable workloads
- Monitor with CloudWatch metrics

## Cost Optimization

### DynamoDB Pricing (On-Demand)

- **Reads**: $0.25 per million request units
- **Writes**: $1.25 per million request units
- **Storage**: $0.25 per GB-month

### Estimates

- 1000 users, 100 measurements each: ~$0.10/month
- 10,000 users, 1000 measurements each: ~$10/month

## Testing

```bash
# Run with test data
./scripts/run-local.sh tests/data/test_weights.csv

# Verify in Admin UI
open http://localhost:8001

# Query specific user
aws dynamodb query \
  --endpoint-url http://localhost:8000 \
  --table-name weight-processor-state \
  --key-condition-expression "userId = :uid" \
  --expression-attribute-values '{":uid":{"S":"user123"}}'
```

## Migration from SQLite

If you have existing SQLite data:

1. Export from SQLite to CSV
2. Process CSV with DynamoDB backend
3. Data will be imported automatically

## Support

- View data: http://localhost:8001
- Check logs: `docker-compose logs -f`
- AWS Status: https://status.aws.amazon.com