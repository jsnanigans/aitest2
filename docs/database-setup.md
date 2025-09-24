# Database Configuration for AWS Deployment

## Overview

The weight processor supports both SQLite (for simple local development) and DynamoDB (for production AWS deployment and local development with Docker).

## Database Backends

### 1. SQLite (Simple Local Development)
- Default for local development without Docker
- Data stored in `/tmp/weight-processor.db`
- No setup required
- Single-user only

### 2. DynamoDB Local (Recommended for Development)
- Runs in Docker container
- Mimics production DynamoDB behavior
- Supports concurrent users
- Data persists between runs

### 3. AWS DynamoDB (Production)
- Fully managed AWS service
- Automatic scaling
- High availability
- Pay-per-request pricing

## Local Development Setup

### Using DynamoDB Local (Recommended)

1. **Start DynamoDB Local:**
```bash
# Start DynamoDB Local and Admin UI
docker-compose up -d dynamodb-local dynamodb-admin

# Or use the convenience script
./scripts/run-local.sh data/weights.csv
```

2. **View data in Admin UI:**
- Open http://localhost:8001 in your browser
- You can view and query all tables and data

3. **Environment variables (`.env.local`):**
```bash
DB_BACKEND=dynamodb
DYNAMODB_ENDPOINT=http://localhost:8000
DYNAMODB_TABLE_NAME=weight-processor-state
AWS_ACCESS_KEY_ID=local
AWS_SECRET_ACCESS_KEY=local
```

### Using SQLite (Simple)

1. **Set environment variable:**
```bash
export DB_BACKEND=sqlite
```

2. **Run normally:**
```bash
uv run python main.py data/weights.csv
```

## AWS Production Setup

### Prerequisites

1. **AWS Account** with appropriate permissions
2. **AWS CLI** configured: `aws configure`
3. **DynamoDB table** (created automatically on first run)

### Running in Production

```bash
# Use the convenience script
./scripts/run-aws.sh data/weights.csv

# Or set environment manually
export DB_BACKEND=dynamodb
export DYNAMODB_TABLE_NAME=weight-processor-state
unset DYNAMODB_ENDPOINT  # Important: clear local endpoint
uv run python main.py data/weights.csv
```

### AWS Lambda Deployment

The database configuration is automatic in Lambda:

1. **Environment variables in Lambda:**
```bash
DB_BACKEND=dynamodb
DYNAMODB_TABLE_NAME=weight-processor-state
```

2. **IAM Role permissions required:**
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

## Database Schema

### DynamoDB Table Structure

- **Table Name:** `weight-processor-state`
- **Partition Key:** `userId` (String)
- **Sort Key:** `stateType` (String)
- **Billing Mode:** PAY_PER_REQUEST (on-demand)

### State Types

- `current` - Current user state
- `snapshot_<timestamp>` - State snapshots for replay functionality

## Switching Between Environments

### Local Development → Production

1. Commit and push your code
2. Deploy to Lambda (see deployment guide)
3. Ensure Lambda has proper IAM permissions
4. Set Lambda environment variables

### Production → Local Development

1. Export production data (optional):
```bash
# Export current states to CSV
./scripts/run-aws.sh data/weights.csv --export-only
```

2. Start local DynamoDB:
```bash
docker-compose up -d dynamodb-local
```

3. Run with local configuration:
```bash
./scripts/run-local.sh data/weights.csv
```

## Troubleshooting

### DynamoDB Local Issues

**Container won't start:**
```bash
# Check if port 8000 is in use
lsof -i :8000

# Stop and restart
docker-compose down
docker-compose up -d dynamodb-local
```

**Connection refused:**
```bash
# Verify container is running
docker ps | grep dynamodb

# Check logs
docker-compose logs dynamodb-local
```

### AWS DynamoDB Issues

**Access Denied:**
- Verify IAM permissions
- Check AWS credentials: `aws sts get-caller-identity`
- Ensure correct region: `echo $AWS_REGION`

**Table not found:**
- Table is created automatically on first use
- Check table name matches environment variable
- Verify region is correct

### Data Migration

**Export from SQLite to DynamoDB:**
```python
# Use migration script
uv run python scripts/migrate_sqlite_to_dynamodb.py
```

**Backup DynamoDB data:**
```bash
# Use AWS Backup or DynamoDB export
aws dynamodb create-backup \
  --table-name weight-processor-state \
  --backup-name weight-processor-backup-$(date +%Y%m%d)
```

## Performance Considerations

### DynamoDB Optimization

1. **Use batch operations** for bulk processing
2. **Enable DynamoDB Accelerator (DAX)** for caching in production
3. **Monitor with CloudWatch** for throttling and latency

### Lambda Optimization

1. **Connection reuse:** Database connections are reused across invocations
2. **Cold starts:** First invocation creates table if needed
3. **Memory allocation:** 512MB+ recommended for optimal performance

## Cost Optimization

### DynamoDB Pricing (On-Demand)

- **Read:** $0.25 per million read request units
- **Write:** $1.25 per million write request units
- **Storage:** $0.25 per GB-month

### Recommendations

1. Use **on-demand billing** for unpredictable workloads
2. Consider **provisioned capacity** for steady workloads
3. Enable **point-in-time recovery** for production
4. Set up **auto-scaling** for provisioned capacity