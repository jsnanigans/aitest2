# Weight Processor Service

A hosted service for processing weight measurements using advanced Kalman filtering and statistical analysis. The service provides both local development environments and AWS Lambda deployment options for scalable, production-ready weight data processing.

## 🚀 Quick Start

Get up and running in under 2 minutes:

```bash
make setup           # Install dependencies

# Start services
make docker-up       # Start Docker containers
make db-reset        # Initialize database

# Run the API
make sam-local       # Start API on http://localhost:3000

# Test it works
make test-api        # Test health endpoint
```

## Detailed Setup Options

### Prerequisites

- **Docker & Docker Compose**: For containerized development
- **AWS SAM CLI**: For serverless local testing and deployment
- **Python 3.12+**: For local development
- **uv**: Python package manager (`pip install uv`)
- **AWS Account**: For production deployment (optional for local dev)

### Option 1: Docker Development (Recommended)

Complete isolated environment with all AWS services emulated locally:

```bash
# 1. Initial setup
make setup           # Install Python dependencies

# 2. Start Docker services
make docker-up       # Start LocalStack, DynamoDB, Admin UI

# 3. Initialize database
make db-reset        # Create and initialize DynamoDB tables

# 4. Verify services
make docker-status   # Check container status

# Access services:
# - DynamoDB Admin UI: http://localhost:8001
# - LocalStack Dashboard: http://localhost:4566
# - DynamoDB endpoint: http://localhost:8000
```

### Option 2: Local Development with Make

Quick development workflow using Make commands:

```bash
# 1. Initial setup
make setup           # Install all dependencies

# 2. Start database
make db-start        # Start DynamoDB Local (requires Docker)

# 3. Run processor
make run             # Run with sample data
make run-file FILE=data/sample_weights.csv  # Run with specific file

# 4. Manage database
make db-admin        # Open DynamoDB Admin UI
make db-reset        # Reset database tables
make db-stop         # Stop DynamoDB
```

### Option 3: AWS SAM Local Testing

Test the Lambda function locally using SAM:

```bash
# 1. Prerequisites
# Install AWS SAM CLI if needed:
# macOS: brew install aws-sam-cli
# Others: https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/install-sam-cli.html

# 2. Start local API with one command
make sam-local       # Builds and starts API on port 3000
                    # (automatically starts DynamoDB if needed)

# 3. Test the API
make test-api        # Test health endpoint
make test-process    # Test process endpoint

# Or test manually:
curl -X POST http://localhost:3000/process \
  -H "Content-Type: application/json" \
  -d '{
    "device_id": "scale-001",
    "measurements": [
      {"weight": 75.5, "timestamp": "2024-01-01T10:00:00Z", "source": "patient-device"}
    ]
  }'

# 4. View logs
make sam-logs        # View local Lambda logs

# 5. Invoke function directly
make sam-invoke FUNC=WeightProcessorFunction
```

### Option 4: AWS Deployment with SAM

Deploy to AWS Lambda and API Gateway:

```bash
# 1. Configure AWS credentials (first time only)
aws configure
# Enter: Access Key ID, Secret Key, Region (us-east-1), Output format (json)

# 2. Interactive deployment (first time)
make sam-deploy      # Guided deployment with prompts

# 3. Deploy to specific environment
make sam-deploy-env ENV=dev      # Development
make sam-deploy-env ENV=staging  # Staging
make sam-deploy-env ENV=prod     # Production

# 4. View deployed stack logs
make sam-logs STACK=weight-processor-dev

# 5. Get API endpoint URL
aws cloudformation describe-stacks \
  --stack-name weight-processor-dev \
  --query 'Stacks[0].Outputs[?OutputKey==`ApiUrl`].OutputValue' \
  --output text

# 6. Test deployed API
API_URL=$(aws cloudformation describe-stacks \
  --stack-name weight-processor-dev \
  --query 'Stacks[0].Outputs[?OutputKey==`ApiUrl`].OutputValue' \
  --output text)

curl -X POST $API_URL/process \
  -H "Content-Type: application/json" \
  -d '{"device_id": "scale-001", "measurements": [...]}'

# 7. Delete stack (cleanup)
make sam-delete STACK=weight-processor-dev
```

## Production Deployment

### Environment Configuration

SAM supports multiple deployment environments via `samconfig.toml`:

```toml
# aws/samconfig.toml
version = 0.1

[default.deploy.parameters]
stack_name = "weight-processor-dev"
parameter_overrides = "Environment=dev"

[staging.deploy.parameters]
stack_name = "weight-processor-staging"
parameter_overrides = "Environment=staging"

[prod.deploy.parameters]
stack_name = "weight-processor-prod"
parameter_overrides = "Environment=prod"
```

### Monitoring & Observability

**CloudWatch Metrics:**
```bash
# View Lambda metrics
aws cloudwatch get-metric-statistics \
  --namespace AWS/Lambda \
  --metric-name Duration \
  --dimensions Name=FunctionName,Value=weight-processor-dev \
  --start-time 2024-01-01T00:00:00Z \
  --end-time 2024-01-02T00:00:00Z \
  --period 3600 \
  --statistics Average,Maximum
```

**CloudWatch Logs:**
```bash
# Stream logs in real-time
sam logs -n WeightProcessorFunction \
  --stack-name weight-processor-prod \
  --tail

# Query logs with CloudWatch Insights
aws logs start-query \
  --log-group-name /aws/lambda/weight-processor-prod \
  --start-time $(date -u -d '1 hour ago' +%s) \
  --end-time $(date +%s) \
  --query-string 'fields @timestamp, @message | filter @message like /ERROR/'
```

**X-Ray Tracing:**
```bash
# Enable tracing in template.yaml
# Globals > Function > Tracing: Active

# View traces
aws xray get-trace-summaries \
  --time-range-type LastHour \
  --query 'TraceSummaries[?ServiceNames[?contains(@, `weight-processor`)]]'
```

### Performance Optimization

**Lambda Configuration:**
```yaml
# aws/template.yaml
Properties:
  MemorySize: 1024  # Adjust based on profiling
  Timeout: 30       # Maximum execution time
  ReservedConcurrentExecutions: 100  # Limit concurrent executions
  Environment:
    Variables:
      PYTHONPATH: /var/runtime:/var/task
      PYTHONDONTWRITEBYTECODE: 1
```

**DynamoDB Optimization:**
```yaml
# Configure read/write capacity
BillingMode: PAY_PER_REQUEST  # Or PROVISIONED
# For PROVISIONED:
ProvisionedThroughput:
  ReadCapacityUnits: 5
  WriteCapacityUnits: 5
```

### Security Best Practices

**IAM Policies:**
```yaml
# Minimal permissions in template.yaml
Policies:
  - DynamoDBCrudPolicy:
      TableName: !Ref StateTable
  - CloudWatchPutMetricPolicy: {}
```

**Environment Variables:**
```bash
# Use AWS Systems Manager Parameter Store
aws ssm put-parameter \
  --name /weight-processor/prod/api-key \
  --value "secret-value" \
  --type SecureString

# Reference in template.yaml
Environment:
  Variables:
    API_KEY: !Sub '{{resolve:ssm:/weight-processor/${Environment}/api-key}}'
```

### CI/CD Pipeline

**GitHub Actions Example:**
```yaml
# .github/workflows/deploy.yml
name: Deploy to AWS
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: aws-actions/setup-sam@v2
      - run: sam build
      - run: sam deploy --no-confirm-changeset --no-fail-on-empty-changeset
```

## Architecture

### Docker Services

The `docker-compose.yml` provides essential local AWS services:

- **LocalStack**: AWS service emulation for S3, CloudWatch, and more (port 4566)
- **DynamoDB Local**: Standalone DynamoDB database (port 8000)
- **DynamoDB Admin UI**: Web interface for viewing/managing data (port 8001)

Note: Lambda functions are tested using SAM Local (`make sam-local`) rather than Docker containers for better compatibility with AWS deployment.

### SAM Configuration

The service uses AWS SAM (`aws/template.yaml`) for serverless deployment:

- **API Gateway**: RESTful API endpoints for weight processing
- **Lambda Function**: Python 3.12 runtime with optimized dependencies
- **DynamoDB**: State persistence for Kalman filters and processing history
- **CloudWatch**: Logging and monitoring

### Available Make Commands

Complete list of Make commands for development and deployment:

```bash
# Quick Start
make help            # Show all available commands
make setup           # Install dependencies
make info            # Show environment information

# Docker Management
make docker-up       # Start all Docker services
make docker-down     # Stop Docker services
make docker-restart  # Restart Docker services
make docker-status   # Show container status
make docker-logs     # View container logs
make docker-clean    # Remove containers and volumes
make docker-shell    # Open shell in container

# Database Management
make db-start        # Start DynamoDB Local
make db-stop         # Stop DynamoDB Local
make db-reset        # Reset database and tables
make db-admin        # Open DynamoDB Admin UI
make db-clear        # Clear database data

# SAM Operations
make sam-build       # Build Lambda package
make sam-local       # Start local API (port 3000)
make sam-deploy      # Deploy to AWS (interactive)
make sam-deploy-env ENV=dev  # Deploy to specific environment
make sam-invoke FUNC=name    # Invoke Lambda function
make sam-logs        # View Lambda logs
make sam-delete STACK=name   # Delete AWS stack

# Testing
make test-api        # Test health endpoint
make test-process    # Test process endpoint

# Cleanup
make clean           # Remove build artifacts
```

## Source Code Organization

### `/src` Directory Structure

```
src/
├── __init__.py           # Package initialization and logging setup
├── aws/                  # AWS Lambda specific code
│   ├── lambda_handler.py    # Lambda entry point
│   ├── api/                 # API endpoint handlers
│   └── services/            # AWS service integrations
├── core/                 # Core business logic
│   ├── constants.py         # Configuration constants
│   ├── exceptions.py        # Custom exceptions
│   ├── utils.py            # Utility functions
│   ├── database/           # Database models and operations
│   │   ├── models.py          # DynamoDB models
│   │   └── repository.py      # Data access layer
│   ├── processing/         # Signal processing algorithms
│   │   ├── processor.py       # Main processing pipeline
│   │   ├── kalman_filter.py   # Kalman filter implementation
│   │   ├── detector.py        # Weight change detection
│   │   └── validators.py      # Data validation
│   └── replay/            # State replay and recovery
│       ├── replay_manager.py  # Replay coordination
│       └── buffer_manager.py  # Measurement buffering
├── local/               # Local development tools
│   ├── main.py            # CLI entry point
│   ├── batch/             # Batch processing utilities
│   └── visualization/     # Data visualization tools
└── factories/           # Factory pattern implementations
    └── component_factory.py  # Component instantiation
```

### Core Components

**Processing Pipeline** (`core/processing/`)
- **Kalman Filter**: Adaptive filtering for noisy weight measurements
- **Change Detection**: Identifies significant weight changes and stable periods
- **State Validation**: Ensures filter stability and measurement consistency

**Database Layer** (`core/database/`)
- **Models**: DynamoDB schema definitions for state persistence
- **Repository**: Abstract data access with support for local and AWS DynamoDB

**Replay System** (`core/replay/`)
- **State Recovery**: Rebuilds processing state from historical measurements
- **Buffer Management**: Handles measurement queuing and replay sequences

**AWS Integration** (`aws/`)
- **Lambda Handler**: Serverless function entry point with error handling
- **API Routes**: RESTful endpoints for processing, state management, and queries
- **Service Layer**: AWS service abstractions for DynamoDB, S3, and CloudWatch

## API Endpoints

The service exposes the following REST API endpoints:

- `POST /process` - Process weight measurement batch
- `GET /state/{device_id}` - Retrieve current processing state
- `POST /reset/{device_id}` - Reset device processing state
- `GET /history/{device_id}` - Get measurement history
- `DELETE /state/{device_id}` - Remove device state

## Configuration

Environment variables for service configuration:

```bash
# DynamoDB Configuration
DYNAMODB_ENDPOINT=http://localhost:8000  # Local DynamoDB endpoint
DYNAMODB_TABLE_NAME=weight-processor-state

# AWS Configuration
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=local
AWS_SECRET_ACCESS_KEY=local

# Service Configuration
LOG_LEVEL=INFO
ENVIRONMENT=dev
```

## Testing

### Unit Tests

Run tests using uv and pytest:

```bash
# Install test dependencies
uv pip install -r requirements-dev.txt

# Run all tests
uv run pytest tests/

# Run tests with verbose output
uv run pytest tests/ -xvs

# Run specific test file
uv run pytest tests/test_processor.py

# Generate coverage report
uv run pytest tests/ --cov=src --cov-report=html
# View HTML report: open htmlcov/index.html
```

### Integration Testing

Test against local SAM API:

```bash
# 1. Start local API
make sam-local

# 2. Test endpoints (in another terminal)
make test-api        # Test health endpoint
make test-process    # Test process endpoint

# 3. Use Postman collection
# Import: weight-processor-api-v2.postman_collection.json
# Set base URL to: http://localhost:3000
```

Test against deployed AWS service:

```bash
# Get API URL from deployed stack
API_URL=$(aws cloudformation describe-stacks \
  --stack-name weight-processor-dev \
  --query 'Stacks[0].Outputs[?OutputKey==`ApiUrl`].OutputValue' \
  --output text)

# Test with curl
curl -X POST $API_URL/process \
  -H "Content-Type: application/json" \
  -d @test_events/process_event.json
```

### Load Testing

```bash
# Using Apache Bench (ab)
ab -n 100 -c 10 -T application/json \
  -p test_events/process_event.json \
  http://localhost:3000/process

# Using hey (https://github.com/rakyll/hey)
hey -n 1000 -c 50 -m POST \
  -H "Content-Type: application/json" \
  -d @test_events/process_event.json \
  http://localhost:3000/process
```

## Troubleshooting

### Common Issues

**DynamoDB Local won't start:**
```bash
# Check if port 8000 is already in use
lsof -i :8000
# Kill the process or use a different port

# Reset DynamoDB
make db-reset
```

**SAM build fails:**
```bash
# Clear SAM cache
rm -rf aws/.aws-sam

# Rebuild with verbose output
cd aws
sam build --debug

# Check Python version
python --version  # Should be 3.12+
```

**Lambda function times out locally:**
```bash
# Increase timeout in template.yaml
# Globals > Function > Timeout: 60

# Or set environment variable
export SAM_CLI_TIMEOUT=60
```

**Import errors in tests:**
```bash
# Ensure dev dependencies are installed
uv pip install -r requirements-dev.txt

# Add project root to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

**Docker compose issues:**
```bash
# Clean up all containers
docker-compose down -v

# Remove orphaned containers
docker-compose down --remove-orphans

# Rebuild services
docker-compose build --no-cache
docker-compose up -d
```

## Development Tools

- **Postman Collection**: Import `weight-processor-api-v2.postman_collection.json` for API testing
- **DynamoDB Admin**: Access at `http://localhost:8001` when using Docker
- **LocalStack Dashboard**: Monitor local AWS services at `http://localhost:4566`

## License

Proprietary - All rights reserved
