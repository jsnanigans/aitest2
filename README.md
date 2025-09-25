# Weight Processor Service

A hosted service for processing weight measurements using advanced Kalman filtering and statistical analysis. The service provides both local development environments and AWS Lambda deployment options for scalable, production-ready weight data processing.

## Quick Start

### Option 1: Docker Development (Recommended)

The fastest way to get started with a fully isolated environment:

```bash
# Start the complete development environment
make -f Makefile.docker quick-start

# Run the API locally with SAM
make -f Makefile.docker sam-api

# Access the development shell
make -f Makefile.docker docker-shell
```

### Option 2: Local Development with Make

For direct local development:

```bash
# Install dependencies
make setup

# Run with sample data
make run

# Run with specific data file
make run-file FILE=path/to/data.csv

# Start local API server (requires SAM CLI)
make sam-local
```

### Option 3: AWS SAM Deployment

Deploy to AWS Lambda:

```bash
# Build the Lambda package
cd aws
make build

# Deploy to AWS
make deploy ENV=dev

# Run integration tests
make test-deployed ENV=dev
```

## Architecture

### Docker Services

The `docker-compose.yml` provides a complete local AWS environment:

- **LocalStack**: Full AWS service emulation (Lambda, API Gateway, DynamoDB, S3)
- **DynamoDB Local**: Standalone DynamoDB for development
- **DynamoDB Admin UI**: Web interface for viewing data (port 8001)

### SAM Configuration

The service uses AWS SAM (`aws/template.yaml`) for serverless deployment:

- **API Gateway**: RESTful API endpoints for weight processing
- **Lambda Function**: Python 3.12 runtime with optimized dependencies
- **DynamoDB**: State persistence for Kalman filters and processing history
- **CloudWatch**: Logging and monitoring

### Make Commands

Key Make targets for development workflow:

```bash
# Database Management
make db-start          # Start DynamoDB Local
make db-stop           # Stop DynamoDB Local
make db-reset          # Reset database and tables

# Testing
make test              # Run unit tests
make test-integration  # Run integration tests
make test-coverage     # Generate coverage report

# Development
make lint              # Run code linting
make format            # Format code
make type-check        # Run type checking
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

The service includes comprehensive test coverage:

```bash
# Run all tests
make test

# Run specific test file
make test-file FILE=tests/test_processor.py

# Generate coverage report
make test-coverage

# Run integration tests against deployed service
make test-deployed ENV=dev
```

## Development Tools

- **Postman Collection**: Import `weight-processor-api-v2.postman_collection.json` for API testing
- **DynamoDB Admin**: Access at `http://localhost:8001` when using Docker
- **LocalStack Dashboard**: Monitor local AWS services at `http://localhost:4566`

## License

Proprietary - All rights reserved