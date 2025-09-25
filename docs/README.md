# Weight Stream Processor

A high-performance, production-ready weight measurement processing system with Kalman filtering and intelligent outlier detection. Available as both a CLI tool and AWS Lambda service.

## Performance

- **Processing Speed**: 0.21ms per measurement (14x faster than requirements)
- **Code Size**: 3,472 lines (40% reduction from original)
- **Architecture**: Clean, linear processing pipeline
- **Lambda Support**: Full AWS Lambda integration with API Gateway

## Features

- **Kalman Filtering**: Adaptive noise-based filtering for smooth weight tracking
- **Source-Specific Processing**: Intelligent handling based on data source reliability
- **BMI Detection**: Automatic detection and conversion of BMI values
- **Physiological Validation**: Comprehensive validation against human limits
- **Gap Detection**: Automatic reset after extended measurement gaps
- **Structured Logging**: Production-ready logging and metrics
- **AWS Lambda API**: RESTful API for cloud deployment
- **Replay System**: Reprocess historical data with state rollback

## Installation

```bash
# Install dependencies
uv pip install -r requirements.txt

# For Lambda development
uv pip install -r requirements-lambda.txt
```

## Usage

### Basic Processing

```bash
# Process a CSV file
uv run python main.py data/weights.csv

# With configuration
uv run python main.py data/weights.csv --config config.toml

# Generate visualizations
uv run python main.py data/weights.csv --visualize
```

### AWS Lambda Local Development

```bash
# Quick start - build and run locally (no Docker required)
make local

# Test the API endpoints
make local-health    # Check health status
make local-test      # Test with sample data

# Individual commands
make build-local     # Build Lambda package locally
make clean          # Clean build artifacts
```

#### API Endpoints (Local)

All endpoints are available at `http://localhost:5448` with no authentication required for local testing:

- `GET  /api/v1/health` - Health check
- `POST /api/v1/process/{userId}` - Process weight measurements
- `POST /api/v1/replay/{userId}` - Replay measurements from timestamp
- `POST /api/v1/cleanup/{userId}` - Cleanup with Kalman reset
- `GET  /api/v1/state/{userId}` - Get user's Kalman state
- `DELETE /api/v1/state/{userId}` - Delete user state

#### Example API Request

```bash
curl -X POST http://localhost:5448/api/v1/process/user-123 \
  -H "Content-Type: application/json" \
  -d '{
    "measurements": [{
      "uuid": "550e8400-e29b-41d4-a716-446655440000",
      "userId": "user-123",
      "weight": 75.5,
      "unit": "kg",
      "timestamp": "2024-01-01T10:00:00Z",
      "effectiveDateTime": "2024-01-01T10:00:00Z",
      "source": "patient-device"
    }]
  }'
```

### Docker Alternative (Optional)

If you prefer using Docker containers:

```bash
make docker-build    # Build with Docker
make docker-run      # Start API with Docker
make docker-test     # Test with Docker
```

### Performance Testing

```bash
# Run performance benchmark
uv run python scripts/measure_performance.py
```

### Running Tests

```bash
# Run all tests
uv run python -m pytest tests/

# Run specific test
uv run python tests/test_processor.py

# Run Lambda handler tests
make test-lambda
```

## Architecture

```
main.py                     # CLI entry point and CSV processing
├── src/
│   ├── processing/         # Core processing modules
│   │   ├── processor.py    # Main processing pipeline
│   │   ├── kalman.py       # Adaptive Kalman filter
│   │   ├── validation.py   # Data validation
│   │   ├── outlier_detection.py  # Statistical outlier detection
│   │   └── unified_quality_scorer.py  # Quality scoring system
│   ├── replay/             # Replay system
│   │   ├── replay_manager.py  # Orchestrates replay operations
│   │   ├── replay_buffer.py   # Measurement buffering
│   │   └── replay_processor.py  # Enhanced replay logic
│   ├── services/           # Lambda service layer
│   │   ├── weight_processor_service.py  # Business logic
│   │   └── replay_service.py  # Replay service
│   ├── lambda_handler.py  # AWS Lambda entry point
│   ├── database.py        # State persistence
│   ├── constants.py       # Safety limits & constants
│   └── visualization.py   # Data visualization
├── tests/                  # Comprehensive test suite
└── template-local.yaml     # SAM template for local development
```

## Configuration

The system uses a combination of hard-coded safety limits (in `constants.py`) and configurable parameters (in `config.toml`).

### Key Parameters

- **Kalman Filter**: Optimized parameters for weight tracking
- **Source Profiles**: Reliability and noise characteristics per source
- **Physiological Limits**: Safety boundaries for human weight
- **Processing Thresholds**: Adaptive thresholds based on time gaps

## Data Sources

The system intelligently handles different data sources with varying reliability:

- `patient-upload`: Most reliable (noise multiplier: 0.7)
- `care-team-upload`: Excellent reliability (noise multiplier: 0.5)
- `questionnaire`: Good reliability (noise multiplier: 0.8)
- `patient-device`: Moderate reliability (noise multiplier: 1.0)
- `connectivehealth.io`: Lower reliability (noise multiplier: 1.5)
- `iglucose.com`: Requires extra validation (noise multiplier: 3.0)

## Performance Metrics

Current performance (100 measurements test):
- Average: 0.21ms
- Median: 0.21ms
- Min: 0.20ms
- Max: 0.25ms
- Target: <3ms ✅

## Development

### Code Style
- Python 3.11+
- Type hints optional (pyright mode: off)
- No comments unless critical
- Single-purpose functions
- Clear module boundaries

### Testing
- Unit tests for each module
- Integration tests for full pipeline
- Performance benchmarks
- Golden dataset regression tests

## AWS Deployment

### Prerequisites

- AWS CLI configured with appropriate credentials
- SAM CLI installed (`brew install aws-sam-cli`)
- Python 3.12 runtime

### Deploy to AWS

```bash
# Deploy to development environment (includes API Gateway)
make deploy-dev

# Deploy to production (Lambda only, no API Gateway)
make deploy-prod
```

### Environment Variables

The Lambda function uses these environment variables (configured in SAM templates):

- `KALMAN_ENABLED`: Enable/disable Kalman filtering
- `KALMAN_ADAPTIVE`: Enable adaptive noise parameters
- `QUALITY_SCORING_ENABLED`: Enable quality scoring system
- `OUTLIER_DETECTION_ENABLED`: Enable outlier detection
- `REPLAY_ENABLED`: Enable replay functionality
- `DB_BACKEND`: Database backend (`memory` for local, `sqlite` for production)
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)

## License

[Your License Here]