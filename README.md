# Weight Stream Processor

A robust weight measurement processing system with adaptive Kalman filtering and intelligent outlier detection.

## Project Structure

```
strem_process_anchor/
├── src/                    # Source code
│   ├── core/              # Core business logic (shared)
│   │   ├── processing/    # Kalman filter, validation, quality scoring
│   │   ├── replay/        # Replay system for historical data
│   │   ├── database/      # Database interfaces and state management
│   │   ├── constants.py   # Physical constants and limits
│   │   ├── exceptions.py  # Custom exceptions
│   │   └── utils.py       # Utility functions
│   │
│   ├── aws/               # AWS Lambda specific code
│   │   ├── lambda_handler.py    # Lambda entry point
│   │   ├── api/                  # API models and interfaces
│   │   ├── config/               # Configuration management
│   │   └── services/             # AWS-specific services
│   │
│   └── local/             # Local-only code
│       ├── main.py               # Local CLI entry point
│       ├── analysis/             # Data analysis scripts
│       └── viz/                  # Visualization tools
│
├── aws/                   # AWS deployment configurations
│   ├── template.yaml             # SAM template for deployment
│   ├── template-prod.yaml        # Production SAM template
│   ├── template-local.yaml       # Local testing template
│   └── samconfig.toml            # SAM configuration
│
├── config/                # Configuration files
│   ├── local/            # Local development configs
│   └── aws/              # AWS deployment configs
│
├── tests/                 # Test suite
├── docs/                  # Documentation
├── scripts/               # Build and utility scripts
└── data/                  # Sample and test data
```

## Usage

### Local Development

Run the processor locally with CSV data:

```bash
# Process a CSV file
uv run python src/local/main.py data/weights.csv

# With custom configuration
uv run python src/local/main.py data/weights.csv --config config/local/config.toml

# Generate visualizations
uv run python src/local/main.py data/weights.csv --visualize
```

### AWS Deployment

Deploy to AWS using SAM:

```bash
# Build the Lambda package
cd aws
sam build

# Deploy to development
sam deploy --guided

# Deploy to production
sam deploy --config-env prod
```

### Testing

Run the test suite:

```bash
# All tests
uv run python -m pytest tests/

# Specific test file
uv run python -m pytest tests/test_processor.py -xvs

# With coverage
uv run python -m pytest tests/ --cov=src
```

## Architecture

### Core Processing Pipeline

The system uses an adaptive Kalman filter with intelligent outlier detection:

1. **Data Validation**: Input validation and unit conversion
2. **Kalman Filtering**: Adaptive noise parameters based on source reliability
3. **Quality Scoring**: Multi-factor quality assessment
4. **Outlier Detection**: Statistical methods with quality override
5. **State Management**: Persistent state storage (SQLite local, DynamoDB AWS)

### Key Features

- **Adaptive Kalman Filter**: Adjusts to gaps in data and source reliability
- **Source-Based Reliability**: Different noise profiles for different data sources
- **Quality Override System**: High-quality measurements can override outlier detection
- **Replay System**: Process historical data with proper temporal ordering
- **Circuit Breaker**: Protects against cascading failures

## Configuration

Configuration is managed through TOML files with sections for:
- Kalman filter parameters
- Quality scoring weights
- Outlier detection thresholds
- Source reliability mappings
- Replay processing settings

See `config/local/config.toml` for the default configuration.

## Development

This project uses:
- Python 3.12+
- uv for dependency management
- AWS SAM for serverless deployment
- pytest for testing

Install dependencies:

```bash
uv pip sync requirements.txt
```

For AWS Lambda deployment:

```bash
uv pip sync requirements-lambda.txt
```