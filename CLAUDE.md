# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Weight Processor Service that processes weight measurements using Kalman filtering and statistical analysis. The system has been refactored with a clean separation:

- **`python_lib/`**: Core infrastructure-agnostic processing library
- **`be_implementation_service/`**: AWS Lambda/API implementation layer (minimal, infrastructure-specific)
- **`weight_values/`**: ⚠️ **DEPRECATED** - Old implementation, use `be_implementation_service/` instead

## Key Commands

### python_lib (Core Library)
```bash
cd python_lib

# Install in editable mode with dev dependencies
uv pip install -e ".[dev]"

# Run unit tests
uv run pytest tests/ -xvs

# Run with coverage
uv run pytest tests/ --cov=src/weight_processor_lib --cov-report=html

# Linting & formatting
uv run ruff check src/ tests/
uv run black src/ tests/
```

### be_implementation_service (AWS Lambda Implementation)
```bash
cd be_implementation_service

# Install dependencies (includes python_lib)
uv pip install -r requirements-dev.txt

# Docker services
make docker-up                      # Start DynamoDB local and admin UI
make docker-down                    # Stop Docker services
make db-reset                       # Reset database tables

# SAM Local API
make sam-local                      # Build and start API on port 3080
make test-api                       # Test health endpoint
make test-process                   # Test process endpoint

# Run integration tests
uv run pytest tests/ -xvs

# Type checking (pyrightconfig.json)
uv run pyright

# Linting
uv run ruff check src/ tests/
```

## Architecture Overview

### Directory Structure

#### python_lib/ - Core Library (Infrastructure-Agnostic)
```
python_lib/
├── src/weight_processor_lib/
│   └── core/
│       ├── processing/           # Core processing logic
│       │   ├── processor.py            # Main processing orchestrator
│       │   ├── kalman.py               # Adaptive Kalman filter
│       │   ├── unified_quality_scorer.py # Quality scoring system
│       │   ├── validation.py           # Input validation
│       │   ├── reset_manager.py        # Reset logic
│       │   └── circuit_breaker.py      # Failure protection
│       ├── database/              # Storage abstraction
│       │   ├── base.py                 # Abstract StateStore interface
│       │   └── dynamodb_store.py       # DynamoDB implementation
│       ├── constants.py           # Configuration constants
│       ├── exceptions.py          # Custom exceptions
│       └── utils.py               # Shared utilities
├── tests/                        # Unit tests for core logic
│   └── processing/
└── pyproject.toml                # Package configuration
```

**Key Features**:
- Infrastructure-agnostic (can be used in Lambda, local apps, batch jobs, etc.)
- Comprehensive unit test coverage
- Installable Python package with dependencies
- Abstract storage interface (plug in any backend)

#### be_implementation_service/ - AWS Lambda Implementation
```
be_implementation_service/
├── src/aws/
│   ├── api/
│   │   └── models.py             # API request/response models (Pydantic)
│   ├── config/
│   │   └── config_manager.py     # Configuration management
│   ├── services/
│   │   ├── weight_processor_service.py  # Orchestration layer
│   │   └── replay_service.py            # Buffered replay logic
│   └── lambda_handler.py         # Lambda entry point
├── tests/
│   ├── integration/              # Integration tests (API + Lambda)
│   ├── unit/services/            # Service layer tests
│   └── fixtures/                 # API test fixtures
├── sam-template-*.yaml           # SAM deployment templates
├── requirements.txt              # Depends on python_lib
└── docker-compose.yml            # Local DynamoDB for testing
```

**Key Features**:
- Minimal, infrastructure-specific code
- Depends on `python_lib` for core logic
- Integration and service layer tests
- SAM templates for AWS deployment
- API models and Lambda handler

### Separation of Concerns

| Layer | Location | Responsibilities |
|-------|----------|------------------|
| **Core Library** | `python_lib/` | Processing, Kalman filtering, quality scoring, storage abstraction |
| **Infrastructure** | `be_implementation_service/` | Lambda handlers, API models, service orchestration, AWS config |

### Core Components

1. **Processing Pipeline** (`python_lib/src/weight_processor_lib/core/processing/`)
   - `processor.py`: Main processing orchestrator
   - `kalman.py`: Adaptive Kalman filter implementation
   - `unified_quality_scorer.py`: Multi-component quality scoring
   - `validation.py`: Input validation and preprocessing
   - `reset_manager.py`: Handles state resets
   - `circuit_breaker.py`: Failure protection

2. **Database Layer** (`python_lib/src/weight_processor_lib/core/database/`)
   - `base.py`: Abstract `StateStore` interface
   - `dynamodb_store.py`: DynamoDB implementation
   - Pluggable storage backends

3. **Replay System** (Integrated in `be_implementation_service/src/aws/services/`)
   - Buffered replay integrated into service layer
   - Time gap, batch end, and overflow triggers
   - Simple and maintainable
   - Documented in `BUFFERED_REPLAY.md`

4. **API Layer** (`be_implementation_service/src/aws/`)
   - `lambda_handler.py`: AWS Lambda entry point
   - `api/models.py`: Pydantic request/response models
   - `services/`: Orchestration between API and core library

## Important Configuration

### Environment Variables
```bash
# DynamoDB configuration
DYNAMODB_ENDPOINT=http://localhost:8000  # For local development
DYNAMODB_TABLE_NAME=weight-processor-state
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=local
AWS_SECRET_ACCESS_KEY=local
```

### SAM Local Configuration
- API runs on port 3080 (configured in be_implementation_service/Makefile)
- Template: `be_implementation_service/sam-template-local.yaml`
- Local DynamoDB endpoint: `http://localhost:8000`

## Testing Strategy

### Two-Layer Testing Approach

#### 1. Core Library Tests (`python_lib/tests/`)
**Unit tests for infrastructure-agnostic logic**:
- `tests/processing/test_processor.py`: Core processing logic
- `tests/processing/test_kalman.py`: Kalman filter tests
- `tests/processing/test_quality_scorer.py`: Quality scoring tests
- `tests/processing/test_validation.py`: Input validation tests
- `tests/processing/test_reset_manager.py`: Reset logic tests

```bash
cd python_lib
uv run pytest tests/ -xvs
uv run pytest tests/ --cov=src/weight_processor_lib --cov-report=html
```

#### 2. Infrastructure Tests (`be_implementation_service/tests/`)
**Integration and service layer tests**:
- `tests/integration/test_buffered_replay.py`: Buffered replay integration
- `tests/unit/services/test_weight_processor_service.py`: Service orchestration
- `tests/fixtures/`: API test fixtures and examples

```bash
cd be_implementation_service
uv run pytest tests/ -xvs
uv run pytest tests/ --cov=src/aws --cov-report=html
```

## Key Processing Concepts

### Kalman Filter State
The system maintains adaptive Kalman filter states per device/user:
- Tracks weight trends and velocity
- Handles measurement gaps intelligently
- Supports reset operations for significant changes

### Quality Scoring
Multi-component quality assessment:
- Plausibility checks (physiological limits)
- Temporal consistency
- Statistical validation
- Source reliability weighting

### Replay Mechanism
- Buffers recent measurements for reprocessing
- Handles state recovery after resets
- Maintains measurement history for analysis

## Development Workflow

### Working on Core Logic
1. **Navigate to python_lib**: `cd python_lib`
2. **Install in dev mode**: `uv pip install -e ".[dev]"`
3. **Make changes** to `src/weight_processor_lib/core/`
4. **Run unit tests**: `uv run pytest tests/ -xvs`
5. **Verify coverage**: `uv run pytest tests/ --cov=src/weight_processor_lib`

### Working on AWS Lambda Implementation
1. **Navigate to be_implementation**: `cd be_implementation_service`
2. **Install dependencies**: `uv pip install -r requirements-dev.txt`
3. **Start local services**: `make docker-up`
4. **Initialize database**: `make db-reset`
5. **Start SAM API**: `make sam-local`
6. **Run integration tests**: `uv run pytest tests/ -xvs`
7. **Test endpoints**: `make test-api` or use Postman collection

### Making Changes That Span Both Layers
1. **Update core logic** in `python_lib/`
2. **Run core tests** in `python_lib/`
3. **Update service layer** in `be_implementation_service/` if needed
4. **Run integration tests** in `be_implementation_service/`
5. **Verify end-to-end** with SAM local

## Common Issues & Solutions

### Import Errors
- Ensure `python_lib` is installed: `cd python_lib && uv pip install -e .`
- Ensure you're using `uv run` to execute Python commands
- For `be_implementation_service`, verify it can import from `weight_processor_lib`

### DynamoDB Connection Issues
- Verify Docker is running: `docker ps`
- Check port 8000 is not in use: `lsof -i :8000`
- Reset database: `cd be_implementation_service && make db-reset`

### SAM Build Issues
- Clean build artifacts: `rm -rf be_implementation_service/.aws-sam`
- Rebuild: `cd be_implementation_service && make sam-build-local`

### Test Import Errors
- Core tests: Ensure imports use `weight_processor_lib.core.*`
- Service tests: Ensure imports use `weight_processor_lib.core.*` for core and `src.aws.*` for services

## API Endpoints

Main endpoints (when running locally on port 3080):
- `GET /health` - Health check
- `POST /process` - Process weight measurements
- `GET /state/{device_id}/{user_id}` - Get processing state
- `POST /reset` - Reset Kalman filter state
- `GET /history` - Get measurement history

## Code Style Guidelines

- Use type hints for all function signatures
- Follow existing patterns in the codebase
- Maintain consistency with existing error handling patterns
- All new features should include corresponding tests in `tests/api/`
