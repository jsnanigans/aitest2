# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Weight Processor Service that processes weight measurements using Kalman filtering and statistical analysis. The system has been refactored with the main code now residing in the `weight_values/` directory, while API tests remain in the root `tests/api/` directory.

## Key Commands

### Development & Testing
```bash
# Setup and install dependencies
make setup                          # Install all dependencies with uv

# Run tests
uv run pytest tests/api/ -xvs       # Run API tests with verbose output
uv run pytest tests/api/test_api_endpoints.py::test_process_single_measurement -xvs  # Run single test

# Docker services
make docker-up                      # Start DynamoDB local and admin UI
make docker-down                    # Stop Docker services
make db-reset                       # Reset database tables

# SAM Local API
make sam-local                      # Build and start API on port 3080
make test-api                       # Test health endpoint
make test-process                   # Test process endpoint
```

### Linting & Type Checking
```bash
# The codebase uses pyrightconfig.json for type checking
uv run pyright                      # Run type checking
uv run ruff check .                 # Run linting
uv run ruff format .                # Format code
```

## Architecture Overview

### Directory Structure
- **`weight_values/`**: Main application code (recently moved from root)
  - `src/aws/`: Lambda handlers and API endpoints
  - `src/core/`: Core business logic (processing, database, replay)
  - `src/local/`: Local development tools and scripts
  - `src/factories/`: Component factory patterns
  - SAM templates and requirements files

- **`tests/api/`**: API integration tests (remain in root directory)
  - Comprehensive test coverage for endpoints
  - Fixtures in `conftest.py`

### Core Components

1. **Processing Pipeline** (`weight_values/src/core/processing/`)
   - `processor.py`: Main processing orchestrator
   - `kalman.py`: Adaptive Kalman filter implementation
   - `unified_quality_scorer.py`: Quality scoring system
   - State validation and reset management

2. **Database Layer** (`weight_values/src/core/database/`)
   - `dynamodb_store.py`: DynamoDB persistence
   - `database.py`: SQLite local storage
   - Supports both local and AWS DynamoDB

3. **Replay System** (`weight_values/src/core/replay/`)
   - `replay_manager.py`: State recovery from history
   - `replay_buffer.py`: Measurement buffering
   - Handles state restoration after resets

4. **API Layer** (`weight_values/src/aws/`)
   - `lambda_handler.py`: AWS Lambda entry point
   - `api/models.py`: Request/response models
   - `services/`: Service layer for processing

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
- API runs on port 3080 (configured in Makefile)
- Template: `weight_values/sam-template-local.yaml`
- Local DynamoDB endpoint: `http://localhost:8000`

## Testing Strategy

### API Tests Location
All API tests are in `tests/api/` directory:
- `test_api_endpoints.py`: Core endpoint testing
- `test_state_management.py`: State persistence tests
- `test_validation.py`: Input validation tests
- `test_error_handling.py`: Error scenarios
- `test_real_world_scenarios.py`: Complex use cases
- `test_historic_data.py`: Historical data processing

### Running Tests
```bash
# Run all API tests
uv run pytest tests/api/ -xvs

# Run specific test file
uv run pytest tests/api/test_api_endpoints.py -xvs

# Run with coverage
uv run pytest tests/api/ --cov=weight_values/src --cov-report=html
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

1. **Start local services**: `make docker-up`
2. **Initialize database**: `make db-reset`
3. **Start SAM API**: `make sam-local`
4. **Run tests**: `uv run pytest tests/api/ -xvs`
5. **Test endpoints**: `make test-api` or use Postman collection

## Common Issues & Solutions

### Import Errors
- Ensure you're using `uv run` to execute Python commands
- Check PYTHONPATH includes project root

### DynamoDB Connection Issues
- Verify Docker is running: `docker ps`
- Check port 8000 is not in use: `lsof -i :8000`
- Reset database: `make db-reset`

### SAM Build Issues
- Clean build artifacts: `rm -rf weight_values/.aws-sam`
- Rebuild: `make sam-build-local`

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

## Active Work

### TypeScript Weight Processor Port (2025-11-04)
- **Goal**: Complete port of Python weight processing pipeline to TypeScript
- **Runtime**: Bun (CLI + publishable library)
- **Scope**: ~30 TypeScript files, complete feature parity
- **TempDoc**: `/Users/brendanmullins/Documents/Log/TempDoc/strem_process_anchor/2025-11/04/browser-migration-prep-progress.md`
- **Spec Directory**: `./spec/2025-11-04-browser-weight-processor/`
  - `specifications.md` - Requirements and scope
  - `research.md` - Technical research and algorithms
  - `discussion.md` - Solution approaches
  - `recommendation.md` - Final recommendations
  - `plan.md` - Detailed implementation plan (100+ tasks)
- **Status**: ✅ **PREP COMPLETE** - Ready for implementation
- **Timeline**: 6-7 weeks estimated for full implementation

### Replay Logic Analysis (2025-11-06) - ✅ RESOLVED
- **Status**: All implementations now match be_implementation_service exactly
- **Analysis**: `/Users/brendanmullins/Documents/Log/TempDoc/strem_process_anchor/2025-11/06/replay-logic-implementation-comparison.md`
- **Fixes Applied**:
  - ✅ `weight_values/src/aws/services/weight_processor_service.py` - Added time_gap trigger, removed sliding window, standardized triggers and result merging
  - ✅ `weight-processor-ts/src/services/weight_processor_service.ts` - Fixed buffering to include ALL measurements, added time_gap trigger, standardized result merging
  - ✅ `local_main.py` - Now displays replay metadata
  - ✅ `weight-processor-ts/local_main.ts` - Now displays replay metadata