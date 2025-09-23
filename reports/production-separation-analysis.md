# Production Separation Analysis

## Executive Summary

After analyzing the `src/` folder, I've identified a clear need for separation between AWS/Lambda code, shared processing code, and local-only utilities. Currently, the codebase mixes production and development concerns, which increases Lambda package size and creates unnecessary complexity.

## Current Structure Analysis

### 1. AWS/Lambda Components (Production)
**Files that must be deployed to AWS:**

```
src/lambda_handler.py              # AWS Lambda entry point
src/api/models.py                  # API request/response models
src/services/                      # Service layer
├── weight_processor_service.py    # Main processing service
└── replay_service.py              # Replay functionality

src/database/
├── base.py                        # Abstract base
├── dynamodb_store.py              # DynamoDB implementation
└── memory_store.py                # For Lambda testing

src/config/config_manager.py       # Configuration management
```

### 2. Shared Core Processing (Used by Both AWS and Local)
**Business logic used by both environments:**

```
src/processing/
├── processor.py                   # Core processing pipeline
├── kalman.py                      # Kalman filter implementation
├── unified_quality_scorer.py      # Quality scoring
├── validation.py                  # Data validation
├── outlier_detection.py          # Outlier detection
├── reset_manager.py               # Reset management
├── circuit_breaker.py             # Safety mechanisms
├── state_validator.py             # State validation
└── buffer_factory.py              # Buffer management

src/replay/
├── replay_manager.py              # Replay orchestration
├── replay_buffer.py              # Buffer system
├── replay_processor.py           # Enhanced replay logic
└── sliding_window_processor.py   # Window processing

src/constants.py                  # Shared constants
src/exceptions.py                 # Custom exceptions
src/utils.py                      # Utility functions (partial)
```

### 3. Local-Only Components (Not for Production)
**Files used only for local testing/analysis:**

```
main.py                           # Local CSV processor entry point

src/batch/csv_processor.py        # CSV batch processing

src/analysis/                     # ALL analysis tools (15 files)
├── visualization_generator.py
├── csv_generator.py
├── markdown_reporter.py
├── quarterly_reporting.py
├── daily_weight_analyzer.py
├── statistics.py
└── ... (all other analysis files)

src/viz/                         # ALL visualization tools
├── visualization.py
├── viz_index.py
└── __init__.py

src/database/
├── database.py                  # SQLite implementation (local only)
└── db_wrapper.py               # Local DB wrapper

src/factories/component_factory.py  # Local testing factory
```

## Key Issues Identified

### 1. Mixed Dependencies
- Lambda package includes visualization libraries (matplotlib, plotly, pandas)
- Local-only analysis code increases Lambda deployment size
- SQLite database code unnecessary in Lambda

### 2. Configuration Complexity
- Config system tries to handle both local and AWS scenarios
- Environment detection logic scattered throughout code
- `config.toml` contains local-specific settings

### 3. Import Coupling
- Local-only code imports from production modules
- No clear boundary enforcement
- Circular dependency risks

### 4. Database Abstraction
- Good abstraction pattern exists (StateStore base class)
- But local SQLite implementation mixed with production code
- Memory store could be shared for testing

## Recommended Folder Structure

```
stream-processor/
├── lambda/                      # AWS Lambda deployment package
│   ├── handler.py              # Lambda entry point
│   ├── requirements.txt        # Lambda-only dependencies
│   ├── api/
│   │   └── models.py
│   ├── services/
│   │   ├── processor.py
│   │   └── replay.py
│   └── stores/
│       ├── dynamodb.py
│       └── memory.py
│
├── core/                       # Shared processing logic
│   ├── __init__.py
│   ├── processing/
│   │   ├── processor.py
│   │   ├── kalman.py
│   │   ├── quality_scorer.py
│   │   ├── validation.py
│   │   ├── outlier_detection.py
│   │   └── reset_manager.py
│   ├── replay/
│   │   ├── buffer.py
│   │   ├── manager.py
│   │   └── processor.py
│   ├── constants.py
│   ├── exceptions.py
│   └── config.py              # Config schema only
│
├── local/                      # Local-only tools
│   ├── main.py                # CSV processor
│   ├── requirements.txt       # Local dependencies
│   ├── analysis/
│   │   └── ... (all analysis tools)
│   ├── viz/
│   │   └── ... (all visualization)
│   ├── database/
│   │   ├── sqlite_store.py
│   │   └── csv_export.py
│   └── utils/
│       └── local_helpers.py
│
├── tests/                      # All tests
│   ├── unit/
│   ├── integration/
│   └── fixtures/
│
├── config/                     # Configuration files
│   ├── lambda.env             # Lambda environment template
│   ├── local.toml            # Local development config
│   └── test.toml             # Test configuration
│
└── scripts/                   # Build and deployment
    ├── build_lambda.py       # Package Lambda deployment
    ├── deploy.sh            # Deployment script
    └── test_local.py        # Local testing
```

## Migration Strategy

### Phase 1: Create Core Package (Week 1)
1. Create `core/` directory
2. Move shared processing logic with no external dependencies
3. Ensure all imports use relative paths within core
4. Create `__init__.py` with clean exports

### Phase 2: Separate Lambda Code (Week 1)
1. Create `lambda/` directory
2. Move Lambda handler and API models
3. Create Lambda-specific service layer that imports from `core/`
4. Create minimal `requirements.txt` for Lambda

### Phase 3: Isolate Local Tools (Week 2)
1. Create `local/` directory
2. Move all analysis and visualization code
3. Move SQLite database implementation
4. Update `main.py` to import from correct locations

### Phase 4: Package and Deploy (Week 2)
1. Create build script for Lambda deployment
2. Package only `lambda/` + `core/` for AWS
3. Test Lambda package size (<50MB unzipped)
4. Deploy and test in AWS environment

## Dependency Management

### Lambda Package (requirements-lambda.txt)
```
numpy==1.26.4
pydantic==2.11.9
# boto3 provided by Lambda runtime
```

### Core Package (requirements-core.txt)
```
numpy==1.26.4
pykalman>=0.10.2
pydantic>=2.11.9
```

### Local Package (requirements-local.txt)
```
# Include core requirements
-r ../core/requirements.txt

# Visualization and analysis
matplotlib>=3.10.6
plotly>=6.3.0
pandas>=2.3.2

# Local database
# (SQLite is built-in)

# Development tools
pytest>=8.3.4
black>=24.10.0
mypy>=1.14.2
```

## Configuration Strategy

### 1. Lambda Configuration
- Use environment variables for all settings
- No file-based config in Lambda
- ConfigManager reads from `os.environ`
- Secrets in AWS Secrets Manager

### 2. Core Configuration
- Define configuration schema in `core/config.py`
- Use Pydantic models for validation
- No default values in core (provided by environment)

### 3. Local Configuration
- Continue using `config.toml` for local runs
- Local ConfigManager translates TOML to core schema
- Support for multiple config profiles

## Build Process

### Lambda Build Script (`scripts/build_lambda.py`)
```python
#!/usr/bin/env python3
"""Build Lambda deployment package."""

import shutil
import subprocess
from pathlib import Path

def build_lambda():
    # Create build directory
    build_dir = Path("build/lambda")
    build_dir.mkdir(parents=True, exist_ok=True)

    # Copy lambda code
    shutil.copytree("lambda", build_dir / "lambda")

    # Copy core code
    shutil.copytree("core", build_dir / "core")

    # Install dependencies
    subprocess.run([
        "pip", "install",
        "-r", "lambda/requirements.txt",
        "-t", str(build_dir)
    ])

    # Create deployment package
    shutil.make_archive("lambda-deployment", "zip", build_dir)

    print(f"Lambda package created: lambda-deployment.zip")
    print(f"Package size: {Path('lambda-deployment.zip').stat().st_size / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    build_lambda()
```

## Testing Strategy

### 1. Unit Tests
- Test `core/` modules independently
- Mock external dependencies
- 100% coverage for core business logic

### 2. Integration Tests
- Test Lambda handler with memory store
- Test local processor with SQLite
- Verify core package works in both contexts

### 3. Lambda Local Testing
- Use AWS SAM for local Lambda testing
- Test with production-like payloads
- Verify memory and timeout constraints

## Benefits of Separation

### 1. Reduced Lambda Package Size
- Current: ~100MB with all dependencies
- Target: <20MB with only core dependencies
- Faster cold starts and deployments

### 2. Clear Boundaries
- Core logic is environment-agnostic
- Lambda code focused on AWS integration
- Local tools can evolve independently

### 3. Improved Testing
- Core logic tested in isolation
- Lambda handler tested with mocks
- Local tools tested with real data

### 4. Easier Maintenance
- Clear ownership and responsibilities
- Reduced coupling between components
- Simpler dependency management

## Next Steps

1. **Review and approve** this separation plan
2. **Create feature branch** for restructuring
3. **Execute Phase 1** - Extract core package
4. **Test core package** independently
5. **Continue with phases** 2-4
6. **Update CI/CD** for new structure
7. **Deploy to staging** for validation
8. **Production deployment** after testing

## Risks and Mitigations

### Risk 1: Breaking Existing Functionality
**Mitigation:** Comprehensive test suite before restructuring

### Risk 2: Import Path Issues
**Mitigation:** Use find/replace for import updates, test thoroughly

### Risk 3: Configuration Incompatibility
**Mitigation:** Maintain backward compatibility during transition

### Risk 4: Deployment Complexity
**Mitigation:** Automate build process, document clearly

## Conclusion

The proposed separation will create a cleaner, more maintainable codebase with:
- Minimal Lambda deployment package
- Clear separation of concerns
- Improved testability
- Reduced coupling

The migration can be completed in 2 weeks with minimal risk if executed systematically.