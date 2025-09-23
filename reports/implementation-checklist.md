# Production Separation Implementation Checklist

## Pre-Migration Tasks
- [ ] Create feature branch `feature/production-separation`
- [ ] Backup current working state
- [ ] Run full test suite to establish baseline
- [ ] Document current Lambda package size

## Phase 1: Core Package Extraction (Days 1-3)

### Day 1: Setup Core Structure
- [ ] Create `core/` directory
- [ ] Create `core/__init__.py`
- [ ] Create subdirectories: `core/processing/`, `core/replay/`

### Day 2: Move Processing Logic
- [ ] Move `src/processing/processor.py` → `core/processing/processor.py`
- [ ] Move `src/processing/kalman.py` → `core/processing/kalman.py`
- [ ] Move `src/processing/unified_quality_scorer.py` → `core/processing/quality_scorer.py`
- [ ] Move `src/processing/validation.py` → `core/processing/validation.py`
- [ ] Move `src/processing/outlier_detection.py` → `core/processing/outlier_detection.py`
- [ ] Move `src/processing/reset_manager.py` → `core/processing/reset_manager.py`
- [ ] Move `src/processing/circuit_breaker.py` → `core/processing/circuit_breaker.py`
- [ ] Move `src/processing/state_validator.py` → `core/processing/state_validator.py`
- [ ] Update all imports within core to use relative paths

### Day 3: Move Replay Logic
- [ ] Move `src/replay/replay_manager.py` → `core/replay/manager.py`
- [ ] Move `src/replay/replay_buffer.py` → `core/replay/buffer.py`
- [ ] Move `src/replay/replay_processor.py` → `core/replay/processor.py`
- [ ] Move `src/constants.py` → `core/constants.py`
- [ ] Move `src/exceptions.py` → `core/exceptions.py`
- [ ] Create `core/requirements.txt`
- [ ] Test core package imports independently

## Phase 2: Lambda Package Creation (Days 4-5)

### Day 4: Lambda Structure
- [ ] Create `lambda/` directory
- [ ] Move `src/lambda_handler.py` → `lambda/handler.py`
- [ ] Create `lambda/api/` directory
- [ ] Move `src/api/models.py` → `lambda/api/models.py`
- [ ] Create `lambda/services/` directory
- [ ] Move `src/services/weight_processor_service.py` → `lambda/services/processor.py`
- [ ] Move `src/services/replay_service.py` → `lambda/services/replay.py`

### Day 5: Lambda Dependencies
- [ ] Create `lambda/stores/` directory
- [ ] Move `src/database/dynamodb_store.py` → `lambda/stores/dynamodb.py`
- [ ] Move `src/database/memory_store.py` → `lambda/stores/memory.py`
- [ ] Move `src/database/base.py` → `lambda/stores/base.py`
- [ ] Update Lambda imports to reference `core` package
- [ ] Create `lambda/requirements.txt` (minimal deps)
- [ ] Create `lambda/config.py` for environment variable handling

## Phase 3: Local Tools Isolation (Days 6-7)

### Day 6: Local Structure
- [ ] Create `local/` directory
- [ ] Move `main.py` → `local/main.py`
- [ ] Create `local/analysis/` directory
- [ ] Move all files from `src/analysis/` → `local/analysis/`
- [ ] Create `local/viz/` directory
- [ ] Move all files from `src/viz/` → `local/viz/`

### Day 7: Local Database
- [ ] Create `local/database/` directory
- [ ] Move `src/database/database.py` → `local/database/sqlite_store.py`
- [ ] Move `src/database/db_wrapper.py` → `local/database/wrapper.py`
- [ ] Move `src/batch/csv_processor.py` → `local/batch/csv_processor.py`
- [ ] Update local imports to reference `core` package
- [ ] Create `local/requirements.txt` (include viz dependencies)
- [ ] Test local CSV processing

## Phase 4: Build and Package (Days 8-9)

### Day 8: Build Scripts
- [ ] Create `scripts/build_lambda.sh`:
  ```bash
  #!/bin/bash
  rm -rf build/lambda lambda-deployment.zip
  mkdir -p build/lambda
  cp -r lambda/* build/lambda/
  cp -r core build/lambda/
  cd build/lambda
  pip install -r requirements.txt -t .
  zip -r ../../lambda-deployment.zip .
  cd ../..
  echo "Package size: $(du -h lambda-deployment.zip | cut -f1)"
  ```
- [ ] Create `scripts/test_lambda_local.py`
- [ ] Create `scripts/run_local.sh` for local testing
- [ ] Test Lambda package size (<50MB)

### Day 9: Configuration Updates
- [ ] Create `config/lambda.env.template`
- [ ] Update `config.toml` → `config/local.toml`
- [ ] Create environment variable mapping document
- [ ] Update `template.yaml` for new Lambda structure
- [ ] Update `Makefile` for new build process

## Phase 5: Testing and Validation (Days 10-12)

### Day 10: Unit Tests
- [ ] Update test imports for new structure
- [ ] Run core package tests independently
- [ ] Run Lambda handler tests
- [ ] Run local tool tests
- [ ] Achieve 100% test passage

### Day 11: Integration Testing
- [ ] Test Lambda locally with SAM
- [ ] Test local CSV processing
- [ ] Test visualization generation
- [ ] Verify all replay functionality

### Day 12: Performance Testing
- [ ] Measure Lambda cold start time
- [ ] Measure Lambda package size
- [ ] Test with production-like payload
- [ ] Document performance metrics

## Phase 6: Deployment (Days 13-14)

### Day 13: Staging Deployment
- [ ] Deploy to AWS staging environment
- [ ] Run smoke tests
- [ ] Test all API endpoints
- [ ] Verify CloudWatch logs
- [ ] Load test with sample data

### Day 14: Production Preparation
- [ ] Create rollback plan
- [ ] Update deployment documentation
- [ ] Create migration runbook
- [ ] Schedule production deployment
- [ ] Notify stakeholders

## Post-Migration Tasks
- [ ] Monitor production metrics for 24 hours
- [ ] Update README with new structure
- [ ] Update CI/CD pipelines
- [ ] Remove old `src/` directory
- [ ] Archive migration branch
- [ ] Document lessons learned

## Validation Checklist

### Lambda Package
- [ ] Size < 50MB uncompressed
- [ ] No visualization dependencies
- [ ] No pandas dependency
- [ ] No matplotlib dependency
- [ ] Only production stores (DynamoDB, Memory)
- [ ] Clean imports from core

### Core Package
- [ ] No AWS-specific imports
- [ ] No visualization imports
- [ ] No database implementations
- [ ] Only abstract interfaces
- [ ] Comprehensive unit tests

### Local Package
- [ ] All visualization tools present
- [ ] CSV processing functional
- [ ] SQLite database working
- [ ] Can import and use core
- [ ] Generates reports correctly

## Success Metrics
- [ ] Lambda package reduced by >50%
- [ ] Cold start time improved by >30%
- [ ] All tests passing
- [ ] No production incidents
- [ ] Clean separation achieved

## Risk Register
1. **Import path confusion** → Use find/replace carefully
2. **Missing dependencies** → Test each package independently
3. **Configuration issues** → Maintain backward compatibility
4. **Test failures** → Fix before proceeding to next phase
5. **Production impact** → Deploy during low-traffic window

## Quick Commands

### Test Core Package
```bash
cd core && python -m pytest tests/
```

### Build Lambda Package
```bash
./scripts/build_lambda.sh
```

### Test Local Processing
```bash
cd local && python main.py ../data/weights.csv
```

### Deploy to Staging
```bash
sam deploy --config-env staging
```

## Notes
- Keep original `src/` until migration is complete
- Commit after each successful phase
- Run tests after each major change
- Document any deviations from plan