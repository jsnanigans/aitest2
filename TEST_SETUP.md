# Test Setup Summary

## Overview

This document describes the test infrastructure for comparing Python and TypeScript weight processor implementations.

## Test Structure

### Shared Fixture
- **Location**: `test_fixtures/july11_replay_scenario.json`
- **Purpose**: Documents the exact scenario where Python/TypeScript diverge in full batch processing
- **Contains**:
  - 6 setup measurements to establish Kalman filter state
  - 3 test measurements that trigger divergence
  - Expected results from both implementations in full batch

### Python Tests
- **Location**: `python_lib/tests/processing/test_july11_scenario.py`
- **Test Runner**: `pytest` via `uv run pytest`
- **Run Command**:
  ```bash
  cd python_lib
  uv run pytest tests/processing/test_july11_scenario.py -v -s
  ```

### TypeScript Tests
- **Location**: `typescript_lib/tests/july11_scenario.test.ts`
- **Test Runner**: `bun test`
- **Run Command**:
  ```bash
  cd typescript_lib
  bun test tests/july11_scenario.test.ts
  ```

## Test Results

### Isolated Scenario (6 setup + 3 test measurements)

Both Python and TypeScript produce **identical results** in the isolated scenario:

| Measurement | Weight | Python Result | TypeScript Result |
|-------------|--------|---------------|-------------------|
| 52ec2c45 (first) | 59.6kg | accepted=True, score=0.8545 | accepted=true, score=0.8545 |
| **4f07af66 (middle)** | **58.4kg** | **accepted=False, score=0.0063** | **accepted=false, score=0.0063** |
| 726b441f (third) | 59.6kg | accepted=True, score=0.5410 | accepted=true, score=0.5410 |

### Full Batch Scenario (120 measurements)

When processing all 120 measurements in sequence, the implementations diverge:

| Measurement | Python Full Batch | TypeScript Full Batch |
|-------------|-------------------|----------------------|
| **4f07af66** | **accepted=False, score=0.346** | **accepted=True, score=0.585** |

## Key Findings

1. ✅ **Isolated scenario**: Python and TypeScript are **identical**
2. ❌ **Full batch scenario**: Python and TypeScript **diverge** (quality score diff: 0.238)
3. 🔍 **Root cause**: The divergence occurs due to accumulated differences in Kalman filter state over 120 measurements
4. 📊 **Impact**: Different acceptance decisions for the same measurement depending on processing history

## Next Steps

To resolve the divergence:

1. Compare Kalman filter implementations in detail
2. Check for floating point precision differences
3. Verify quality scorer component weights match exactly
4. Add integration tests that process full 120-measurement batch in both implementations
5. Create unit tests for specific Kalman filter edge cases

## Comparison Script

The main comparison script validates behavior across all 120 measurements:

```bash
bash run_comparison.sh
```

This script:
- Runs both Python and TypeScript on the full test dataset
- Compares accepted measurement IDs
- Reports any divergences
- Currently shows 1 measurement difference (4f07af66)
