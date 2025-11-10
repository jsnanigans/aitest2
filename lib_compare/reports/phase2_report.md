# Cross-Language Test Report
**Test Suite**: Phase 2: Component Tests
**Date**: 2025-11-10T13:15:47.030Z
**Total Tests**: 7
**Passed**: 2
**Failed**: 5
**Success Rate**: 28.6%
**Duration**: 3.98s

## Summary

| Status | Count | Percentage |
|--------|-------|------------|
| ✅ Passed | 2 | 28.6% |
| ❌ Failed | 5 | 71.4% |

## Performance Comparison

- **Python avg**: 132.51ms
- **TypeScript avg**: 436.13ms
- **Speed ratio**: Python is 0.70x faster

## Failed Tests

### Test 2.1: Kalman Filter Initialization
**Description**: Verify Kalman filter initializes with correct variance and state

**Comparison**: ✗ Found 3 difference(s): 3 numeric, 0 structural

**Differences**:
```
Found 3 difference(s):

  root.kalman_variance:
    Type: value
    Python:     3.6399999999999997
    TypeScript: 0.364
    Difference: 3.276e+0
    Numeric difference exceeds tolerance: abs=3.276e+0, rel=90.000000%

  root.kalman_confidence_upper:
    Type: value
    Python:     73.81575680566779
    TypeScript: 71.20664825031987
    Difference: 2.609e+0
    Numeric difference exceeds tolerance: abs=2.609e+0, rel=3.534623%

  root.kalman_confidence_lower:
    Type: value
    Python:     66.18424319433221
    TypeScript: 68.79335174968013
    Difference: 2.609e+0
    Numeric difference exceeds tolerance: abs=2.609e+0, rel=3.792675%

```

### Test 2.2: Kalman Filter Prediction
**Description**: Verify Kalman filter prediction step (state extrapolation)

**Comparison**: ✗ Found 1 difference(s): 1 numeric, 0 structural

**Differences**:
```
Found 1 difference(s):

  root.secondMeasurement.kalman_variance:
    Type: value
    Python:     4.00252950373697
    TypeScript: 0.3557495820174625
    Difference: 3.647e+0
    Numeric difference exceeds tolerance: abs=3.647e+0, rel=91.111881%

```

### Test 2.3: Quality Scoring
**Description**: Verify quality score calculation matches

**Comparison**: ✗ Found 2 difference(s): 2 numeric, 0 structural

**Differences**:
```
Found 2 difference(s):

  root.measurement2.quality_score:
    Type: value
    Python:     0.9692612832885642
    TypeScript: 0.9642257045944764
    Difference: 5.036e-3
    Numeric difference exceeds tolerance: abs=5.036e-3, rel=0.519527%

  root.measurement2.quality_components.kalman_fit:
    Type: value
    Python:     0.9960954820081681
    TypeScript: 0.9829073472102502
    Difference: 1.319e-2
    Numeric difference exceeds tolerance: abs=1.319e-2, rel=1.323983%

```

### Test 2.4: Output Structure
**Description**: Verify both implementations return the same fields

**Comparison**: ✗ Found 2 difference(s): 0 numeric, 2 structural

**Differences**:
```
Found 2 difference(s):

  root.resultKeys:
    Type: missing
    Python:     [
  "accepted",
  "bmi_details",
  "confidence",
  "filtered_weight",
  "gap_days",
  "innovation",

    TypeScript: [
  "accepted",
  "bmi_details",
  "confidence",
  "filtered_weight",
  "innovation",
  "kalman_conf
    Array length mismatch: Python 25, TypeScript 21

  root.stateKeys:
    Type: missing
    Python:     [
  "kalman_params",
  "last_accepted_timestamp",
  "last_covariance",
  "last_raw_weight",
  "last_
    TypeScript: [
  "adaptation_state",
  "kalman_params",
  "last_accepted_timestamp",
  "last_covariance",
  "last
    Array length mismatch: Python 14, TypeScript 16

```

### Test 2.7: Timestamp Handling
**Description**: Verify timestamp conversion is consistent

**Comparison**: ✗ Found 2 difference(s): 0 numeric, 2 structural

**Differences**:
```
Found 2 difference(s):

  root.resultTimestamp:
    Type: type
    Python:     1699632000000
    TypeScript: 2023-11-10T16:00:00.000Z
    Type mismatch: Python number, TypeScript string

  root.stateTimestamp:
    Type: type
    Python:     1699632000000
    TypeScript: 2023-11-10T16:00:00.000Z
    Type mismatch: Python number, TypeScript string

```

## All Test Results

| Test Name | Status | Py Time | TS Time | Differences |
|-----------|--------|---------|---------|-------------|
| Test 2.1: Kalman Filter Initialization | ❌ | 214.56ms | 3046.69ms | 3 |
| Test 2.2: Kalman Filter Prediction | ❌ | 129.63ms | 1.91ms | 1 |
| Test 2.3: Quality Scoring | ❌ | 126.27ms | 0.99ms | 2 |
| Test 2.4: Output Structure | ❌ | 115.80ms | 0.55ms | 2 |
| Test 2.5: Reset Detection | ✅ | 118.90ms | 1.26ms | 0 |
| Test 2.6: Acceptance/Rejection Logic | ✅ | 111.97ms | 0.86ms | 0 |
| Test 2.7: Timestamp Handling | ❌ | 110.46ms | 0.63ms | 2 |
