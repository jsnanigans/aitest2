# Cross-Language Test Report
**Test Suite**: Phase 2: Component Tests
**Date**: 2025-11-10T16:04:01.365Z
**Total Tests**: 7
**Passed**: 6
**Failed**: 1
**Success Rate**: 85.7%
**Duration**: 1.89s

## Summary

| Status | Count | Percentage |
|--------|-------|------------|
| ✅ Passed | 6 | 85.7% |
| ❌ Failed | 1 | 14.3% |

## Performance Comparison

- **Python avg**: 121.44ms
- **TypeScript avg**: 149.12ms
- **Speed ratio**: Python is 0.19x faster

## Failed Tests

### Test 2.2: Kalman Filter Prediction
**Description**: Verify Kalman filter prediction step (state extrapolation)

**Comparison**: ✗ Found 1 difference(s): 1 numeric, 0 structural

**Differences**:
```
Found 1 difference(s):

  root.secondMeasurement.kalman_variance:
    Type: value
    Python:     4.00252950373697
    TypeScript: 2.379729588093491
    Difference: 1.623e+0
    Numeric difference exceeds tolerance: abs=1.623e+0, rel=40.544359%

```

## All Test Results

| Test Name | Status | Py Time | TS Time | Differences |
|-----------|--------|---------|---------|-------------|
| Test 2.1: Kalman Filter Initialization | ✅ | 103.47ms | 1037.99ms | 0 |
| Test 2.2: Kalman Filter Prediction | ❌ | 173.88ms | 1.94ms | 1 |
| Test 2.3: Quality Scoring | ✅ | 120.25ms | 0.91ms | 0 |
| Test 2.4: Output Structure | ✅ | 123.84ms | 0.59ms | 0 |
| Test 2.5: Reset Detection | ✅ | 115.84ms | 0.97ms | 0 |
| Test 2.6: Acceptance/Rejection Logic | ✅ | 107.17ms | 0.88ms | 0 |
| Test 2.7: Timestamp Handling | ✅ | 105.61ms | 0.55ms | 0 |
