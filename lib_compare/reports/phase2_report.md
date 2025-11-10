# Cross-Language Test Report
**Test Suite**: Phase 2: Component Tests
**Date**: 2025-11-10T14:59:49.288Z
**Total Tests**: 7
**Passed**: 6
**Failed**: 1
**Success Rate**: 85.7%
**Duration**: 1.98s

## Summary

| Status | Count | Percentage |
|--------|-------|------------|
| ✅ Passed | 6 | 85.7% |
| ❌ Failed | 1 | 14.3% |

## Performance Comparison

- **Python avg**: 122.86ms
- **TypeScript avg**: 160.24ms
- **Speed ratio**: Python is 0.23x faster

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
| Test 2.1: Kalman Filter Initialization | ✅ | 121.75ms | 1114.61ms | 0 |
| Test 2.2: Kalman Filter Prediction | ❌ | 137.85ms | 2.32ms | 1 |
| Test 2.3: Quality Scoring | ✅ | 127.88ms | 1.01ms | 0 |
| Test 2.4: Output Structure | ✅ | 117.90ms | 1.09ms | 0 |
| Test 2.5: Reset Detection | ✅ | 118.93ms | 1.06ms | 0 |
| Test 2.6: Acceptance/Rejection Logic | ✅ | 119.84ms | 0.93ms | 0 |
| Test 2.7: Timestamp Handling | ✅ | 115.87ms | 0.68ms | 0 |
