# Cross-Language Test Report
**Test Suite**: Phase 2: Component Tests
<<<<<<< Updated upstream
**Date**: 2025-11-10T16:04:01.365Z
=======
**Date**: 2025-11-10T16:02:04.941Z
>>>>>>> Stashed changes
**Total Tests**: 7
**Passed**: 6
**Failed**: 1
**Success Rate**: 85.7%
<<<<<<< Updated upstream
**Duration**: 1.89s
=======
**Duration**: 1.86s
>>>>>>> Stashed changes

## Summary

| Status | Count | Percentage |
|--------|-------|------------|
| ✅ Passed | 6 | 85.7% |
| ❌ Failed | 1 | 14.3% |

## Performance Comparison

<<<<<<< Updated upstream
- **Python avg**: 121.44ms
- **TypeScript avg**: 149.12ms
- **Speed ratio**: Python is 0.19x faster
=======
- **Python avg**: 126.00ms
- **TypeScript avg**: 139.44ms
- **Speed ratio**: Python is 0.10x faster
>>>>>>> Stashed changes

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
<<<<<<< Updated upstream
| Test 2.1: Kalman Filter Initialization | ✅ | 103.47ms | 1037.99ms | 0 |
| Test 2.2: Kalman Filter Prediction | ❌ | 173.88ms | 1.94ms | 1 |
| Test 2.3: Quality Scoring | ✅ | 120.25ms | 0.91ms | 0 |
| Test 2.4: Output Structure | ✅ | 123.84ms | 0.59ms | 0 |
| Test 2.5: Reset Detection | ✅ | 115.84ms | 0.97ms | 0 |
| Test 2.6: Acceptance/Rejection Logic | ✅ | 107.17ms | 0.88ms | 0 |
| Test 2.7: Timestamp Handling | ✅ | 105.61ms | 0.55ms | 0 |
=======
| Test 2.1: Kalman Filter Initialization | ✅ | 101.92ms | 969.00ms | 0 |
| Test 2.2: Kalman Filter Prediction | ❌ | 159.15ms | 3.46ms | 1 |
| Test 2.3: Quality Scoring | ✅ | 143.75ms | 0.85ms | 0 |
| Test 2.4: Output Structure | ✅ | 123.57ms | 0.56ms | 0 |
| Test 2.5: Reset Detection | ✅ | 115.05ms | 0.88ms | 0 |
| Test 2.6: Acceptance/Rejection Logic | ✅ | 109.28ms | 0.81ms | 0 |
| Test 2.7: Timestamp Handling | ✅ | 129.27ms | 0.52ms | 0 |
>>>>>>> Stashed changes
