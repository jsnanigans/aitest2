# Replay System Fix Summary

## Issue Fixed
The replay system was never being triggered due to method name mismatches between the pipeline and the actual replay components.

## Root Causes
1. **Method Mismatch**: Pipeline called non-existent methods `validate_replay()` and `process_replay()` instead of `replay_clean_measurements()`
2. **Parameter Order**: OutlierDetector.detect_outliers() was called with parameters in wrong order (user_id, measurements instead of measurements, user_id)
3. **Buffer Method Names**: Pipeline used wrong method names for ReplayBuffer (e.g., `should_trigger()` instead of checking buffer info)
4. **Return Type Confusion**: OutlierDetector returns indices, not measurement objects

## Changes Made

### 1. Fixed Pipeline Method Calls (`src/processing/pipeline.py`)
- Line 343-358: Fixed buffer trigger check to use `get_buffer_info()` and `get_buffer_measurements()`
- Line 376-408: Replaced non-existent replay methods with proper `replay_clean_measurements()` call
- Line 391: Fixed parameter order for `detect_outliers(measurements, user_id)`
- Line 394: Fixed outlier filtering to handle indices correctly
- Line 417: Changed `clear_user()` to `clear_buffer()`

### 2. Configuration Update (`config.toml`)
- Line 141: Increased `buffer_hours` from 1 to 24 for better outlier detection window

## Test Results
✅ **Test Passed**: The problematic BMI value (34.56kg) on 2025-04-10 was correctly:
- Rejected by quality scoring (score 0.03 below threshold)
- Identified as an outlier by the replay system
- Filtered out, allowing the correct ~79kg measurements to be accepted

## Impact
- Replay system now properly triggers and processes measurements
- Outliers are correctly identified and filtered
- Clean measurements are replayed through the Kalman filter
- This fixes data quality issues for all users, not just the test case

## Verification
Run the test with:
```bash
uv run python test_replay_fix.py
```

Expected output:
- BMI value (34.56kg) rejected
- Normal weights (~79kg) accepted
- No abnormally high weights in final data