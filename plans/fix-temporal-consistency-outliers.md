# Plan: Fix Temporal Consistency for Outlier Detection

## Decision
**Approach**: Add temporal consistency validation that prevents contradictory outliers from being accepted within a time window
**Why**: Current system accepts/rejects outliers based solely on individual quality scores without considering temporal proximity, causing non-straight trend lines
**Risk Level**: Medium

## Implementation Steps

### Step 1: Add Temporal Consistency Validator
**File**: `src/outlier_detection.py`
**Action**: Create new method `_check_temporal_consistency()` after line 222
```python
def _check_temporal_consistency(
    self,
    measurement_idx: int,
    sorted_measurements: List[Dict[str, Any]],
    protected_indices: Set[int],
    window_hours: float = 24.0
) -> bool:
    """
    Check if protecting this measurement creates temporal inconsistency.
    Returns True if measurement should be allowed, False if it violates consistency.
    """
```

### Step 2: Implement Consistency Logic
**Location**: Inside new `_check_temporal_consistency()` method
**Logic**:
- Find all protected measurements within `window_hours` (default 24h)
- Calculate coefficient of variation (CV) for protected measurements
- If CV > 0.15 (15% variation), flag as inconsistent
- Check for contradictory patterns (e.g., 80kg and 120kg both protected)

### Step 3: Integrate into `detect_outliers()`
**File**: `src/outlier_detection.py:68-80`
**Changes**:
- After identifying protected indices by quality score (line 75)
- Before finalizing protection, validate temporal consistency
- Remove protection if measurement creates inconsistency
```python
# After line 79, add:
# Validate temporal consistency for protected measurements
validated_protected = set()
for idx in protected_indices:
    if self._check_temporal_consistency(idx, sorted_measurements, protected_indices):
        validated_protected.add(idx)
    else:
        # Log that protection was revoked due to temporal inconsistency
        metadata = sorted_measurements[idx].get('metadata', {})
        metadata['temporal_inconsistency'] = True
protected_indices = validated_protected
```

### Step 4: Add Sliding Window Analysis
**File**: `src/outlier_detection.py`
**Location**: Within `_check_temporal_consistency()` method
**Implementation**:
```python
# Get measurements in time window
current_time = sorted_measurements[measurement_idx]['timestamp']
window_start = current_time - timedelta(hours=window_hours/2)
window_end = current_time + timedelta(hours=window_hours/2)

window_measurements = []
for i, m in enumerate(sorted_measurements):
    if window_start <= m['timestamp'] <= window_end:
        if i in protected_indices:
            window_measurements.append(m['weight'])
```

### Step 5: Add Configuration Parameters
**File**: `config.toml`
**Location**: Under `[retrospective.outlier_detection]` section
```toml
temporal_window_hours = 24.0  # Window for consistency check
temporal_cv_threshold = 0.15  # Max coefficient of variation
temporal_range_threshold = 40.0  # Max kg range for protected measurements
```

### Step 6: Update Quality Scorer Context
**File**: `src/quality_scorer.py:96-149`
**Enhancement**: Add optional `nearby_protected_weights` parameter to `calculate_quality_score()`
- Pass nearby protected measurements for context
- Reduce quality score if measurement contradicts protected neighbors

### Step 7: Add Test Coverage
**File**: Create `tests/test_temporal_consistency.py`
**Tests**:
1. `test_contradictory_outliers_rejected()` - Both 80kg and 120kg with high quality
2. `test_consistent_trend_preserved()` - Gradual changes remain protected
3. `test_window_size_impact()` - Different window sizes
4. `test_quality_override_with_consistency()` - Integration test

## Files to Change
- `src/outlier_detection.py:222` - Add `_check_temporal_consistency()` method
- `src/outlier_detection.py:68-80` - Integrate consistency check into protection logic
- `src/outlier_detection.py:25-46` - Add config parameters to `__init__()`
- `config.toml` - Add temporal consistency parameters
- `src/quality_scorer.py:96` - Optional: Add nearby_protected_weights parameter
- `tests/test_temporal_consistency.py` - New test file for validation

## Acceptance Criteria
- [ ] Contradictory outliers (80kg and 120kg) cannot both be protected within 24h window
- [ ] Protected measurements within window have CV < 15%
- [ ] Range of protected measurements within window < 40kg
- [ ] Gradual legitimate changes still pass validation
- [ ] System logs when temporal inconsistency overrides quality protection
- [ ] All existing tests pass
- [ ] New test suite validates temporal consistency logic

## Risks & Mitigations
**Main Risk**: Rejecting legitimate rapid weight changes (e.g., medical interventions)
**Mitigation**:
- Make CV threshold configurable (default 0.15 is conservative)
- Log all temporal overrides for monitoring
- Add source-based exemptions (e.g., care-team-upload bypasses check)

**Secondary Risk**: Performance impact from window calculations
**Mitigation**:
- Cache window calculations per batch
- Limit window to reasonable size (24-48 hours)
- Only run for protected measurements, not all

## Out of Scope
- Reprocessing historical data with new logic
- Machine learning based anomaly detection
- Multi-user consistency validation
- Real-time streaming consistency (batch only)