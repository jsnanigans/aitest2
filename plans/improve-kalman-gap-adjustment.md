# Plan: Improve Kalman Filter Gap Adjustment - Hybrid Approach

## Summary
Implement a complete hybrid approach to improve Kalman filter performance after data gaps, combining warmup buffering, adaptive covariances, and trend estimation. This will reduce adjustment time from 20+ measurements to under 10 by properly managing state through the recursive measurement processing system.

## Context
- **Source**: Investigation report `reports/kalman_gap_adjustment_investigation.md`
- **Problem**: Kalman filter initializes with zero trend after gaps, causing slow adaptation
- **Impact**: Poor weight tracking for 2-3 weeks after 18+ day gaps
- **Root Cause**: Hard-coded zero trend + low covariances = overconfident wrong state
- **System Constraint**: Measurements processed one-by-one recursively, requiring careful state management

## Requirements

### Functional
- Improve post-gap RMSE from 0.386 kg to < 0.25 kg
- Achieve trend convergence within 10 measurements (currently 20+)
- Support configurable warmup period (3 measurements default)
- Handle recursive one-by-one measurement processing
- Maintain state consistency across measurement cycles
- Handle both short gaps (< 10 days) and long gaps (> 30 days)

### Non-functional
- No performance degradation for continuous data streams
- Maintain stateless architecture in processor.py (state only in database)
- Keep memory overhead minimal (< 1KB per user)
- Support concurrent user processing without conflicts
- State must be JSON-serializable for database storage

## Solution: Hybrid Approach (Option C)

### Overview
Combine three techniques into a unified system:
1. **Warmup Buffer**: Collect 3 measurements after gaps
2. **Adaptive Covariances**: Scale uncertainties with gap duration
3. **Trend Estimation**: Calculate initial trend from buffer
4. **Exponential Decay**: Gradually return to normal parameters

### Why This Approach
- Best overall performance (RMSE 0.206 kg vs 0.386 kg current)
- Handles the recursive nature of the system
- State management fits existing database architecture
- No need for staged rollout (pre-release system)

## State Management Design

### Core Challenge
The system processes measurements one-by-one recursively:
```python
for measurement in stream:
    state = load_state(user_id)
    result = process_measurement(state, measurement)
    save_state(user_id, state)
```
This requires careful state design to handle warmup buffering across multiple calls.

### State Structure Enhancement

#### Current State (in database)
```python
{
    "kalman_params": {...},
    "last_state": [...],
    "last_covariance": [...],
    "last_timestamp": "2024-01-01T00:00:00",
    "last_raw_weight": 85.0
}
```

#### Enhanced State (with gap handling)
```python
{
    "kalman_params": {...},
    "last_state": [...],
    "last_covariance": [...],
    "last_timestamp": "2024-01-01T00:00:00",
    "last_raw_weight": 85.0,
    
    # NEW: Gap handling state
    "gap_buffer": {
        "active": true,
        "gap_days": 18.5,
        "gap_detected_at": "2024-01-19T00:00:00",
        "measurements": [
            {"weight": 82.1, "timestamp": "2024-01-19T00:00:00", "source": "patient-device"},
            {"weight": 82.0, "timestamp": "2024-01-20T00:00:00", "source": "patient-device"}
        ],
        "target_size": 3,
        "timeout_days": 7
    },
    
    # NEW: Adaptation tracking
    "gap_adaptation": {
        "active": true,
        "gap_factor": 0.6,  # gap_days / 30, capped at 3
        "measurements_since_gap": 2,
        "initial_trend": -0.1,
        "decay_rate": 5.0
    }
}
```

### State Lifecycle

#### 1. Normal Processing (No Gap)
```
Load State → Check Gap → Process Normally → Save State
```

#### 2. Gap Detection
```
Load State → Detect Gap → Initialize Buffer → Store Measurement → Save Enhanced State
```

#### 3. Buffer Collection Phase
```
Load State → Check Buffer → Add to Buffer → Check Completion → Save State
                                ↓
                    If Complete: Initialize Kalman with Buffer Data
```

#### 4. Adaptation Phase
```
Load State → Apply Adaptive Parameters → Process → Decay Parameters → Save State
```

### Database Schema Changes

Add new columns to state storage:
```sql
-- Option 1: Single JSON column (Recommended)
ALTER TABLE user_states ADD COLUMN gap_handling JSON;

-- Option 2: Separate columns
ALTER TABLE user_states ADD COLUMN gap_buffer JSON;
ALTER TABLE user_states ADD COLUMN gap_adaptation JSON;
```

### State Management Implementation

#### Gap Detection and Buffer Initialization
```python
def handle_gap_detection(state, current_timestamp, gap_threshold=10):
    """Detect gap and initialize buffer if needed."""
    if state.get('last_timestamp'):
        gap_days = calculate_gap_days(state['last_timestamp'], current_timestamp)
        
        if gap_days > gap_threshold:
            # Initialize gap buffer
            state['gap_buffer'] = {
                'active': True,
                'gap_days': gap_days,
                'gap_detected_at': current_timestamp.isoformat(),
                'measurements': [],
                'target_size': 3,  # From config
                'timeout_days': 7
            }
            # Store pre-gap trend if available
            if state.get('last_state'):
                _, trend = get_current_state_values(state)
                state['gap_buffer']['pre_gap_trend'] = trend
    
    return state
```

#### Buffer Management
```python
def update_gap_buffer(state, weight, timestamp, source):
    """Add measurement to buffer and check completion."""
    buffer = state.get('gap_buffer', {})
    
    if not buffer.get('active'):
        return state, False
    
    # Add measurement
    buffer['measurements'].append({
        'weight': weight,
        'timestamp': timestamp.isoformat(),
        'source': source
    })
    
    # Check completion conditions
    is_complete = False
    if len(buffer['measurements']) >= buffer['target_size']:
        is_complete = True
    elif len(buffer['measurements']) >= 2:
        # Check timeout
        first_timestamp = datetime.fromisoformat(buffer['measurements'][0]['timestamp'])
        time_span_days = (timestamp - first_timestamp).total_seconds() / 86400
        if time_span_days >= buffer['timeout_days']:
            is_complete = True
    
    state['gap_buffer'] = buffer
    return state, is_complete
```

#### Kalman Initialization from Buffer
```python
def initialize_from_buffer(state):
    """Initialize Kalman with buffered measurements."""
    buffer = state['gap_buffer']
    measurements = buffer['measurements']
    
    # Extract data
    weights = [m['weight'] for m in measurements]
    timestamps = [datetime.fromisoformat(m['timestamp']) for m in measurements]
    
    # Estimate initial values
    initial_weight = np.median(weights)
    initial_trend = estimate_trend(weights, timestamps, buffer.get('pre_gap_trend'))
    
    # Calculate adaptive parameters
    gap_factor = min(buffer['gap_days'] / 30, 3.0)
    
    # Initialize Kalman state
    state['kalman_params'] = create_adaptive_kalman_params(
        initial_weight, initial_trend, gap_factor
    )
    state['last_state'] = np.array([[initial_weight, initial_trend]])
    state['last_covariance'] = create_adaptive_covariance(gap_factor)
    
    # Set up adaptation tracking
    state['gap_adaptation'] = {
        'active': True,
        'gap_factor': gap_factor,
        'measurements_since_gap': len(measurements),
        'initial_trend': initial_trend,
        'decay_rate': 5.0
    }
    
    # Clear buffer
    state['gap_buffer']['active'] = False
    
    return state
```

#### Adaptive Parameter Decay
```python
def apply_adaptation_decay(state):
    """Apply exponential decay to adaptive parameters."""
    if not state.get('gap_adaptation', {}).get('active'):
        return state
    
    adaptation = state['gap_adaptation']
    measurements_since = adaptation['measurements_since_gap']
    
    # Calculate decay factor
    decay_factor = np.exp(-measurements_since / adaptation['decay_rate'])
    
    if decay_factor > 0.01:  # Still adapting
        gap_factor = adaptation['gap_factor']
        
        # Update transition covariances with decay
        base_weight_cov = KALMAN_DEFAULTS['transition_covariance_weight']
        base_trend_cov = KALMAN_DEFAULTS['transition_covariance_trend']
        
        state['kalman_params']['transition_covariance'] = [
            [base_weight_cov * (1 + gap_factor * 3 * decay_factor), 0],
            [0, base_trend_cov * (1 + gap_factor * 50 * decay_factor)]
        ]
        
        adaptation['measurements_since_gap'] += 1
    else:
        # End adaptation
        state['gap_adaptation']['active'] = False
        # Reset to normal parameters
        state['kalman_params']['transition_covariance'] = [
            [KALMAN_DEFAULTS['transition_covariance_weight'], 0],
            [0, KALMAN_DEFAULTS['transition_covariance_trend']]
        ]
    
    return state
```

### Configuration
```toml
[kalman.gap_handling]
enabled = true
gap_threshold_days = 10  # When to trigger gap handling
warmup_size = 3  # Measurements to collect
max_warmup_days = 7  # Max time to wait for warmup
adaptive_covariance = true
gap_variance_multiplier = 2.0
trend_variance_multiplier = 20.0
adaptation_decay_rate = 5.0  # Measurements for 63% decay
```

## Implementation Plan (No Code)

### Processing Flow with State Management

#### Measurement Processing Lifecycle
```
1. Load user state from database
2. Check for gap (compare timestamps)
3. If gap detected:
   - Initialize gap_buffer in state
   - Set buffer active flag
4. If buffer active:
   - Add measurement to buffer
   - Check if buffer complete (3 measurements or timeout)
   - If complete: Initialize Kalman from buffer
   - If not: Save state and return preliminary result
5. If adaptation active:
   - Apply adaptive parameters
   - Process measurement
   - Decay adaptation
6. Save updated state to database
7. Return result
```

### Implementation Steps

#### Step 1: Database State Structure
**Files**: `src/database.py`
- Extend state dictionary with `gap_buffer` and `gap_adaptation` fields
- Ensure JSON serialization handles numpy arrays
- Add validation for new state fields
- Maintain backward compatibility (handle missing fields)

#### Step 2: Gap Detection Logic
**Files**: `src/processor.py`
- Enhance gap detection to initialize buffer state
- Store gap metadata (gap_days, detection timestamp)
- Preserve pre-gap trend for later use
- Handle different gap thresholds (10 days normal, 30 days reset)

#### Step 3: Buffer Management System
**Files**: `src/processor.py`, new `src/gap_handler.py`
- Create buffer management functions
- Add measurement to buffer array
- Check completion conditions (size or timeout)
- Handle buffer overflow scenarios
- Clear buffer after use

#### Step 4: Enhanced Kalman Initialization
**Files**: `src/kalman.py`
- Add `initialize_from_buffer()` method
- Implement trend estimation from multiple points
- Calculate adaptive parameters based on gap duration
- Set up initial state with estimated trend
- Initialize adaptation tracking

#### Step 5: Adaptive Parameter System
**Files**: `src/kalman.py`
- Add `calculate_adaptive_parameters()` function
- Implement exponential decay calculation
- Update transition covariances dynamically
- Track measurements since gap
- Detect when to end adaptation

#### Step 6: Processor Integration
**Files**: `src/processor.py`
- Modify main processing loop for buffer handling
- Route measurements based on state flags
- Handle buffer completion transition
- Apply adaptation during normal processing
- Ensure state consistency

#### Step 7: Result Generation
**Files**: `src/processor.py`, `src/kalman.py`
- Generate preliminary results during buffer phase
- Include buffer status in result metadata
- Add adaptation metrics to results
- Maintain backward compatibility

### Detailed State Transitions

#### State A: Normal Processing
```python
state = {
    "kalman_params": {...},
    "last_timestamp": "2024-01-01",
    # No gap_buffer or gap_adaptation
}
```

#### State B: Gap Detected, Buffer Active
```python
state = {
    "kalman_params": None,  # Not yet initialized
    "last_timestamp": "2024-01-01",
    "gap_buffer": {
        "active": True,
        "gap_days": 18.5,
        "measurements": [
            {"weight": 82.1, "timestamp": "2024-01-19", "source": "device"}
        ],
        "target_size": 3
    }
}
```

#### State C: Buffer Complete, Adaptation Active
```python
state = {
    "kalman_params": {...},  # Initialized from buffer
    "last_timestamp": "2024-01-21",
    "gap_buffer": {
        "active": False,  # Completed
        "measurements": [...]  # Keep for debugging
    },
    "gap_adaptation": {
        "active": True,
        "gap_factor": 0.6,
        "measurements_since_gap": 3,
        "decay_rate": 5.0
    }
}
```

#### State D: Adaptation Complete, Back to Normal
```python
state = {
    "kalman_params": {...},  # Normal parameters
    "last_timestamp": "2024-02-01",
    "gap_adaptation": {
        "active": False,  # Completed
        "measurements_since_gap": 15
    }
}
```

### Edge Cases and Handling

#### Edge Case 1: Single Measurement After Long Gap
- Buffer timeout after 7 days
- Initialize with single point (zero trend)
- Use maximum adaptive parameters

#### Edge Case 2: Outlier in Buffer
- Use median instead of mean for robustness
- Detect outliers using IQR method
- Option to extend buffer if outlier detected

#### Edge Case 3: Multiple Gaps in Sequence
- Reset buffer for each new gap
- Preserve adaptation state if still active
- Compound gap factors with maximum cap

#### Edge Case 4: State Corruption
- Validate state structure on load
- Fallback to fresh initialization if corrupt
- Log corruption events for debugging

### Testing Strategy

#### Unit Tests
- Gap detection with various durations
- Buffer management operations
- Trend estimation accuracy
- Adaptive parameter calculations
- State serialization/deserialization

#### Integration Tests
- Full processing cycle with gap
- Multiple users with different gap patterns
- State persistence across restarts
- Concurrent processing safety

#### Scenario Tests
- 18-day gap with declining trend
- 45-day gap with flat trend
- Multiple small gaps (5-10 days)
- Questionnaire source gaps
- Mixed source types in buffer

## Validation & Testing

### Test Data Scenarios
1. **Standard Gap (18 days)**
   - Pre-gap: 30 days declining trend (-0.1 kg/day)
   - Gap: 18 days
   - Post-gap: 20 days continuing trend
   - Expected: RMSE < 0.25 kg, convergence < 10 measurements

2. **Long Gap (45 days)**
   - Tests maximum adaptation
   - Buffer timeout handling
   - Trend decay assumptions

3. **Short Gap (5 days)**
   - Below threshold, no buffer needed
   - Normal processing continues

4. **Multiple Sequential Gaps**
   - 10-day gap → 5 measurements → 15-day gap
   - Tests state management robustness

### Validation Metrics
- **Accuracy**: Post-gap RMSE < 0.25 kg
- **Convergence**: Trend within 0.03 kg/day of actual within 10 measurements
- **Robustness**: No state corruption across 1000 gap scenarios
- **Performance**: < 10ms additional processing time per measurement
- **Memory**: < 1KB additional state per user

### Implementation Validation Checklist
- [ ] State serialization handles all data types
- [ ] Buffer correctly accumulates measurements
- [ ] Trend estimation produces reasonable values
- [ ] Adaptive parameters decay smoothly
- [ ] Edge cases handled gracefully
- [ ] No memory leaks in buffer management
- [ ] Concurrent user processing safe
- [ ] Results include buffer/adaptation metadata

## Risks & Mitigations

### Risk 1: State Corruption During Buffer Phase
**Impact**: High - Loss of measurements, incorrect initialization
**Mitigation**: 
- Validate state structure on every load
- Keep buffer measurements immutable (append-only)
- Implement state recovery mechanism
- Add comprehensive logging

### Risk 2: Buffer Never Completes
**Impact**: Medium - Stuck in buffer phase indefinitely
**Mitigation**:
- 7-day timeout mechanism
- Minimum 2 measurements for initialization
- Fallback to single-point initialization
- Alert on timeout events

### Risk 3: Trend Estimation Errors
**Impact**: Medium - Poor initial state
**Mitigation**:
- Use median for robustness
- Cap trend estimates at reasonable bounds
- Blend with pre-gap trend
- Increase uncertainty for extreme trends

### Risk 4: State Size Growth
**Impact**: Low - Database storage concerns
**Mitigation**:
- Limit buffer to 5 measurements max
- Clear completed buffers after 30 days
- Compress old adaptation data
- Monitor state size metrics

## Acceptance Criteria

### Core Functionality
- [ ] Gap detection triggers at configurable threshold (default 10 days)
- [ ] Buffer collects exactly 3 measurements (configurable)
- [ ] Buffer timeout works at 7 days
- [ ] Trend estimation from buffer produces reasonable values (-0.5 to 0.5 kg/day)
- [ ] Adaptive parameters scale with gap duration
- [ ] Exponential decay reduces adaptation over ~10 measurements

### State Management
- [ ] State persists correctly across measurement cycles
- [ ] Buffer state survives system restart
- [ ] Concurrent users don't interfere with each other
- [ ] State size < 2KB per user with active buffer
- [ ] Old buffer data cleaned up after completion

### Performance Metrics
- [ ] Post-gap RMSE < 0.25 kg (from 0.386 kg baseline)
- [ ] Trend convergence within 10 measurements
- [ ] Processing time < 10ms per measurement
- [ ] No memory leaks over 10,000 measurements
- [ ] State serialization < 1ms

### Edge Cases
- [ ] Single measurement after gap initializes correctly
- [ ] Multiple gaps in sequence handled properly
- [ ] Outliers in buffer don't crash system
- [ ] Corrupt state recovers gracefully
- [ ] Source type changes during buffer handled

### System Integration
- [ ] Results include buffer status metadata
- [ ] Visualization shows buffer phase clearly
- [ ] Config.toml controls all parameters
- [ ] Logging captures gap events
- [ ] No regression in continuous data processing

## Out of Scope
- Machine learning predictions
- Historical data reprocessing
- Multi-user pattern detection
- Predictive gap filling
- Custom per-user parameters

## Open Questions

### Critical Decisions Needed
1. **Buffer Storage**: Should buffer be in main state dict or separate column?
   - Option A: Embedded in state dict (simpler)
   - Option B: Separate `gap_buffer` column (cleaner)
   - **Recommendation**: Embedded for simplicity

2. **Incomplete Buffer Handling**: What to return during buffer collection?
   - Option A: No result until buffer complete
   - Option B: Raw measurements only
   - Option C: Simple average as preliminary result
   - **Recommendation**: Option C with metadata flag

3. **Pre-gap Trend Usage**: How much to weight pre-gap trend?
   - Current proposal: 30% pre-gap, 70% measured
   - Alternative: Exponential decay based on gap duration
   - **Recommendation**: Exponential decay

### Implementation Questions
4. **State Validation**: How often to validate state structure?
   - Every load (safe but slower)
   - Only on errors (faster but risky)
   - **Recommendation**: Every load with caching

5. **Buffer Outlier Detection**: How to handle outliers in buffer?
   - Reject and extend buffer
   - Use median (current proposal)
   - Statistical outlier removal
   - **Recommendation**: Use median for robustness

6. **Source-Specific Settings**: Different parameters per source?
   - Same gap threshold for all
   - Questionnaires: 10 days, Others: 30 days
   - **Recommendation**: Configurable per source

## Implementation Sequence

### Step-by-Step Implementation Order

1. **State Structure Setup** (Day 1)
   - Extend database state schema
   - Add gap_buffer and gap_adaptation fields
   - Implement state validation
   - Test serialization/deserialization

2. **Gap Detection Enhancement** (Day 1)
   - Modify processor gap detection
   - Initialize buffer on gap
   - Store gap metadata
   - Test gap detection logic

3. **Buffer Management** (Day 2)
   - Implement buffer addition logic
   - Add completion detection
   - Handle timeout cases
   - Test buffer operations

4. **Kalman Initialization** (Day 2-3)
   - Create initialize_from_buffer method
   - Implement trend estimation
   - Add adaptive parameter calculation
   - Test initialization accuracy

5. **Adaptation System** (Day 3)
   - Implement decay calculations
   - Update parameters dynamically
   - Track adaptation state
   - Test convergence behavior

6. **Integration** (Day 4)
   - Wire everything in processor.py
   - Handle all state transitions
   - Add result metadata
   - Full integration testing

7. **Validation** (Day 4-5)
   - Run test scenarios
   - Measure performance
   - Validate accuracy
   - Document results

## Code Organization

### New Functions to Add

#### In `src/kalman.py`:
- `calculate_adaptive_parameters(gap_days: float) -> Dict`
- `estimate_trend_from_buffer(measurements: List, timestamps: List, pre_gap_trend: Optional[float]) -> float`
- `initialize_from_buffer(buffer_data: Dict, gap_days: float, kalman_config: Dict) -> Dict`
- `apply_adaptation_decay(state: Dict) -> Dict`

#### In `src/processor.py`:
- `handle_gap_detection(state: Dict, current_timestamp: datetime, config: Dict) -> Dict`
- `update_gap_buffer(state: Dict, weight: float, timestamp: datetime, source: str) -> Tuple[Dict, bool]`
- `process_buffer_completion(state: Dict, config: Dict) -> Dict`

#### New file `src/gap_handler.py` (optional):
- Consolidate all gap-handling logic
- Keep processor.py focused on main flow
- Easier testing and maintenance

## Council Review

**Butler Lampson** (Simplicity): "The state management is getting complex. Consider whether the full buffer approach is necessary, or if adaptive covariances alone might suffice."

**Leslie Lamport** (Distributed Systems): "The append-only buffer design is good. Ensure state transitions are atomic - never partially update the state."

**Donald Knuth** (Algorithms): "The trend estimation should use robust statistics. Median is good, but consider Theil-Sen estimator for trend."

**Kent Beck** (Testing): "Build the simplest version first - just adaptive covariances. Only add buffering if tests show it's needed."

## Final Recommendations

### Immediate Action (This Week)
1. Implement state structure changes
2. Add gap detection and buffer management
3. Create adaptive initialization
4. Test with real gap scenarios
5. Measure improvement metrics

### Success Metrics
- Primary: RMSE < 0.25 kg after gaps
- Secondary: Trend convergence < 10 measurements
- Tertiary: No performance regression

### Go/No-Go Decision Points
- After buffer implementation: Is complexity justified?
- After trend estimation: Does it improve over median?
- After full integration: Does it meet acceptance criteria?

This is a pre-release system, so we can iterate quickly without migration concerns. Focus on getting the core functionality working, then optimize based on real data.