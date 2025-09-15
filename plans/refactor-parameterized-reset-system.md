# Plan: Parameterized Reset System for Multiple Scenarios

## Summary
Refactor the reset system to be reusable with different parameters for different triggers:
1. **Hard Reset** (30+ day gaps) - Current behavior
2. **Initial Reset** (first measurement) - Current behavior  
3. **Soft Reset** (manual data entry) - New, gentler adaptation

## Context
- Current system has reset logic scattered across multiple places
- Manual data (questionnaires, user uploads, care team uploads) is more trustworthy
- Need different adaptation parameters for different scenarios
- Want to maintain backward compatibility

## Requirements

### Functional
- Create reusable reset function with configurable parameters
- Support three reset types: hard, initial, soft
- Trigger soft reset on manual data sources
- Allow different boost factors and decay rates per reset type
- Maintain state tracking for reset events

### Non-functional  
- Clean, modular design
- Easy to add new reset types
- Configurable via config.toml
- Backward compatible

## Design

### Reset Types
```python
class ResetType(Enum):
    HARD = "hard"      # 30+ day gaps
    INITIAL = "initial" # First measurement
    SOFT = "soft"      # Manual data entry
```

### Manual Data Sources
```python
MANUAL_DATA_SOURCES = {
    # Questionnaires
    'internal-questionnaire',
    'initial-questionnaire', 
    'questionnaire',
    
    # User uploads
    'patient-upload',
    'user-upload',
    
    # Care team
    'care-team-upload',
    'care-team-entry'
}
```

### Reset Parameters Structure
```python
{
    'type': ResetType,
    'weight_boost_factor': float,  # Multiplier for weight covariance
    'trend_boost_factor': float,   # Multiplier for trend covariance
    'decay_rate': float,           # Exponential decay rate
    'warmup_measurements': int,    # Number of measurements to adapt over
    'adaptive_days': int,          # Days to stay adaptive
    'quality_threshold': float,    # Quality score threshold during adaptation
}
```

## Implementation Plan

### Step 1: Create Reset Manager Module
**File**: `src/reset_manager.py`
```python
class ResetManager:
    @staticmethod
    def should_trigger_reset(state, timestamp, source, config):
        """Determine if and what type of reset to trigger."""
        # Check hard reset (30+ day gap)
        # Check initial reset (no kalman_params)
        # Check soft reset (manual source + significant change)
        
    @staticmethod
    def perform_reset(state, reset_type, timestamp, config):
        """Perform reset with appropriate parameters."""
        # Get parameters for reset type from config
        # Clear appropriate state fields
        # Set adaptation parameters
        # Track reset event
        
    @staticmethod
    def get_reset_parameters(reset_type, config):
        """Get parameters for specific reset type from config."""
```

### Step 2: Update Processor Integration
**File**: `src/processor.py`

Replace current reset logic with:
```python
# Check for any type of reset
reset_type = ResetManager.should_trigger_reset(
    state, timestamp, source, config
)

if reset_type:
    state, reset_event = ResetManager.perform_reset(
        state, reset_type, timestamp, config
    )
    reset_occurred = True
```

### Step 3: Configuration Structure
**File**: `config.toml`
```toml
[kalman.reset.hard]
# For 30+ day gaps
enabled = true
gap_threshold_days = 30
weight_boost_factor = 10
trend_boost_factor = 100
decay_rate = 3
warmup_measurements = 10
adaptive_days = 7
quality_threshold = 0.4

[kalman.reset.initial]
# For first measurement
enabled = true
weight_boost_factor = 10
trend_boost_factor = 100
decay_rate = 3
warmup_measurements = 10
adaptive_days = 7
quality_threshold = 0.4

[kalman.reset.soft]
# For manual data entry
enabled = true
trigger_sources = ["questionnaire", "patient-upload", "care-team-upload"]
min_change_kg = 5  # Minimum change to trigger soft reset
weight_boost_factor = 3   # Gentler boost (vs 10)
trend_boost_factor = 10   # Gentler boost (vs 100)
decay_rate = 5            # Slower decay (vs 3)
warmup_measurements = 15  # Longer adaptation (vs 10)
adaptive_days = 10        # Longer period (vs 7)
quality_threshold = 0.5   # Less lenient (vs 0.4)
```

### Step 4: Soft Reset Trigger Logic
Trigger soft reset when:
1. Source is in manual data sources list
2. AND weight change from last accepted > threshold (e.g., 5kg)
3. AND no recent reset (avoid reset loops)

```python
def should_trigger_soft_reset(state, weight, source, config):
    # Check if source is manual
    if source not in MANUAL_DATA_SOURCES:
        return False
    
    # Check if significant change
    last_weight = state.get('last_accepted_weight')
    if last_weight and abs(weight - last_weight) > config['min_change_kg']:
        
        # Check no recent reset (within 3 days)
        last_reset = get_last_reset_timestamp(state)
        if not last_reset or days_since(last_reset) > 3:
            return True
    
    return False
```

### Step 5: State Tracking
Track reset events with type:
```python
reset_event = {
    'timestamp': timestamp,
    'type': reset_type.value,
    'trigger': 'gap' | 'initial' | 'manual_entry',
    'source': source,
    'weight_change': weight_change,
    'parameters': reset_params
}
```

## Benefits
1. **Modular**: Single place to manage all reset logic
2. **Flexible**: Easy to add new reset types or adjust parameters
3. **Intelligent**: Different adaptation for different data sources
4. **User-friendly**: Trusts manual data more than automatic
5. **Maintainable**: Clean separation of concerns

## Testing Strategy
1. **Unit tests** for ResetManager functions
2. **Integration tests** for each reset type:
   - Hard reset after 30+ day gap
   - Initial reset on first measurement
   - Soft reset on manual data with large change
3. **Edge cases**:
   - Multiple resets in succession
   - Manual data with small changes (no reset)
   - Mixed sources

## Migration Path
1. Create ResetManager module
2. Add soft reset configuration
3. Update processor to use ResetManager
4. Test with existing data
5. Deploy with monitoring

## Risks & Mitigations
- **Risk**: Too many soft resets
  - **Mitigation**: Minimum change threshold, cooldown period
- **Risk**: Configuration complexity
  - **Mitigation**: Sensible defaults, clear documentation
- **Risk**: Breaking existing behavior
  - **Mitigation**: Extensive testing, gradual rollout

## Success Criteria
- [ ] All three reset types working correctly
- [ ] Manual data triggers appropriate soft reset
- [ ] Configuration is clear and maintainable
- [ ] No regression in existing reset behavior
- [ ] Improved acceptance rate for manual data entries

## Future Enhancements
- Reset confidence levels (how sure are we a reset is needed)
- Learning from reset patterns per user
- Automatic parameter tuning based on outcomes
- Reset reasons in visualization