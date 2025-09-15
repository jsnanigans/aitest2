# Soft Reset Implementation - Step by Step

## Quick Summary
Add "soft reset" for manual data entry (questionnaires, user uploads, care team) that uses gentler adaptation parameters than hard resets.

## Implementation Steps

### 1. Create Reset Manager (`src/reset_manager.py`)
```python
from enum import Enum
from datetime import datetime
from typing import Dict, Optional, Tuple, Any

class ResetType(Enum):
    HARD = "hard"      # 30+ day gaps
    INITIAL = "initial" # First measurement  
    SOFT = "soft"      # Manual data entry

MANUAL_DATA_SOURCES = {
    'internal-questionnaire',
    'initial-questionnaire',
    'questionnaire',
    'patient-upload', 
    'user-upload',
    'care-team-upload',
    'care-team-entry'
}

class ResetManager:
    @staticmethod
    def should_trigger_reset(
        state: Dict[str, Any],
        weight: float,
        timestamp: datetime,
        source: str,
        config: Dict[str, Any]
    ) -> Optional[ResetType]:
        """Determine if and what type of reset to trigger."""
        
        # 1. Check for initial reset (no Kalman params)
        if not state.get('kalman_params'):
            return ResetType.INITIAL
        
        # 2. Check for hard reset (30+ day gap)
        hard_config = config.get('kalman', {}).get('reset', {}).get('hard', {})
        if hard_config.get('enabled', True):
            last_timestamp = state.get('last_accepted_timestamp') or state.get('last_timestamp')
            if last_timestamp:
                gap_days = (timestamp - last_timestamp).total_seconds() / 86400
                if gap_days >= hard_config.get('gap_threshold_days', 30):
                    return ResetType.HARD
        
        # 3. Check for soft reset (manual data with significant change)
        soft_config = config.get('kalman', {}).get('reset', {}).get('soft', {})
        if soft_config.get('enabled', True):
            if source in MANUAL_DATA_SOURCES:
                last_weight = state.get('last_accepted_weight')
                if last_weight:
                    weight_change = abs(weight - last_weight)
                    min_change = soft_config.get('min_change_kg', 5)
                    
                    if weight_change >= min_change:
                        # Check cooldown (no reset in last 3 days)
                        last_reset = get_last_reset_timestamp(state)
                        if not last_reset or (timestamp - last_reset).days > 3:
                            return ResetType.SOFT
        
        return None
    
    @staticmethod
    def perform_reset(
        state: Dict[str, Any],
        reset_type: ResetType,
        timestamp: datetime,
        weight: float,
        config: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Perform reset with appropriate parameters."""
        
        # Get parameters for this reset type
        reset_params = ResetManager.get_reset_parameters(reset_type, config)
        
        # Create reset event
        reset_event = {
            'timestamp': timestamp,
            'type': reset_type.value,
            'weight': weight,
            'last_weight': state.get('last_accepted_weight'),
            'parameters': reset_params
        }
        
        # Create new state with reset
        new_state = {
            'kalman_params': None,
            'last_state': None,
            'last_covariance': None,
            'measurements_since_reset': 0,
            'reset_type': reset_type.value,
            'reset_parameters': reset_params,
            'reset_timestamp': timestamp,
            'reset_events': state.get('reset_events', []) + [reset_event]
        }
        
        # Preserve some fields
        new_state['last_timestamp'] = state.get('last_timestamp')
        new_state['last_source'] = state.get('last_source')
        
        return new_state, reset_event
```

### 2. Update Processor (`src/processor.py`)

Replace lines checking for reset with:
```python
from .reset_manager import ResetManager, ResetType

# In process_measurement function:

# Check for any type of reset
reset_type = ResetManager.should_trigger_reset(
    state, cleaned_weight, timestamp, source, config
)

if reset_type:
    state, reset_event = ResetManager.perform_reset(
        state, reset_type, timestamp, cleaned_weight, config
    )
    reset_occurred = True
    
    # Log reset type
    print(f"Reset triggered: {reset_type.value} for user {user_id}")
```

### 3. Update Adaptive Parameters Usage

In `kalman_adaptive.py`, check reset type:
```python
def get_adaptive_kalman_params(
    state: Dict[str, Any],
    timestamp: datetime,
    base_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Get adaptive parameters based on reset type."""
    
    reset_type = state.get('reset_type', 'hard')
    reset_params = state.get('reset_parameters', {})
    
    if not reset_params:
        # Fallback to config
        reset_config = base_config.get('kalman', {}).get('reset', {}).get(reset_type, {})
        reset_params = {
            'weight_boost_factor': reset_config.get('weight_boost_factor', 10),
            'trend_boost_factor': reset_config.get('trend_boost_factor', 100),
            'decay_rate': reset_config.get('decay_rate', 3),
            'warmup_measurements': reset_config.get('warmup_measurements', 10),
            'adaptive_days': reset_config.get('adaptive_days', 7)
        }
    
    # Calculate adaptive parameters based on reset type...
```

### 4. Update Configuration (`config.toml`)

Add soft reset configuration:
```toml
[kalman.reset.hard]
# For 30+ day gaps (existing behavior)
enabled = true
gap_threshold_days = 30
weight_boost_factor = 10
trend_boost_factor = 100
decay_rate = 3
warmup_measurements = 10
adaptive_days = 7

[kalman.reset.initial]
# For first measurement (existing behavior)
enabled = true
weight_boost_factor = 10
trend_boost_factor = 100
decay_rate = 3
warmup_measurements = 10
adaptive_days = 7

[kalman.reset.soft]
# For manual data entry (NEW)
enabled = true
min_change_kg = 5        # Trigger if weight change > 5kg
weight_boost_factor = 3   # 3x boost (vs 10x for hard)
trend_boost_factor = 10   # 10x boost (vs 100x for hard)
decay_rate = 5           # Slower decay (vs 3)
warmup_measurements = 15  # More measurements (vs 10)
adaptive_days = 10       # Longer period (vs 7)

[quality_scoring.adaptive.hard]
# Quality scoring during hard reset
threshold = 0.4
plausibility_weight = 0.10
consistency_weight = 0.15

[quality_scoring.adaptive.soft]
# Quality scoring during soft reset (less lenient)
threshold = 0.5          # Higher than hard reset (0.4)
plausibility_weight = 0.15  # Higher than hard reset (0.10)
consistency_weight = 0.20   # Higher than hard reset (0.15)
```

## Testing Plan

### Test Scenarios
1. **Soft Reset Trigger**
   - Manual source + >5kg change → Soft reset
   - Manual source + <5kg change → No reset
   - Auto source + >5kg change → No reset

2. **Parameter Application**
   - Soft reset uses gentler parameters
   - Decay is slower for soft reset
   - Quality threshold higher for soft reset

3. **Cooldown Period**
   - No reset within 3 days of previous reset
   - Reset allowed after cooldown

## Benefits
- **Trust manual data more** - Gentler adaptation for user/care team input
- **Prevent over-correction** - Slower decay keeps stability
- **Reduce false rejections** - But not as lenient as hard reset
- **Maintainable** - All reset logic in one place

## Rollout
1. Implement ResetManager
2. Test with existing hard/initial resets
3. Add soft reset logic
4. Test with manual data scenarios
5. Deploy with monitoring