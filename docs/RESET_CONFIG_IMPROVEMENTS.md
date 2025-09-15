# Reset Configuration Improvements

## Summary

The reset configuration system has been significantly improved to provide better adaptation for initial measurements and clearer parameter naming throughout the system.

## Key Problems Solved

1. **Initial measurements were being rejected** - The initial reset wasn't adaptive enough
2. **Hardcoded values** - Many adaptation parameters were hardcoded instead of using config
3. **Unclear naming** - Parameter names like "weight_boost_factor" were confusing
4. **Config not being used** - Some config values were defined but never actually applied

## Improvements Made

### 1. Enhanced Initial Reset Configuration

The initial reset (for first measurements) is now much more adaptive:

**Old defaults:**
- weight_boost_factor: 10
- trend_boost_factor: 100
- quality_threshold: 0.4

**New defaults:**
- initial_variance_multiplier: 10 (properly applied now)
- weight_noise_multiplier: 50 (was 20)
- trend_noise_multiplier: 500 (was 200)
- observation_noise_multiplier: 0.3 (trust measurements more)
- quality_acceptance_threshold: 0.25 (was 0.4)
- adaptation_measurements: 20 (was 10)
- adaptation_days: 21 (was 10)

### 2. Clearer Parameter Names

**Old naming → New naming:**
- `weight_boost_factor` → `weight_noise_multiplier`
- `trend_boost_factor` → `trend_noise_multiplier`
- `warmup_measurements` → `adaptation_measurements`
- `adaptive_days` → `adaptation_days`
- `decay_rate` → `adaptation_decay_rate`
- `quality_threshold` → `quality_acceptance_threshold`

Added new parameters:
- `initial_variance_multiplier` - Multiplies initial state uncertainty
- `observation_noise_multiplier` - Scales measurement noise

### 3. Proper Config Usage

Fixed multiple places where config values were defined but not used:

- **kalman_adaptive.py**: Now properly uses multipliers from reset parameters
- **processor.py**: Uses adaptation thresholds and weights from reset parameters
- **reset_manager.py**: Properly reads all config values with sensible defaults

### 4. Quality Scoring During Adaptation

Quality scoring is now properly adjusted during the adaptation period:

- Uses `quality_acceptance_threshold` from reset parameters
- Component weights are configurable per reset type:
  - `quality_safety_weight`
  - `quality_plausibility_weight`
  - `quality_consistency_weight`
  - `quality_reliability_weight`

## Configuration Examples

### For Initial Measurements (Most Adaptive)

```toml
[kalman.reset.initial]
enabled = true
# Very high multipliers for maximum adaptation
initial_variance_multiplier = 10
weight_noise_multiplier = 50
trend_noise_multiplier = 500
observation_noise_multiplier = 0.3  # Trust measurements

# Long adaptation period
adaptation_measurements = 20
adaptation_days = 21
adaptation_decay_rate = 1.5

# Very lenient quality scoring
quality_acceptance_threshold = 0.25
quality_safety_weight = 0.50
quality_plausibility_weight = 0.05  # Almost ignore
quality_consistency_weight = 0.05   # Almost ignore
quality_reliability_weight = 0.40
```

### For Gap Recovery (Moderately Adaptive)

```toml
[kalman.reset.hard]
enabled = true
gap_threshold_days = 30

# Moderate multipliers
initial_variance_multiplier = 5
weight_noise_multiplier = 20
trend_noise_multiplier = 200
observation_noise_multiplier = 0.5

# Shorter adaptation
adaptation_measurements = 10
adaptation_days = 7
adaptation_decay_rate = 2.5

# Moderately lenient
quality_acceptance_threshold = 0.35
```

### For Manual Entry (Gentle Adaptation)

```toml
[kalman.reset.soft]
enabled = true
min_weight_change_kg = 5
cooldown_days = 3

# Small multipliers
initial_variance_multiplier = 2
weight_noise_multiplier = 5
trend_noise_multiplier = 20
observation_noise_multiplier = 0.7

# Gradual adaptation
adaptation_measurements = 15
adaptation_days = 10
adaptation_decay_rate = 4

# Slightly lenient
quality_acceptance_threshold = 0.45
```

## Files Modified

1. **src/reset_manager.py**
   - Updated default parameters for all reset types
   - Added support for new parameter names
   - Improved documentation

2. **src/kalman_adaptive.py**
   - Properly applies multipliers to base config
   - Uses measurement-based decay when available
   - Supports both old and new parameter names

3. **src/processor.py**
   - Uses reset parameters for adaptive period detection
   - Applies quality thresholds from reset parameters
   - Uses component weights from reset parameters

4. **config.toml**
   - Updated with better initial reset parameters
   - Added comments explaining each parameter

5. **config_improved.toml** (new)
   - Complete rewrite with clearer parameter names
   - Extensive documentation
   - Organized into logical sections

## Testing the Improvements

To verify the improvements work for user 091baa98-cf05-4399-b490-e24324f7607f:

```bash
# Process just this user
python main.py --user 091baa98-cf05-4399-b490-e24324f7607f

# Or process with the improved config
python main.py --config config_improved.toml
```

The initial measurements should now be accepted due to:
1. Much higher noise multipliers allowing rapid adaptation
2. Lower quality threshold (0.25 vs 0.6)
3. Longer adaptation period (20 measurements / 21 days)
4. Properly boosted initial variance

## Backward Compatibility

The system maintains backward compatibility:
- Old parameter names still work (e.g., `weight_boost_factor`)
- New parameter names are checked first, then fall back to old names
- Default values ensure the system works even without config

## Future Improvements

Consider:
1. Auto-tuning adaptation parameters based on data characteristics
2. Per-user adaptation profiles
3. Machine learning to predict optimal reset parameters
4. Visualization of adaptation progress
