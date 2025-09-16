# Feature Toggle System Documentation

## Overview

A comprehensive feature toggle system has been implemented for the weight processing pipeline. This allows fine-grained control over individual features through the `config.toml` file.

## How to Use

### Basic Usage

Edit the `[features]` section in `config.toml` to enable/disable features:

```toml
[features]
# Core processing features
kalman_filtering = false          # Disable Kalman filtering
quality_scoring = false            # Disable quality assessment
outlier_detection = true           # Keep outlier detection enabled
```

### Available Features

#### Core Processing
- `kalman_filtering` - Core Kalman filter for weight smoothing
- `quality_scoring` - Multi-factor quality assessment system
- `outlier_detection` - Statistical outlier detection
- `quality_override` - Allow high-quality scores to override outlier detection

#### Validation Features
```toml
[features.validation]
enabled = true                    # Master validation switch
physiological = true              # Hard limits (30-400kg)
bmi_checking = true               # BMI plausibility checks
rate_limiting = true              # Daily/weekly change limits
temporal_consistency = true       # Temporal consistency checks
session_variance = true           # Session-based variance detection
```

#### Outlier Detection Methods
```toml
[features.outlier_methods]
iqr = true                        # Interquartile range detection
mad = true                        # Median absolute deviation
temporal = true                   # Temporal change detection
kalman_deviation = true           # Deviation from Kalman prediction
```

#### Quality Scoring Components
```toml
[features.quality_components]
safety = true                     # Safety score (physiological limits)
plausibility = true               # Plausibility score (BMI, trends)
consistency = true                # Consistency score (change rates)
reliability = true                # Source-based reliability
```

#### State Management
```toml
[features.state]
persistence = true                # Save user states to database
history_buffer = true             # Maintain measurement history
reset_tracking = true             # Track reset events
```

#### Reset Management
```toml
[features.resets]
initial = true                    # Enable initial reset for new users
hard = true                       # Enable hard reset after long gaps
soft = true                       # Enable soft reset for manual entries
```

#### Adaptive Features
```toml
[features.adaptive]
noise_model = true                # Source-specific noise multipliers
parameters = true                 # Adaptive Kalman parameters
parameter_decay = true            # Parameter decay over time
```

#### Retrospective Processing
```toml
[features.retrospective]
enabled = true                    # Enable retrospective analysis
outlier_detection = true          # Retrospective outlier detection
rollback = true                   # Allow state rollback
```

## Implementation Details

### FeatureManager Class

The `FeatureManager` class (`src/feature_manager.py`) provides:
- Centralized feature toggle management
- Dependency resolution (e.g., quality_scoring requires kalman_filtering)
- Configuration validation
- Default values for backward compatibility

### Integration Points

Feature checks have been added to:
1. **processor.py** - Main processing pipeline
2. **outlier_detection.py** - Outlier detection methods
3. **quality_scorer.py** - Quality scoring components
4. **validation.py** - Validation layers
5. **kalman.py** - Reset management

### Dependencies

Some features depend on others:
- `quality_scoring` requires `kalman_filtering`
- `quality_override` requires both `quality_scoring` and `outlier_detection`
- `kalman_deviation_check` requires `kalman_filtering`
- `adaptive_parameters` requires `kalman_filtering`
- `reset_tracking` requires `state_persistence`
- `history_buffer` requires `state_persistence`

Dependencies are automatically resolved - if you enable a dependent feature, its requirements will be enabled automatically.

### Testing

Test the feature toggle system:

```bash
# Run feature manager tests
uv run python test_feature_toggles.py

# Test processing with disabled features
uv run python test_processing_with_features.py
```

## Examples

### Minimal Processing (No Filtering)

```toml
[features]
kalman_filtering = false
quality_scoring = false
outlier_detection = false
quality_override = false

[features.state]
persistence = false
history_buffer = false
reset_tracking = false
```

### High-Trust Mode (Minimal Validation)

```toml
[features]
kalman_filtering = true
quality_scoring = false
outlier_detection = false

[features.validation]
physiological = true    # Keep safety limits
bmi_checking = false
rate_limiting = false
```

### Maximum Filtering (Strict Mode)

```toml
[features]
# All features enabled (default)
kalman_filtering = true
quality_scoring = true
outlier_detection = true
quality_override = false    # Don't allow overrides

[features.outlier_methods]
# All methods active
iqr = true
mad = true
temporal = true
kalman_deviation = true
```

### Diagnostic Mode (No State Persistence)

```toml
[features.state]
persistence = false    # Don't save states
history_buffer = false
reset_tracking = false

[features.resets]
initial = false       # No resets
hard = false
soft = false
```

## Backward Compatibility

- All features default to `true` for backward compatibility
- Missing `[features]` section = all features enabled
- Existing configurations continue to work unchanged

## Performance Considerations

- Disabling features reduces computational overhead
- Most significant performance gains from disabling:
  - `kalman_filtering` - Eliminates matrix operations
  - `quality_scoring` - Reduces multiple calculations
  - `state_persistence` - Eliminates database I/O

## Safety Notes

- All safety features can now be disabled (per user request)
- Consider the implications of disabling physiological validation
- Some features may produce unexpected results when core dependencies are disabled