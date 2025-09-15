# Configuration Migration Guide

## Overview
This guide helps you migrate from the old configuration format to the cleaned version (v2.0.0).

## What Changed

### Removed Sections (Completely Deleted)
The following sections were removed as they were not used in the code:

1. **[visualization.interactive]** - All settings removed:
   - `enable_webgl`
   - `max_points_before_decimation`
   - `default_time_range`

2. **[visualization.quality]** - All settings removed:
   - `show_quality_scores`
   - `quality_color_scheme`
   - `highlight_threshold_zone`
   - `threshold_buffer`
   - `show_component_details`

3. **[visualization.export]** - All settings removed:
   - `png_width`
   - `png_height`
   - `png_scale`

### Removed Individual Settings

#### From [processing]:
- `kalman_cleanup_threshold` - Was not used anywhere in code

#### From [visualization]:
- `mode` - Interactive/static distinction not implemented
- `theme` - Not used
- `dashboard_figsize` - Not used
- `moving_average_window` - Not used
- `cropped_months` - Not used

### Modified Settings

#### [visualization]:
- `verbosity` - Now properly connected to the logging system
  - Valid values: "silent", "minimal", "normal", "verbose"
  - Default: "normal"

#### [adaptive_noise]:
- Now properly connected to the processor
- `enabled` - Actually controls adaptive noise (was hardcoded before)
- `default_multiplier` - Used for unknown sources (was hardcoded as 1.5)

## Migration Steps

1. **Backup your current config.toml**
   ```bash
   cp config.toml config.toml.old
   ```

2. **Remove unused sections**
   Delete the entire sections listed above from your config file.

3. **Remove unused individual settings**
   Delete the specific settings listed above.

4. **Update verbosity setting** (if you use it)
   Ensure it's one of: "silent", "minimal", "normal", "verbose"

5. **Verify your config**
   The system now validates configuration on startup and will report any errors.

## Example Migration

### Before (Old Config):
```toml
[visualization]
enabled = true
mode = "interactive"  # NOT USED
use_enhanced = true
verbosity = "auto"  # NOT CONNECTED
theme = "plotly_white"  # NOT USED
dashboard_figsize = [16, 10]  # NOT USED
moving_average_window = 7  # NOT USED
cropped_months = 24  # NOT USED

[visualization.interactive]  # ENTIRE SECTION NOT USED
enable_webgl = true
max_points_before_decimation = 10000
default_time_range = "3M"
```

### After (Cleaned Config):
```toml
[visualization]
enabled = true
use_enhanced = true
verbosity = "normal"  # NOW CONNECTED TO SYSTEM
# All unused settings removed
```

## Validation

The system now validates your configuration on startup. If there are any issues, you'll see clear error messages like:

```
Configuration validation failed:
  - Missing required section: [kalman]
  - Invalid extreme_threshold: 1.5 (must be between 0 and 1)
  - Quality scoring weights must sum to 1.0, got 0.95
```

## Default Values

If you relied on any removed settings, here are the defaults that were being used:

- `kalman_cleanup_threshold`: 2.0 (from constants.py)
- `mode`: "interactive" (but had no effect)
- `theme`: "plotly_white" (but had no effect)
- Adaptive noise multipliers: Hardcoded per source type

## Need Help?

If you have custom configurations that need special attention:

1. Check if the setting was actually affecting behavior (most weren't)
2. Look for the setting in the new config with comments
3. Run with `--help` to see command-line overrides
4. The validation will catch any critical missing settings

## Version Information

- Old version: config.toml (86 lines, ~50 settings)
- New version: config.toml v2.0.0 (45 lines, ~25 settings)
- Date: 2025-09-15

All removed settings were verified to be unused in the codebase through comprehensive analysis.