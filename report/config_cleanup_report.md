# Configuration Cleanup Report

## Executive Summary

After analyzing the codebase, I found that **many configuration options in `config.toml` are NOT being used**. The configuration file contains 86 lines with approximately 50+ settings, but only about **30-40% are actively used** in the code.

## Configuration Usage Analysis

### ✅ ACTIVELY USED Settings

#### [data] Section
- ✅ `csv_file` - Used in main.py for input file
- ✅ `output_dir` - Used in main.py for output directory
- ✅ `max_users` - Used in main.py for limiting users
- ✅ `user_offset` - Used in main.py for user offset
- ✅ `min_readings` - Used in main.py for filtering
- ✅ `min_date` - Used in main.py for date filtering
- ✅ `max_date` - Used in main.py for date filtering
- ✅ `export_database` - Used in main.py for database export

#### [processing] Section
- ✅ `extreme_threshold` - Used in processor.py for deviation checks
- ❌ `kalman_cleanup_threshold` - **NOT USED** (only in constants.py defaults)

#### [kalman] Section
- ✅ `initial_variance` - Used in kalman.py
- ✅ `transition_covariance_weight` - Used in kalman.py
- ✅ `transition_covariance_trend` - Used in kalman.py
- ✅ `observation_covariance` - Used in kalman.py
- ✅ `reset_gap_days` - Used in processor.py
- ✅ `questionnaire_reset_days` - Used in processor.py

#### [visualization] Section
- ✅ `enabled` - Used in main.py to control visualization
- ❌ `mode` - **NOT USED** (interactive/static distinction not implemented)
- ✅ `use_enhanced` - Used in visualization.py
- ❌ `verbosity` - **NOT USED** (verbosity system exists but doesn't read config)
- ❌ `theme` - **NOT USED**
- ❌ `dashboard_figsize` - **NOT USED**
- ❌ `moving_average_window` - **NOT USED**
- ❌ `cropped_months` - **NOT USED**

#### [visualization.interactive] Section
- ❌ `enable_webgl` - **NOT USED**
- ❌ `max_points_before_decimation` - **NOT USED**
- ❌ `default_time_range` - **NOT USED**

#### [visualization.quality] Section
- ❌ `show_quality_scores` - **NOT USED**
- ❌ `quality_color_scheme` - **NOT USED**
- ❌ `highlight_threshold_zone` - **NOT USED**
- ❌ `threshold_buffer` - **NOT USED**
- ❌ `show_component_details` - **NOT USED**

#### [visualization.export] Section
- ❌ `png_width` - **NOT USED**
- ❌ `png_height` - **NOT USED**
- ❌ `png_scale` - **NOT USED**

#### [visualization.markers] Section
- ✅ `show_source_icons` - Used in visualization.py
- ✅ `show_source_legend` - Used in visualization.py
- ✅ `show_reset_markers` - Used in visualization.py
- ✅ `reset_marker_color` - Used in visualization.py
- ✅ `reset_marker_opacity` - Used in visualization.py
- ✅ `reset_marker_width` - Used in visualization.py
- ✅ `reset_marker_style` - Used in visualization.py

#### [visualization.rejection] Section
- ✅ `show_severity_colors` - Used in visualization.py
- ✅ `group_by_severity` - Used in visualization.py

#### [adaptive_noise] Section
- ❌ `enabled` - **NOT USED** (adaptive noise is hardcoded)
- ❌ `default_multiplier` - **NOT USED**

#### [logging] Section
- ✅ `progress_interval` - Used in main.py
- ✅ `timestamp_format` - Used in main.py

#### [quality_scoring] Section
- ✅ `enabled` - Used in processor.py
- ✅ `threshold` - Used in quality_scorer.py
- ✅ `use_harmonic_mean` - Used in quality_scorer.py

#### [quality_scoring.component_weights] Section
- ✅ `safety` - Used in quality_scorer.py
- ✅ `plausibility` - Used in quality_scorer.py
- ✅ `consistency` - Used in quality_scorer.py
- ✅ `reliability` - Used in quality_scorer.py

## 🔴 UNUSED Configuration Sections

The following entire sections or settings are completely unused:

1. **[visualization.interactive]** - Entire section unused
2. **[visualization.quality]** - Entire section unused
3. **[visualization.export]** - Entire section unused
4. **[adaptive_noise]** - Entire section unused
5. **Many [visualization] settings** - mode, verbosity, theme, dashboard_figsize, moving_average_window, cropped_months

## 📊 Statistics

- **Total settings**: ~50+
- **Used settings**: ~25 (50%)
- **Unused settings**: ~25 (50%)
- **Entire unused sections**: 4

## 🎯 Recommendations

### 1. Remove Unused Sections
Delete these entire sections from config.toml:
```toml
# DELETE THESE SECTIONS:
[visualization.interactive]
[visualization.quality]
[visualization.export]
[adaptive_noise]
```

### 2. Clean Up Visualization Section
Remove unused settings:
```toml
[visualization]
enabled = true
use_enhanced = true
# DELETE: mode, verbosity, theme, dashboard_figsize, moving_average_window, cropped_months
```

### 3. Remove Unused Processing Settings
```toml
[processing]
extreme_threshold = 0.20
# DELETE: kalman_cleanup_threshold
```

### 4. Consider Moving Hardcoded Values
Some values are hardcoded in constants.py but have config entries:
- `kalman_cleanup_threshold` - only used from constants.py
- Adaptive noise multipliers - hardcoded in processor.py

### 5. Proposed Clean Config Structure

```toml
# Weight Stream Processor Configuration - CLEANED VERSION

[data]
csv_file = "./data/2025-09-05_optimized.csv"
output_dir = "output"
max_users = 100
user_offset = 0
min_readings = 2
min_date = "2015-01-01"
max_date = "2026-01-01"
export_database = true

[processing]
extreme_threshold = 0.20

[kalman]
initial_variance = 0.361
transition_covariance_weight = 0.0160
transition_covariance_trend = 0.0001
observation_covariance = 3.490
reset_gap_days = 30
questionnaire_reset_days = 10

[visualization]
enabled = true
use_enhanced = true

[visualization.markers]
show_source_icons = true
show_source_legend = true
show_reset_markers = true
reset_marker_color = "#FF6600"
reset_marker_opacity = 0.2
reset_marker_width = 1
reset_marker_style = "dot"

[visualization.rejection]
show_severity_colors = true
group_by_severity = true

[logging]
progress_interval = 10000
timestamp_format = "test_no_date"

[quality_scoring]
enabled = true
threshold = 0.61
use_harmonic_mean = true

[quality_scoring.component_weights]
safety = 0.35
plausibility = 0.25
consistency = 0.25
reliability = 0.15
```

## 🚨 Critical Findings

1. **Adaptive Noise Configuration Not Used**: The `[adaptive_noise]` section exists but the feature is hardcoded in processor.py
2. **Verbosity System Disconnected**: A verbosity system exists in utils.py but doesn't read from config
3. **Many Visualization Features Configured but Not Implemented**: Interactive settings, quality display settings, and export settings are all configured but never used
4. **Mode Setting Ignored**: The visualization mode (interactive/static) is configured but not actually used

## 💡 Implementation Priority

1. **High Priority**: Remove unused sections to avoid confusion
2. **Medium Priority**: Either implement or remove the verbosity config connection
3. **Low Priority**: Consider if unused visualization features should be implemented or permanently removed

## Code References

Key files where config is used:
- `main.py`: Lines 58-75, 160, 227, 274, 286, 396-411
- `processor.py`: Lines 98-99, 110, 146, 175-177, 270, 291
- `kalman.py`: Lines 27-37
- `visualization.py`: Lines 56-68
- `quality_scorer.py`: Lines 84-86

## Conclusion

The configuration file has accumulated many unused settings over time. A cleanup would:
1. Reduce confusion for developers
2. Make it clear what can actually be configured
3. Reduce maintenance burden
4. Improve code clarity

The proposed cleaned version above removes ~50% of the configuration while maintaining all actually-used functionality.
