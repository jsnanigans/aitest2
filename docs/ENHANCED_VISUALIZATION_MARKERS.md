# Enhanced Visualization with Reset Markers and Source Icons

## Summary
Successfully implemented enhanced visualization features to display reset events after data gaps and source type indicators using distinct marker icons, as specified in the plan.

## Implementation Details

### 1. Source Type Marker Mapping (constants.py)
- Added `SOURCE_MARKER_SYMBOLS` dictionary mapping each source to a Plotly symbol:
  - `care-team-upload`: triangle-up (professional/authoritative)
  - `patient-upload`: circle (standard/default)
  - `internal-questionnaire`: square (form/structured)
  - `patient-device`: diamond (automated/digital)
  - `https://connectivehealth.io`: hexagon (external API)
  - `https://api.iglucose.com`: hexagon (external API)

### 2. Rejection Severity Colors (constants.py)
- Added `REJECTION_SEVERITY_COLORS` for rejected points:
  - Critical: '#8B0000' (dark red) - for impossible values
  - High: '#CC0000' (medium-dark red) - for extreme deviations
  - Medium: '#FF4444' (medium red) - for suspicious values
  - Low: '#FF9999' (light red) - for minor issues

### 3. Reset Event Tracking (processor.py)
- Verified reset metadata is properly included in results:
  - `was_reset`: boolean flag
  - `reset_reason`: string description
  - `gap_days`: numeric gap duration
- Metadata persists through entire pipeline

### 4. Basic Visualization Updates (visualization.py)
- Extract reset events from results
- Map source types to marker symbols for both accepted and rejected points
- Accepted points: Use source-based markers with quality score color gradient
- Rejected points: Use source-based markers with severity-based red colors
- Reset events: Display as vertical dashed lines with gap duration labels
- Updated hover text to include reset context and source information

### 5. Enhanced Visualization Updates (viz_enhanced.py)
- Applied same marker and color scheme to enhanced visualization
- Grouped rejected points by severity level for clearer legend
- Added reset annotations to main plot
- Enhanced hover text with reset information
- Updated legend to explain shapes (sources) and colors (status/quality)

### 6. Configuration Options (config.toml)
Added new configuration sections:
```toml
[visualization.markers]
show_source_icons = true  # Use different marker shapes for each source type
show_source_legend = true  # Display legend for source types
show_reset_markers = true  # Display reset event annotations
reset_marker_color = "#FF6600"  # Orange color for reset markers
reset_marker_opacity = 0.2  # Transparency of reset lines (0-1)
reset_marker_width = 1  # Width of reset lines
reset_marker_style = "dot"  # "solid" | "dash" | "dot"

[visualization.rejection]
show_severity_colors = true  # Use color intensity to show rejection severity
group_by_severity = true  # Create separate legend entries for each severity level
```

## Visual Encoding Strategy
- **Shape**: Always indicates data source type (consistent for both accepted and rejected)
- **Color**: 
  - Accepted points: Green gradient (based on quality score)
  - Rejected points: Red shades (based on rejection severity)
- **Vertical Lines**: Reset events shown as subtle dotted vertical lines behind data
- **Annotations**: Minimal gap duration labels (e.g., "35d") at top of chart
- **Legend**: Dedicated section showing all source type shapes with labels

## Testing
Created comprehensive test scripts:
- `test_enhanced_markers.py`: Generates synthetic data with various sources and reset events
- `config_test_viz.toml`: Test configuration with all new features enabled
- Successfully tested with both synthetic and real data

## Benefits
1. **Improved Visibility**: Users can now see when and why Kalman filter resets occur via subtle vertical lines
2. **Source Differentiation**: Different marker shapes make it easy to identify data sources
3. **Clear Legend**: Dedicated source type legend explains all marker shapes
4. **Rejection Clarity**: Red color intensity shows severity of rejection reasons
5. **Consistency**: Same shape language for both accepted and rejected points
6. **Non-intrusive Reset Indicators**: Subtle dotted lines behind data show gaps without cluttering
7. **Accessibility**: Shape differentiation works for color-blind users
8. **Configurability**: All features can be toggled and styled via configuration

## Files Modified
- `src/constants.py`: Added marker symbols and severity colors
- `src/visualization.py`: Updated basic visualization with new features
- `src/viz_enhanced.py`: Updated enhanced visualization with new features
- `config.toml`: Added new configuration options
- Created test files for validation

## Usage Example
```python
from visualization import create_weight_timeline

config = {
    'visualization': {
        'markers': {
            'show_source_icons': True,
            'show_reset_markers': True,
            'reset_marker_color': '#FF6600'
        },
        'rejection': {
            'show_severity_colors': True,
            'group_by_severity': True
        }
    }
}

output_file = create_weight_timeline(
    results, 
    user_id="USER_001",
    config=config
)
```

## Next Steps
- Consider adding tooltips explaining what each marker shape means
- Could add option to show confidence level changes after reset
- Might add animation for reset events in future versions