# Source Visibility Implementation

## Overview
Implemented enhanced source visibility in the weight stream processor visualization to treat each unique source string as a distinct entity, providing better insight into data origins for both accepted and rejected measurements.

## Key Changes

### 1. Source Management System
- **Unique Source Extraction**: Each source string is treated as unique (no grouping into categories)
- **Source Registry**: Maps each unique source to visual properties (marker, color, size)
- **Frequency-Based Priority**: Sources are prioritized by frequency for visual hierarchy

### 2. Visual Encoding Strategy

#### Accepted Measurements
- **Primary Sources** (top 10-15 by frequency):
  - Unique marker shapes (circle, square, triangle, diamond, etc.)
  - Distinct colors from a carefully chosen palette
  - Size variation based on frequency
  - Individual legend entries with counts

- **Secondary Sources** (lower frequency):
  - Grouped as "Other Sources" to prevent clutter
  - Smaller dot markers
  - Muted gray color
  - Single legend entry for all

#### Rejected Measurements
- **Colored Outline System**:
  - Source marker shape preserved (shows data origin)
  - Colored outline indicates rejection reason
  - Outline color maps to specific rejection category
  - Both source and rejection status clearly visible

### 3. Implementation Details

#### New Functions in `visualization.py`
```python
get_unique_sources(results)          # Extract unique sources with counts
truncate_source_label(source)        # Smart label truncation
generate_marker_sequence()           # 15 distinct marker shapes
generate_color_palette(n)            # Visually distinct colors
create_source_registry(results)      # Map sources to visual properties
```

#### Visual Properties
- **Markers**: 15 distinct shapes (o, s, ^, v, D, p, h, *, P, X, d, <, >, 8, H)
- **Colors**: 15 base colors with brightness variations for overflow
- **Sizes**: 40-80 pixels based on frequency and importance
- **Outlines**: 
  - Green (#4CAF50) for accepted values
  - Category-specific colors for rejected values
- **Alpha**: 0.85 for accepted, 0.6 for rejected

#### Rejection Categories & Colors
- **BMI Value** (#FF1744): BMI value detected instead of weight
- **Unit Convert** (#FF6F00): Pounds/unit conversion detected
- **Physio Limit** (#E91E63): Outside physiological bounds
- **Extreme Dev** (#D32F2F): Extreme deviation from baseline
- **Out of Bounds** (#9C27B0): Outside normal bounds
- **High Variance** (#FFA726): Session variance too high
- **Daily Flux** (#FFC107): Daily fluctuation exceeded
- **Other** (#757575): Uncategorized rejections

### 4. Legend Optimization
- **Two-Tier System**:
  - Primary sources listed individually with counts
  - Secondary sources grouped as "Other"
  - Rejection categories shown separately
  
- **Smart Sorting**:
  - Sources sorted by frequency
  - Most common sources appear first
  - Counts shown in parentheses

### 5. Statistics Panel Updates
- **Source Analysis Section**:
  - Top 5 sources with acceptance rates
  - Total unique source count
  - Per-source statistics
  
- **Source Distribution Chart**:
  - Stacked bars showing accepted/rejected per source
  - Up to 8 individual sources + "Other"
  - Percentage labels for clarity

## Benefits

### Improved Insights
- **Data Quality**: Immediately see which sources provide reliable data
- **Source Patterns**: Identify problematic sources by rejection rates
- **Traceability**: Every measurement traceable to its exact origin
- **Acceptance Rates**: Per-source acceptance statistics

### Visual Clarity
- **No Clutter**: Smart grouping prevents overwhelming charts
- **Clear Hierarchy**: Most important sources visually prominent
- **Dual Information**: Source and status shown simultaneously
- **Readable Labels**: Smart truncation keeps labels manageable

### Scalability
- **Handles 50+ Sources**: Graceful degradation with many sources
- **Performance**: Minimal impact on rendering time
- **Future-Proof**: Ready for interactive enhancements

## Usage Example

```python
from src.visualization import create_dashboard

# Results with unique source strings
results = [
    {
        'source': 'patient-device-scale-ABC123',
        'timestamp': datetime.now(),
        'raw_weight': 70.5,
        'accepted': True,
        # ... other fields
    },
    {
        'source': 'https://api.iglucose.com/v1/weights',
        'timestamp': datetime.now(),
        'raw_weight': 71.0,
        'accepted': False,
        'reason': 'Outside physiological bounds',
        # ... other fields
    }
]

# Create visualization with unique sources
create_dashboard(user_id, results, output_dir, viz_config)
```

## Testing
- `test_unique_source_visualization.py`: Unit tests for all new functions
- `test_source_integration.py`: Integration test with processor
- Verified with 10-50 unique sources
- Performance tested with 100+ sources

## Migration Notes
- **Backwards Compatible**: Existing code continues to work
- **No Data Changes**: Source field unchanged in processor
- **Visual Only**: Changes limited to visualization layer
- **Config Optional**: No new configuration required

## Future Enhancements
1. **Interactive Filtering**: Click sources to show/hide
2. **Source Grouping**: User-defined source categories
3. **Historical Trends**: Source reliability over time
4. **Export Options**: Source-specific data export
5. **Custom Icons**: User-uploadable source icons