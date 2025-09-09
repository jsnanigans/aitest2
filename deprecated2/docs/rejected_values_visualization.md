# Rejected Values Visualization Enhancement

## Overview
Enhanced the "2025 Validated Data (Zoomed)" chart in the dashboard visualization to display rejected values similar to how they are shown in the "Weight Trajectory" chart.

## Changes Made

### 1. Data Collection (`src/visualization/dashboard.py`)
- Modified the zoomed chart section to collect rejected measurements from `all_readings`
- Added variables to track rejected data points specifically for the 2025 timeframe:
  - `zoom_rejected_dates` - dates of rejected readings
  - `zoom_rejected_weights` - weight values of rejected readings  
  - `zoom_rejected_sources` - data sources of rejected readings
  - `zoom_rejected_reasons` - rejection reasons

### 2. Visual Display
**Rejected Points Rendering:**
- Rejected points are plotted FIRST with lower z-order (20) to ensure they appear behind accepted points
- Visual style for rejected points:
  - Fill color: `#FF4444` (red)
  - Edge color: `darkred` 
  - Edge width: 2 pixels
  - Marker size: 80% of normal size
  - Marker shape: Inherits from data source type

**Accepted Points Rendering:**
- Accepted points plotted SECOND with higher z-order (50) to appear on top
- Color-coded by confidence level:
  - Green (`#4CAF50`): confidence ≥ 0.9
  - Light green (`#8BC34A`): confidence ≥ 0.75
  - Yellow (`#FFC107`): confidence ≥ 0.6
  - Orange (`#FF9800`): confidence < 0.6

### 3. Chart Adjustments
**Y-axis Range:**
- Updated to include BOTH accepted and rejected weights
- Increased padding to 15% to ensure all outlier points are visible
- Calculation: `y_padding = max(y_range * 0.15, 3)`

**Chart Title:**
- Dynamically updates based on presence of rejected data:
  - With rejections: "2025 Data (Zoomed) - X validated, Y rejected"
  - Without rejections: "2025 Validated Data (Zoomed) - X readings"

**Legend:**
- Added entry for rejected points when present
- Shows count of rejected points: "Rejected (n=Y)"
- Red marker with dark red edge to match chart display

### 4. Data Source Compatibility
The implementation supports two data structures:
1. **Primary**: Uses `all_readings` field with `is_rejected` flag
2. **Fallback**: Uses `kalman_time_series` with `measurement_accepted` field

## Usage

To enable rejected points visualization:

1. Set configuration flag:
```python
config = {
    "use_all_readings_for_viz": True
}
```

2. Ensure data includes `all_readings` field with rejection information:
```python
data = {
    "all_readings": [
        {
            "date": "2025-01-15T10:00:00",
            "weight": 95.0,
            "source": "patient-device",
            "is_rejected": True,
            "rejection_reason": "Outlier detected"
        },
        # ... more readings
    ],
    "time_series": [...],  # Accepted readings only
    # ... other fields
}
```

## Visual Result

The enhanced chart now clearly shows:
- ✅ **Accepted readings**: Color-coded by confidence level
- ❌ **Rejected readings**: Red markers with dark red outline
- 📊 **Complete picture**: All data points from 2025 onwards
- 📈 **Kalman filter**: Overlay showing smoothed trajectory
- 📋 **Clear statistics**: Title and legend show counts

## Testing

Three test scripts were created to verify the implementation:
1. `test_rejected_viz.py` - Basic test with sample data
2. `verify_rejected_display.py` - Focused test on 2025 data
3. `final_rejected_test.py` - Comprehensive test with realistic data patterns

All tests pass successfully, confirming that rejected values are now properly displayed in the zoomed chart.