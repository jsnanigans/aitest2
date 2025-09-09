# Gap Filling Implementation with Kalman Predictions

## Overview
Implemented automatic gap filling in time series data using Kalman filter predictions. When there are gaps larger than a configurable threshold (default: 3 days), the system now automatically inserts predicted data points to create smoother, more continuous visualizations of the Kalman filter output.

## Key Components

### 1. Prediction Utilities (`src/utils/prediction_utils.py`)
- **`fill_time_series_gaps()`**: Main function that identifies gaps and fills them with predictions
- **`create_predicted_point()`**: Creates individual predicted data points using Kalman state
- **`interpolate_kalman_predictions()`**: Helper for visualization interpolation

### 2. Main Processor Updates (`main.py`)
- Integrated gap filling into the `finish_user()` method
- Predictions are added AFTER calculating percentiles (to preserve statistics on actual data)
- Moving averages are recalculated to include predicted points
- Tracks prediction statistics in output

### 3. Visualization Updates (`src/visualization/dashboard.py`)
- Predicted points shown with distinct 'x' markers in red
- Kalman filter line shows dashed segments for predicted regions
- Solid line for actual measurements, dashed for predictions
- Legend indicates count of predicted points

## Configuration

Added to `config.toml`:
```toml
# Prediction settings for filling gaps
prediction_max_gap_days = 3  # Fill gaps larger than this with predictions
enable_gap_filling = true     # Enable/disable gap filling with predictions
```

## Output Format

Each predicted point in the time series includes:
```json
{
  "date": "2025-02-16T08:09:43",
  "weight": 95.21,
  "confidence": 0.6,
  "source": "kalman-prediction",
  "is_predicted": true,
  "prediction_gap_days": 3,
  "kalman_filtered": 95.21,
  "kalman_uncertainty": 1.58
}
```

## Algorithm Details

1. **Gap Detection**: Scans time series for gaps > `prediction_max_gap_days`
2. **Prediction Generation**: 
   - Uses Kalman filter state (weight + trend) for prediction
   - Blends Kalman prediction with linear interpolation for stability
   - Uncertainty grows with √(days) from last measurement
3. **Confidence Scoring**: Predicted points have reduced confidence (0.3-0.9 based on gap size)

## Visual Representation

In the dashboard visualizations:
- **Actual measurements**: Solid markers with source-specific colors/shapes
- **Predicted points**: Red 'x' markers with reduced opacity
- **Kalman line**: Solid green for actual segments, dashed green for predicted
- **Uncertainty bands**: Lighter shading for predicted regions

## Benefits

1. **Improved Visualization**: Continuous Kalman filter lines show trends more clearly
2. **Gap Awareness**: Predictions are clearly marked as such
3. **Trend Preservation**: Kalman filter trend is maintained through gaps
4. **Configurable**: Can be disabled or adjusted via configuration

## Usage

The feature is enabled by default. To disable:
```toml
enable_gap_filling = false
```

To adjust the gap threshold:
```toml
prediction_max_gap_days = 7  # Only fill gaps larger than 7 days
```

## Statistics

The output JSON includes:
- `predicted_points_added`: Count of predicted points added
- `prediction_gap_days`: The gap threshold used
- Each predicted point is marked with `"is_predicted": true`