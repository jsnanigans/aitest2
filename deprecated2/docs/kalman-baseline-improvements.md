# Kalman Filter & Baseline Visualization Improvements

## Improvements Made (September 2024)

### 1. Fixed "Unknown Source" Label for Predictions
**Problem**: Gap-filled predictions were showing as "Unknown Source" in the visualization.

**Solution**: Added proper source mapping for `kalman-extrapolation`:
```python
'kalman-extrapolation': {'marker': '.', 'color': '#FF9800', 'label': 'Gap-Filled (Kalman)'}
```

### 2. Added Baseline Establishment Visual Markers
**Problem**: Users couldn't see when the baseline was established on the chart.

**Solution**: Added visual indicators:
- **Vertical line** at signup date (baseline start)
- **Shaded region** for 7-day baseline establishment window
- **Enhanced label** showing baseline confidence level
- **Info box** with baseline quality metrics

### 3. Kalman Filter Timing After Baseline
**Problem**: Kalman filter was processing data before baseline establishment, which doesn't make sense.

**Solution**: Modified the processing flow:
- Kalman filter now **starts AFTER** the 7-day baseline window
- Only processes readings after baseline establishment
- Uses baseline weight and variance for initialization

## Visual Elements Now Shown

### On the Dashboard:
1. **Baseline Reference Line** - Purple dashed line at baseline weight
2. **±3% Confidence Band** - Shaded region around baseline
3. **Signup Marker** - Vertical dotted line showing when user signed up
4. **Baseline Window** - Shaded region showing data collection period
5. **Baseline Info Box** - Shows:
   - Baseline weight
   - Confidence level (high/medium/low)
   - Number of readings used
   - Outliers removed
   - Standard deviation

### Understanding the Timeline:
```
Timeline:
|--signup--|--7 days baseline--|--Kalman starts-->
   ↑             ↑                    ↑
   User      Baseline            Filter begins
   joins     established         with baseline
```

## Key Insights

### Why Baseline First, Then Kalman?
1. **Accurate Initialization**: Kalman needs a good starting point
2. **Proper Variance**: MAD from baseline provides realistic measurement noise
3. **Clean Data**: IQR removes outliers before filter initialization
4. **Logical Flow**: Establish ground truth, then track changes

### Current Implementation:
- **74% of users** have baselines established
- **Kalman only runs** for users with valid baselines
- **Predictions start** after baseline period ends

## Configuration

In `config.toml`:
```toml
# Baseline settings
baseline_min_readings = 3       # Min readings for baseline
baseline_window_days = 7        # Collection window
iqr_multiplier = 1.5            # Outlier detection threshold

# Kalman settings
enable_kalman = true            # Enable filter
enable_gap_filling = true       # Fill gaps with predictions
```

## Future Improvements

1. **Adaptive Baseline Window**: Extend if insufficient data
2. **Multiple Baselines**: Re-establish after major changes
3. **Confidence Decay**: Show prediction confidence decreasing over time
4. **Baseline Quality Score**: Visual indicator of baseline reliability

## Visual Example Interpretation

When viewing a dashboard:
- **Purple diamond** = Initial questionnaire (signup)
- **Purple shaded area** = Baseline collection period
- **Blue squares** = Actual measurements
- **Orange dots** = Gap-filled predictions
- **Orange dashed line** = Kalman filter trajectory
- **Yellow band** = Uncertainty envelope

The Kalman filter and predictions only appear AFTER the baseline period, ensuring they're initialized with accurate user-specific parameters rather than generic defaults.