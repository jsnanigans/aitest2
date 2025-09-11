# Understanding the Kalman Filter Processing Visualization

This guide explains how to read and interpret the weight processing visualization dashboard.

## Dashboard Overview

The visualization provides a comprehensive view of how the Kalman filter processes raw weight measurements, identifying patterns, outliers, and trends in your weight data.

## Main Chart: Kalman Filter Output vs Raw Data

The large chart at the top shows your weight measurements over time.

### Chart Elements

**Lines and Points:**
- **Blue Line** - Kalman filtered weight (smoothed, best estimate)
- **Accepted Measurements** - Different shapes indicate data source (see Source Types below)
- **Colored X Marks** - Rejected measurements (color indicates reason - see below)
- **Light Blue Shading** - Uncertainty band (wider = less confidence)
- **Orange Dashed Line** - Baseline weight (median of first 10 measurements)
- **Gray Dashed Vertical Lines** - State resets (with gap duration label)

### Source Type Indicators

Accepted measurements use different marker shapes to indicate their data source:

- **○ Circle (Device)** - Direct device measurements (scales, connected devices)
  - Full name: `patient-device`, `scale`, or similar device identifiers
  - Most reliable, automatic measurements
  
- **□ Square (Connected)** - API/connected health platforms
  - Full name: `https://connectivehealth.io`, `https://api.iglucose.com`, or other API URLs
  - Data from integrated health platforms
  
- **△ Triangle (Questionnaire)** - Self-reported via questionnaires
  - Full name: `internal-questionnaire`
  - Manual entry through structured forms
  
- **◇ Diamond (Manual)** - Patient uploads
  - Full name: `patient-upload`
  - Manual data entry by patients
  
- **⬟ Pentagon (Test)** - Test data
  - Full name: `test`
  - Data used for testing/validation
  
- **• Dot (Other/Unknown)** - Unrecognized sources
  - Full name: Various unrecognized source identifiers
  - Sources that don't match known patterns

**Visual Hierarchy:**
- Larger, more opaque markers = more reliable sources
- Smaller, more transparent markers = less reliable sources
- Color coding: Green (Device) → Blue (Connected) → Purple (Questionnaire) → Orange (Manual) → Gray (Other)

### Rejection Color Coding

Rejected measurements are marked with colored X's, each color representing a different rejection reason:

- 🔴 **Red (Extreme)** - Extreme deviations from expected weight
- 🩷 **Pink (Bounds)** - Weight outside acceptable range (30-400 kg)
- 🟠 **Dark Orange (Sustained)** - Sustained changes requiring confirmation
- 🟡 **Orange (Variance)** - Session variance or possible different user
- 🟨 **Yellow (Daily)** - Excessive daily fluctuation
- 🟢 **Green (Medium)** - Medium-term changes (meals + hydration)
- 🔵 **Blue (Short)** - Short-term changes (bathroom/hydration only)
- 🟣 **Purple (Limit)** - Exceeds physiological limits
- ⚫ **Gray (Other)** - Other/uncategorized rejections

### Annotations

Small labels appear near some rejected measurements:
- **Single word** (e.g., "Extreme") - Category of rejection
- **Number + x** (e.g., "5x") - Multiple rejections clustered together

## Middle Row Charts

### 1. Kalman Innovation (Measurement - Prediction)

Shows the difference between what was measured and what the filter predicted.

**How to read:**
- **Near 0** - Measurements match predictions well
- **Positive spikes** - Weight higher than expected
- **Negative spikes** - Weight lower than expected
- **Blue shading** - Magnitude of innovation

**What it means:**
- Consistent small innovations = stable, predictable weight
- Large innovations = unexpected changes or measurement errors

### 2. Normalized Innovation (Statistical Significance)

Shows how significant each innovation is statistically.

**How to read:**
- **Green dots (≤2σ)** - Normal variation, good measurements
- **Orange dots (2-3σ)** - Moderate deviation, worth noting
- **Red dots (>3σ)** - Significant deviation, possibly erroneous

**Reference lines:**
- 2σ line - Good threshold
- 3σ line - Warning threshold
- 5σ line - Extreme threshold

**What it means:**
- Most dots should be below 2σ for a well-functioning filter
- Frequent high values suggest noisy measurements or rapid weight changes

### 3. Measurement Confidence

Shows the filter's confidence in each measurement.

**How to read:**
- **1.0** - Full confidence
- **0.5** - Moderate confidence
- **0.0** - No confidence

**Reference lines:**
- High (>0.95) - Very confident
- Medium (>0.7) - Reasonably confident
- Low (<0.5) - Low confidence

**What it means:**
- High confidence = measurement aligns with expectations
- Low confidence = measurement seems questionable
- Drops in confidence often precede rejected measurements

### 4. Daily Change Distribution

Histogram showing the distribution of day-to-day weight changes.

**How to read:**
- **Center at 0** - Weight stable overall
- **Shifted right** - Gaining weight
- **Shifted left** - Losing weight
- **Wide spread** - High variability
- **Narrow spread** - Consistent weight

**What it means:**
- Normal: Centered near 0 with most changes within ±0.5 kg/day
- Concerning: Consistently >1 kg/day changes or very wide distribution

## Bottom Section

### Rejection Categories (Bar Chart - Left)

Horizontal bars showing the frequency of each rejection type.

**How to read:**
- **Longer bars** - More common rejection reason
- **Colors** - Match the rejection markers in main chart
- **Percentages** - Proportion of total rejections

**What to look for:**
- Dominant rejection types indicate measurement patterns
- High "Extreme" or "Bounds" may indicate scale issues
- High "Variance" may indicate multiple users on same scale

### Measurements by Source Type (Bar Chart - Right)

Stacked bar chart showing the distribution of measurements across different data sources.

**How to read:**
- **Green bars** - Accepted measurements from this source
- **Red bars** - Rejected measurements from this source
- **Height** - Total number of measurements
- **Labels** - Show total count and acceptance percentage

**Source reliability ranking (most to least reliable):**
1. **Device** - Direct scale/device measurements
2. **Connected** - API-integrated platforms
3. **Questionnaire** - Structured self-reports
4. **Manual** - Patient uploads
5. **Test** - Test data
6. **Other** - Unrecognized sources

**What to look for:**
- Higher acceptance rates indicate more reliable sources
- Device sources typically have highest acceptance rates
- Manual/questionnaire sources may have more variability
- Consider source distribution when evaluating data quality

## Statistics Panel (Right Side)

### Processing Overview
- **Total Measurements** - Number of weight readings processed
- **Accepted** - Measurements that passed validation
- **Rejected** - Measurements that failed validation
- **Percentage** - Quality indicator (>90% accepted is good)

### Current Status
- **Weight** - Latest filtered weight estimate
- **Baseline** - Initial weight reference point
- **Change** - Total change from baseline
- **Trend** - Current rate of weight change per week

### Filter Performance
- **Mean Innovation** - Average prediction error (closer to 0 is better)
- **Std Innovation** - Consistency of predictions (lower is better)
- **Mean Norm. Innov** - Should be around 1.0σ for optimal performance
- **Max Norm. Innov** - Largest deviation seen
- **Mean Confidence** - Overall confidence level (higher is better)

### Weight Statistics
- **Mean** - Average filtered weight
- **Std Dev** - Weight variability
- **Min/Max** - Weight range observed
- **Range** - Total weight variation

### Source Analysis
Shows distribution of measurements by source type:
- **Source labels** - Shortened names (Device, Connected, etc.)
- **Counts** - Total measurements from each source
- **Acceptance rate** - Percentage accepted (acc %)

### Rejection Analysis
Lists the most common rejection reasons with counts and percentages.

## Interpreting Common Patterns

### Healthy Pattern
- Smooth blue line with minimal fluctuations
- Most measurements accepted (various shapes indicating sources)
- Consistent source types (ideally mostly Device or Connected)
- Few rejections, mostly "Short" or "Daily" categories
- Innovation centered around 0
- Confidence consistently high
- Daily changes centered at 0 with tight distribution

### Noisy Scale Pattern
- Many rejections, especially "Extreme" and "Variance"
- Wide uncertainty band
- Erratic innovations
- Low average confidence
- Wide daily change distribution
- Consider scale calibration or replacement

### Multiple Users Pattern
- Clusters of "Variance" rejections
- Bimodal patterns in daily change distribution
- Sudden jumps in filtered weight
- State resets may be frequent
- Consider separate user profiles

### Weight Loss/Gain Pattern
- Consistent downward/upward trend in blue line
- Daily change distribution shifted left/right
- Trend value significantly negative/positive
- Innovations may show consistent bias

### Mixed Source Pattern
- Multiple marker shapes in accepted measurements
- Varying acceptance rates by source in statistics
- Possible clustering of rejections near manual/questionnaire entries
- Consider focusing on most reliable sources (Device/Connected)
- May indicate need for better measurement standardization

### Data Gap Pattern
- Gray vertical lines showing state resets
- Large gaps between measurements
- Uncertainty increases after gaps
- Filter re-learns after long absences

## Best Practices for Reading

1. **Start with the main chart** - Get overall picture of your weight journey
2. **Check rejection colors** - Identify dominant issues
3. **Review statistics panel** - Understand data quality
4. **Examine middle charts** - Dive into filter performance
5. **Look for patterns** - Consistent issues indicate systemic problems

## Troubleshooting Guide

### High Rejection Rate (>20%)
- Check scale placement (stable, level surface)
- Ensure consistent measurement conditions
- Verify scale calibration
- Consider environmental factors

### Frequent State Resets
- Measure more consistently
- Avoid gaps longer than 30 days
- Maintain regular measurement schedule

### Low Confidence Scores
- Check for consistent measurement time
- Ensure proper scale usage
- Look for environmental interference
- Consider scale battery/maintenance

### Wide Uncertainty Bands
- Increase measurement frequency
- Improve measurement consistency
- Check for scale issues
- Reduce measurement condition variations

### Inconsistent Source Types
- Standardize measurement method
- Prefer Device or Connected sources when possible
- Ensure manual entries follow consistent protocol
- Consider source reliability in data interpretation

## Key Takeaways

1. **Source matters** - Device measurements (circles) are most reliable
2. **Smooth is accurate** - The blue line should be relatively smooth
3. **Patterns matter** - Look for consistent rejection types
4. **Confidence counts** - Higher confidence means better estimates
5. **Consistency helps** - Regular measurements improve accuracy
6. **Check your sources** - High acceptance rates indicate reliable data sources

## Technical Notes

- The Kalman filter adapts to your personal weight patterns
- State resets occur after 30+ day gaps
- Physiological limits prevent impossible weight changes
- The filter learns your typical variation over time
- Uncertainty increases with time between measurements
- Source reliability affects measurement confidence
- Device sources receive higher initial confidence than manual entries

---

*For more technical details about the Kalman filter implementation, see the technical documentation.*