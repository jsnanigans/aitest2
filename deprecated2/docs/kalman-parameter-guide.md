# Kalman Filter Parameter Guide

## Dynamic Source-Based Parameters

The Kalman filter now dynamically adjusts its parameters based on the reliability of the data source. This ensures that measurements from trusted sources (like healthcare professionals) have more influence than potentially unreliable sources.

### Source Trust Configuration

| Source Type | Trust Level | Noise Scale | Description |
|-------------|-------------|-------------|-------------|
| `care-team-upload` | 0.95 | 0.3× | Healthcare professional measurements (most trusted) |
| `internal-questionnaire` | 0.80 | 0.5× | Patient self-reports via official forms |
| `patient-upload` | 0.60 | 0.8× | Patient manual home scale entries |
| `unknown` | 0.50 | 1.0× | Default for unrecognized sources |
| `https://connectivehealth.io` | 0.30 | 1.5× | Third-party API with moderate reliability |
| `patient-device` | 0.70 | 0.8× | Automated patient devices (good reliability based on data) |
| `https://api.iglucose.com` | 0.10 | 3.0× | Known problematic API with frequent errors |

### How Dynamic Parameters Work

1. **Measurement Noise Scaling**: `measurement_noise = base_noise × noise_scale`
   - High trust sources → Lower noise → More weight given to measurement
   - Low trust sources → Higher noise → Less weight given to measurement

2. **Process Noise Adaptation**: 
   - `q_weight = base × (1 + (trust - 0.5) × 0.3)`
   - `q_trend = base × (1 + (trust - 0.5) × 0.5)`
   - High trust → Faster adaptation to legitimate changes
   - Low trust → More conservative, maintains previous estimates

3. **Outlier Rejection**: All sources undergo outlier detection, but:
   - High trust sources need larger deviations to be rejected
   - Low trust sources are rejected more readily

## Parameter Definitions and Tuning Guide

### 1. initial_weight (default: 100.0)
**What it does:** Starting weight estimate for the filter
**When to change:** Set to approximate expected weight for your population
**Connection:** Gets immediately overridden by first measurement, so exact value doesn't matter
**Range:** Any positive value (e.g., 70.0 for adults, 20.0 for children)

### 2. initial_trend (default: 0.0) 
**What it does:** Starting weight change rate in kg/day
**When to change:** Keep at 0.0 unless you know the population is systematically gaining/losing
**Connection:** Gets updated quickly based on actual measurements
**Range:** -0.05 to 0.05 kg/day (negative = losing, positive = gaining)

### 3. process_noise_weight (default: 0.4)
**What it does:** Expected day-to-day weight variance in kg²
**When to change:** 
  - Increase (0.8-1.2) for populations with volatile weight (athletes, water retention issues)
  - Decrease (0.3-0.5) for stable populations or when you want smoother predictions
**Connection:** Higher values make filter trust new measurements more, lower values create smoother curves
**Physical meaning:** √0.6 ≈ 0.77 kg expected daily weight fluctuation

### 4. max_reasonable_trend (default: 0.05)
**What it does:** Maximum believable weight change rate in kg/day
**When to change:**
  - Medical monitoring: 0.05-0.08 (0.35-0.56 kg/week max)
  - Athletic training: 0.10-0.15 (0.7-1.05 kg/week max)  
  - Post-surgery/illness: 0.15-0.20 (1.05-1.4 kg/week max)
**Connection:** Acts as a hard limiter - measurements implying faster changes get heavily discounted
**Physical meaning:** 0.08 kg/day = 0.56 kg/week = 2.4 kg/month maximum rate

### 5. process_noise_trend (default: 0.03)
**What it does:** Expected variation in the weight change rate itself
**When to change:**
  - Increase (0.08-0.15) for erratic patterns (yo-yo dieters, variable training)
  - Decrease (0.01-0.03) for consistent long-term trends
**Connection:** Controls how quickly the filter adapts to trend changes
**Physical meaning:** How much the rate of weight change can vary day-to-day

### 6. measurement_noise (default: 1.0)
**What it does:** Expected measurement error variance in kg²
**When to change:**
  - High-quality scales: 0.2-0.4 (±0.45-0.63 kg error)
  - Consumer scales: 0.6-1.0 (±0.77-1.0 kg error)
  - Self-reported/estimated: 1.5-3.0 (±1.2-1.7 kg error)
**Connection:** Higher values make filter trust measurements less, relying more on predictions
**Physical meaning:** √0.6 ≈ 0.77 kg measurement uncertainty

## Parameter Relationships

### Trust Balance: process_noise_weight vs measurement_noise
- **Ratio > 1** (e.g., process=0.8, measurement=0.4): Trusts measurements more than model
- **Ratio ≈ 1** (e.g., process=0.6, measurement=0.6): Balanced trust
- **Ratio < 1** (e.g., process=0.3, measurement=0.9): Trusts model more than measurements

### Responsiveness: process_noise_trend controls adaptation speed
- Works with max_reasonable_trend as upper bound
- Higher process_noise_trend → Faster trend adaptation
- Lower process_noise_trend → More stable, slower-changing trends

### Outlier Handling: max_reasonable_trend as gatekeeper
- Measurements implying changes > max_reasonable_trend get measurement_noise scaled up
- Prevents single bad measurements from corrupting the state
- Works with normalized innovation threshold (hardcoded at 3σ and 6σ)

## Time-Adaptive Behavior

The filter automatically adjusts parameters based on time gaps:

### Small gaps (< 0.1 days):
- Linear scaling of process noise
- Minimal trend flexibility

### Normal gaps (0.1-1.0 days):
- Standard process noise for weight
- Square-root scaling for trend (allows some flexibility)

### Medium gaps (1-7 days):
- Increased weight uncertainty (factor 1.0-1.5x)
- Increased trend flexibility (factor 1.0-3.0x)

### Large gaps (> 7 days):
- Square-root time scaling for weight
- High trend flexibility (up to 5x base value)
- Allows major trend adjustments after long absences

## Tuning Strategies

### For Stable Weight Monitoring:
```python
CustomKalmanFilter(
    process_noise_weight=0.4,      # Lower daily variance
    max_reasonable_trend=0.05,     # Conservative max rate
    process_noise_trend=0.02,      # Slow trend changes
    measurement_noise=0.8           # Don't overreact to individual readings
)
```

### For Active Weight Loss/Gain Programs:
```python
CustomKalmanFilter(
    process_noise_weight=0.8,      # Higher daily variance expected
    max_reasonable_trend=0.10,     # Allow faster changes
    process_noise_trend=0.08,      # Quick trend adaptation
    measurement_noise=0.6           # Trust measurements moderately
)
```

### For Noisy/Unreliable Data:
```python
CustomKalmanFilter(
    process_noise_weight=0.6,      # Standard variance
    max_reasonable_trend=0.08,     # Standard max rate
    process_noise_trend=0.03,      # Conservative trend changes
    measurement_noise=1.5           # High measurement uncertainty
)
```

## Diagnostic Signals

### Filter performing well:
- normalized_innovation mostly < 3.0
- outlier_rate < 10%
- trend changes direction < 5 times per month
- uncertainty_weight stabilizes after 5-10 measurements

### Need parameter adjustment:
- High outlier_rate (>15%): Increase measurement_noise or decrease max_reasonable_trend
- Oversmoothing: Decrease measurement_noise or increase process_noise_weight
- Erratic trends: Decrease process_noise_trend
- Missing real changes: Increase process_noise_trend and max_reasonable_trend