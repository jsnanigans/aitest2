# Investigation: Dramatic Downward Trend in Kalman Trajectory

## Bottom Line
**Root Cause**: Outlier measurement (53.3kg on 2025-04-13) accepted during adaptive period
**Fix Location**: `src/processor.py:283-294` - Adaptive quality thresholds too lenient
**Confidence**: High

## What's Happening
User 0040872d-333a-4ace-8c5a-b2fcd056e65a shows 32kg drop (88.5kg→56.6kg) over 8 months. The trajectory becomes unstable after accepting a 53.3kg measurement that's 24kg below the previous value, causing the Kalman filter to overcorrect with trends up to -63kg/week.

## Why It Happens
**Primary Cause**: Adaptive period accepts bad measurements
**Trigger**: `src/processor.py:283` - Quality threshold drops to 0.35 during adaptation
**Decision Point**: `config.toml:33-55` - Initial reset allows 50x noise multiplier

### Chain of Events
1. **Hard reset** on 2025-03-03 after 49-day gap initiates adaptive period
2. **Adaptive parameters** multiply noise by 20x, lower quality threshold to 0.35
3. **April 13**: 53.3kg measurement accepted (24kg drop from 77kg)
4. **Kalman overcorrects**: Trend becomes -63kg/week to accommodate outlier
5. **Cascade failure**: All subsequent measurements show extreme trends

## Evidence
- **Key Measurement**: `output/results_test_no_date.json` - 2025-04-13: 53.3kg accepted
- **Extreme Trends**: Multiple -40 to -60kg/week trends after April 13
- **Config Issue**: `config.toml:64` - `weight_noise_multiplier = 20` (hard reset)
- **Threshold Drop**: `src/processor.py:284` - threshold becomes 0.35 vs normal 0.6

## Next Steps
1. **Reduce noise multipliers** in `config.toml`: 
   - `kalman.reset.hard.weight_noise_multiplier`: 20 → 5
   - `kalman.reset.hard.trend_noise_multiplier`: 200 → 50
2. **Tighten adaptive quality thresholds**:
   - `kalman.reset.hard.quality_acceptance_threshold`: 0.35 → 0.45
3. **Add outlier detection** before Kalman update to reject >15% deviations during adaptation
4. **Limit maximum trend** to ±5kg/week regardless of adaptation state

## Risks
- **Unfixed**: Kalman filter will continue accepting outliers, causing nonsensical trajectories
- **Over-correction**: Too tight parameters might reject legitimate weight changes after gaps
