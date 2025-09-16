# Investigation: Retrospective Data Quality Not Improving Kalman Smoothness

## Bottom Line
**Root Cause**: Retrospective processing is disabled (`retrospective.enabled = false` in config.toml), but even when enabled, quality scoring inconsistencies between immediate and replay processing cause acceptance rate drops without smoothness gains.
**Fix Location**: `config.toml:154` and `src/processor.py:284`
**Confidence**: High

## What's Happening
Retrospective processing was intended to improve Kalman filter smoothness by removing outliers from buffered measurements and replaying clean data. However, the system shows functional outlier detection (55 outliers from 200 measurements) but reduced acceptance rates (41.8% → 33.3%) without corresponding smoothness improvements. Current RMS innovation for user `0040872d-333a-4ace-8c5a-b2fcd056e65a` is 2.87.

## Why It Happens
**Primary Cause**: Quality scoring thresholds differ between immediate processing and replay
**Trigger**: `src/processor.py:284` - Adaptive quality thresholds during reset periods
**Decision Point**: `config.toml:284` - Different acceptance thresholds for reset states vs normal processing

**Evidence**:
- Immediate processing uses reset-specific thresholds (0.0-0.45 during adaptation)  
- Replay processing may not correctly restore reset context, causing normal 0.6 threshold
- Rejected measurements during replay show "Quality score X.XX below threshold 0.35/0.6"
- Consistency component failures dominate rejections (weakest component in most cases)

## Evidence
- **Key File**: `config.toml:154` - Retrospective processing disabled
- **Search Used**: `jq '.users["0040872d-333a-4ace-8c5a-b2fcd056e65a"][] | select(.accepted == false)'` - Found 17 rejected measurements with quality score issues
- **Pattern**: Rejected measurements cluster around reset events where thresholds should be lenient

## Next Steps
1. **Enable retrospective processing**: Set `retrospective.enabled = true` in config.toml
2. **Fix replay quality scoring**: Ensure replay uses same adaptive thresholds as immediate processing by preserving reset context during state restoration
3. **Tune outlier detection**: Reduce `temporal_max_change_percent` from 0.30 to 0.50 for less aggressive outlier removal
4. **Add smoothness metrics**: Calculate trajectory derivatives and innovation variance to measure actual smoothness improvements

## Risks
- Enabling retrospective processing may introduce processing delays and complexity
- Incorrect threshold alignment could further reduce acceptance rates
- Less aggressive outlier detection may allow harmful measurements through
