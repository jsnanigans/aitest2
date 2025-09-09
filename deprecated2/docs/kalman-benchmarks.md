# Kalman Filter Benchmark Documentation

## Overview

This document describes the benchmark suite for evaluating Kalman filter performance on real-world weight measurement data with extreme outliers. The benchmarks test both standard and rate-based Kalman filter implementations against challenging user data patterns.

## Benchmark Script

**File**: `benchmark_kalman.py`  
**Purpose**: Automated testing of Kalman filter implementations against known problematic cases

### Running the Benchmark

```bash
uv run python benchmark_kalman.py
```

Output files are saved to `output/benchmark_results_YYYYMMDD_HHMMSS.json`

## Test Cases

### User 01C39CB085ED4D349B3324D30766C156
**Problem**: Erroneous 61kg readings for an ~86kg person

#### Data Pattern
- **True baseline**: ~86.5 kg
- **Outliers**: 60.9 kg, 61.0 kg readings (30% below baseline)
- **Challenge**: Filter incorrectly accepts these physiologically impossible values

#### Test Metrics
| Metric | Standard Kalman | Rate-Based Kalman | Target |
|--------|----------------|-------------------|---------|
| Max deviation during outlier | 10.3 kg | 7.62 kg | < 3.0 kg |
| Outliers detected (>3σ) | 3 | 1 | 3 |
| Avg recovery deviation | 5.63 kg | 1.88 kg | < 3.0 kg |
| Final estimate | 82.74 kg | 85.58 kg | ~86.5 kg |
| **Result** | ❌ FAIL | ❌ FAIL | - |

### User 090CF20AAA7F495595A30C0F3FEE34BE
**Problem**: Extreme low outliers (32-34kg) for a ~118kg person

#### Data Pattern
- **True baseline**: ~118.5 kg
- **Outliers**: 33.9 kg, 32.2 kg, 34.5 kg (72% below baseline!)
- **Critical point**: 118.3 kg reading immediately after first outlier (should help recovery)
- **Challenge**: Filter catastrophically fails, accepting the 32kg values

#### Test Metrics
| Metric | Standard Kalman | Rate-Based Kalman | Target |
|--------|----------------|-------------------|---------|
| Max deviation during outlier | 60.88 kg | 85.99 kg | < 5.0 kg |
| Outliers detected (>3σ) | 3 | 0 | 3 |
| Avg recovery deviation | 15.7 kg | 8.05 kg | < 5.0 kg |
| Final estimate | 115.44 kg | 119.2 kg | ~118.5 kg |
| **Result** | ❌ FAIL | ❌ FAIL | - |

## Key Findings

### Current Issues

1. **Insufficient Outlier Rejection**
   - Both filters accept physiologically impossible values
   - 30-70% deviations from baseline are not properly rejected
   - Trust scores for unreliable sources are too high

2. **Poor Recovery After Outliers**
   - Filters take too long to recover to baseline after outliers
   - Even when correct values appear, filters remain biased by outliers

3. **Rate-Based Filter Problems**
   - Sometimes performs worse than standard filter on extreme outliers
   - Rate limiting not aggressive enough for impossible changes

### Performance Metrics

| Filter Type | Processing Speed | Outlier Handling | Recovery | Overall |
|------------|------------------|------------------|----------|---------|
| Standard | ~4,800 readings/sec | Poor | Poor | ❌ |
| Rate-Based | ~5,100 readings/sec | Slightly Better | Better | ❌ |

## Required Improvements

### 1. Physiological Constraints
```python
# Recommended limits
MAX_DAILY_WEIGHT_CHANGE = 2.0  # kg/day
MAX_INSTANT_CHANGE_PERCENT = 10  # % from baseline
MIN_REALISTIC_WEIGHT = 40  # kg for adults
```

### 2. Enhanced Trust Scoring
```python
# Current problematic trust scores
"https://api.iglucose.com": 0.05,  # Still too high!
"patient-device": 0.05,  # Needs adjustment

# Recommended adjustments
"https://api.iglucose.com": 0.001,  # Known to produce errors
"patient-device": 0.01,  # Reduce further
```

### 3. Outlier Detection Rules
- Reject any reading > 30% different from rolling baseline
- Require multiple consistent readings to shift baseline significantly
- Use adaptive thresholds based on measurement history

### 4. Recovery Strategy
- Quick recovery when valid readings appear after outliers
- Don't let outliers poison the state estimate for extended periods
- Consider reset mechanism for catastrophic filter divergence

## Adding New Test Cases

To add a new problematic user to the benchmark:

1. Identify user with outlier issues from `output/users/*.json`
2. Add to `benchmark_kalman.py` in `_load_test_cases()`:

```python
self.test_cases['USER_ID_HERE'] = {
    'description': 'Brief description of the problem',
    'expected_behavior': 'What the filter should do',
    'readings': [
        {'date': '2025-01-01 00:00:00', 'weight': 75.0, 'source': 'source_type'},
        # ... more readings
    ],
    'metrics': {
        'true_baseline': 75.0,  # Approximate true weight
        'outlier_indices': [3, 5, 7],  # Indices of outlier readings
        'max_acceptable_deviation': 3.0  # kg from baseline
    }
}
```

3. Run benchmark to evaluate performance

## Benchmark Output Structure

```json
{
  "USER_ID": {
    "standard": {
      "test_name": "USER_ID",
      "filter_type": "standard",
      "description": "Problem description",
      "processing_time_ms": 3.5,
      "readings_per_second": 4800,
      "metrics": {
        "outlier_rejection": {...},
        "recovery": {...},
        "overall_performance": {...}
      }
    },
    "rate_based": {
      // Similar structure
    }
  }
}
```

## Success Criteria

A Kalman filter implementation passes the benchmark when:

1. **Outlier Rejection**: Max deviation < target threshold during outliers
2. **Recovery**: Average deviation < threshold after outliers  
3. **Final Accuracy**: Final estimate within 2% of true baseline
4. **Detection Rate**: Correctly identifies >80% of outliers (>3σ rule)

## Next Steps

1. **Implement physiological constraints** in the Kalman filter
2. **Adjust trust scores** for problematic sources
3. **Add adaptive outlier rejection** based on historical data
4. **Test with more users** to ensure robustness
5. **Create unit tests** based on these benchmarks

## Related Files

- `benchmark_kalman.py` - Main benchmark script
- `src/filters/kalman_filter.py` - Kalman filter implementations
- `output/users/*.json` - Source data for test cases
- `output/benchmark_results_*.json` - Benchmark results

## Usage in CI/CD

```bash
# Run benchmarks and check for failures
uv run python benchmark_kalman.py
if [ $? -ne 0 ]; then
    echo "Kalman filter benchmarks failed"
    exit 1
fi
```

## Monitoring Performance

Track these metrics over time:
- Processing speed (readings/second)
- Outlier detection rate
- Recovery time after outliers
- Final estimate accuracy

Use the benchmark regularly to ensure filter improvements don't regress on known problem cases.