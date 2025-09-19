# Plan: Unified Kalman-Centric Quality Scoring System

## Decision

**Approach**: Replace dual validation systems with single Kalman-deviation-based quality scorer
**Why**: Current dual system creates complexity; Kalman filter already encodes expected behavior, making deviation from it the natural quality metric
**Risk Level**: Medium (affects all weight processing)

## Implementation Steps

### Phase 1: Core Scoring Framework
1. **Create `src/processing/unified_quality_scorer.py`** - New unified scorer with Kalman-centric approach
2. **Update `src/processing/processor.py:248-350`** - Replace dual validation logic with unified scorer
3. **Add `config/quality_scoring.toml`** - Configuration for component weights and thresholds

### Phase 2: Kalman Deviation Component (Primary)
4. **Implement `calculate_kalman_fit()`** - Compare measurement to Kalman prediction
   - Use Mahalanobis distance for proper uncertainty weighting
   - Normalize by innovation covariance from Kalman filter
5. **Add `calculate_innovation_consistency()`** - Check if innovation is within expected bounds
   - Use chi-squared test on normalized innovation
   - Account for adaptive noise during reset periods

### Phase 3: Practical Anomaly Detection
6. **Implement `calculate_temporal_consistency()`** - Rate-of-change validation
   - Time-based thresholds (3kg/6hr, 2kg/24hr, 2kg/day sustained)
   - Account for measurement gaps and trends
7. **Add `calculate_cross_user_detection()`** - Different user on same scale
   - Sudden jumps > 10kg with immediate return
   - Pattern: A → B → A within 24 hours
8. **Create `calculate_data_entry_detection()`** - Unit confusion and typos
   - kg/lbs confusion (2.2x factor detection)
   - Decimal point errors (10x factor)
   - BMI vs weight detection (15-50 range check)

### Phase 4: Contextual Scoring
9. **Add `calculate_source_reliability()`** - Source-based trust scoring
   - Use existing SOURCE_PROFILES reliability ratings
   - Apply noise multipliers as quality penalties
10. **Implement `calculate_trend_alignment()`** - Consistency with established trend
    - Linear regression on recent Kalman states
    - Penalize deviations from trend direction

### Phase 5: Integration & Migration
11. **Update `QualityScore` dataclass** - Add new component fields
12. **Modify `processor.py` validation flow** - Single quality check point
13. **Create migration utilities** - Map old thresholds to new system
14. **Update feature flags** - Add `unified_quality_scoring` toggle

## Files to Change

- `src/processing/unified_quality_scorer.py` - [New file with unified scorer]
- `src/processing/processor.py:248-350` - [Replace dual validation with unified call]
- `src/processing/quality_scorer.py` - [Mark deprecated, keep for backward compatibility]
- `src/processing/validation.py` - [Mark deprecated, extract utility functions]
- `config.toml` - [Add quality_scoring section with component weights]
- `src/constants.py` - [Add quality scoring thresholds]
- `tests/test_unified_quality_scorer.py` - [Comprehensive test suite]

## Scoring Mathematics

**Overall Score Calculation:**
```
S = Π(c_i^w_i)^(1/Σw_i)  # Weighted geometric mean
```
Where:
- c_i = component score (0-1)
- w_i = component weight
- Geometric mean penalizes low scores more than arithmetic

**Component Weights:**
- Kalman Fit: 0.40 (primary signal)
- Temporal Consistency: 0.20
- Anomaly Detection: 0.20
- Source Reliability: 0.10
- Trend Alignment: 0.10

**Kalman Deviation Scoring:**
```
innovation = measurement - kalman_prediction
normalized_innovation = innovation / sqrt(innovation_covariance)
chi_squared = normalized_innovation^2
score = 1 - CDF(chi_squared, df=1)
```

## Test Data Strategy

### Test Scenarios:
1. **Normal Daily Variation** - ±1kg fluctuations (score: 0.85-0.95)
2. **Weight Loss Trend** - -0.2kg/day sustained (score: 0.90-0.95)
3. **Post-Meal Variation** - +2kg after large meal (score: 0.70-0.80)
4. **Different User** - 70kg → 95kg → 70kg (score: <0.20)
5. **Scale Error** - Consistent +5kg offset (score: 0.30-0.50)
6. **Unit Confusion** - 70kg as 154lbs entered as kg (score: <0.10)
7. **Post-Vacation** - +3kg after week gap (score: 0.60-0.70)
8. **Illness Drop** - -2kg in 2 days (score: 0.50-0.60)
9. **Morning/Evening** - 2kg daily swing (score: 0.75-0.85)
10. **Clothing Change** - +1kg winter clothes (score: 0.80-0.90)

## Acceptance Criteria

- [ ] Single quality score replaces all validation checks
- [ ] Kalman deviation is primary quality signal (40% weight)
- [ ] Detects different user within 2 measurements
- [ ] Catches kg/lbs confusion with >95% accuracy
- [ ] Allows legitimate ±2kg/day variations
- [ ] Adapts thresholds during reset periods
- [ ] All existing tests pass with new system
- [ ] Performance: <10ms per measurement

## Risks & Mitigations

**Main Risk**: Breaking existing working validation
**Mitigation**: Feature flag rollout, A/B testing with parallel validation

**Secondary Risk**: Over-rejection during adaptation periods
**Mitigation**: Detect reset states and relax thresholds appropriately

## Out of Scope

- Machine learning models for anomaly detection
- Historical pattern mining beyond 20 measurements
- Multi-user household detection algorithms
- Automatic unit detection from value patterns