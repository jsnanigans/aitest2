# Investigation: Quality Scoring vs Legacy Validation

## Bottom Line

**Root Cause**: Two fundamentally different validation philosophies - quality scoring uses multi-factor continuous scoring with weighted components, while legacy validation uses binary pass/fail checks.
**Fix Location**: `src/processing/processor.py:248-372`
**Confidence**: High

## What's Happening

The system has two distinct validation approaches that can be toggled via the `use_quality_scoring` feature flag. The NEW quality scoring system evaluates measurements across multiple dimensions with weighted scores, while the LEGACY validation performs sequential binary checks that can immediately reject measurements.

## Why It Happens

**Primary Cause**: Evolution from rule-based to score-based validation
**Trigger**: `src/processing/processor.py:251` - Feature flag check
**Decision Point**: `src/processing/processor.py:279` - Branch between validation approaches

## Evidence

- **Key File**: `src/processing/processor.py:279-372` - Shows branching logic between approaches
- **Quality Implementation**: `src/processing/quality_scorer.py:94-169` - Multi-component scoring
- **Legacy Implementation**: `src/processing/validation.py:185-242` - Sequential binary checks

## Key Architectural Differences

### 1. Decision Model
- **Quality Scoring**: Continuous scores (0.0-1.0) with configurable threshold
- **Legacy Validation**: Binary pass/fail at each check

### 2. Component Structure

**Quality Scoring Components**:
- `safety` (35%): Physiological limits + BMI validation
- `plausibility` (25%): Statistical deviation with trend awareness
- `consistency` (25%): Rate of change validation
- `reliability` (15%): Source-based scoring

**Legacy Validation Checks**:
- Absolute physiological limits (mandatory)
- Rate of change validation (optional)
- Pattern analysis (informational only)

### 3. Data Structures

**Quality Score Returns** (`QualityScore` object):
```python
{
    'overall': float,          # Combined score
    'components': dict,        # Individual scores
    'threshold': float,        # Acceptance threshold
    'accepted': bool,          # Final decision
    'rejection_reason': str,   # Detailed explanation
    'metadata': dict          # Context data
}
```

**Legacy Validation Returns** (dictionary):
```python
{
    'valid': bool,            # Binary decision
    'weight': float,
    'checks': list,           # Completed checks
    'warnings': list,         # Non-blocking issues
    'rejection_reason': str   # If rejected
}
```

### 4. Adaptability

**Quality Scoring**:
- Dynamically adjusts during adaptation periods
- Configurable component weights
- Threshold can be lowered during initialization
- Uses harmonic mean (penalizes low scores)

**Legacy Validation**:
- Fixed thresholds from constants
- No adaptation mechanism
- All-or-nothing approach

### 5. Edge Case Handling

**Quality Scoring**:
- Trend-aware plausibility (R² > 0.5 triggers trend projection)
- Safety critical threshold (0.3) for early rejection
- Source-specific reliability multipliers
- Graceful degradation with missing data

**Legacy Validation**:
- Hard stops at each validation stage
- No trend awareness
- Limited source-specific adjustments
- Requires all data for full validation

### 6. Configuration

**Quality Scoring**:
- Runtime configurable via `config['quality_scoring']`
- Per-component feature flags
- Adjustable weights and thresholds

**Legacy Validation**:
- Hard-coded constants
- Feature flags only for enable/disable
- No runtime configuration

## Behavioral Differences

1. **During Initialization**: Quality scoring lowers thresholds (0.4) and adjusts weights to be more forgiving
2. **Outlier Handling**: Quality scoring can override outlier detection with high scores (>0.8)
3. **Source Trust**: Quality scoring has granular source reliability (0.5-3.0 multipliers)
4. **Failure Modes**: Quality scoring provides partial credit; legacy is binary rejection

## Next Steps

1. Consider deprecating legacy validation if quality scoring proves stable
2. Add migration path for systems using legacy validation
3. Document threshold tuning guidelines for quality scoring

## Risks

- Systems may behave differently when switching between modes
- Quality scoring may accept marginal data that legacy would reject
- Component weight changes can significantly alter acceptance rates
