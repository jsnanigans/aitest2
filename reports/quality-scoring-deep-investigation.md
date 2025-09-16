# Investigation: Quality Scoring Components Deep Analysis

## Bottom Line
**Root Cause**: Quality scoring uses sophisticated multi-factor analysis with edge case vulnerabilities
**Fix Location**: `src/processing/quality_scorer.py:119-345`
**Confidence**: High

## What's Happening
The quality scoring system evaluates weight measurements through four components (safety, plausibility, consistency, reliability) combined via harmonic mean. Each component has complex mathematical formulas with potential edge cases and boundary conditions that could cause unexpected behavior.

## Why It Happens
**Primary Cause**: Complex mathematical operations with edge case vulnerabilities
**Trigger**: `quality_scorer.py:94` - Main calculation entry point
**Decision Points**: 
- `quality_scorer.py:124` - Safety critical threshold bypass
- `quality_scorer.py:154` - Harmonic vs arithmetic mean selection
- `outlier_detection.py:87` - Quality override mechanism

## Evidence

### 1. Safety Component Edge Cases
**Key File**: `quality_scorer.py:171-201` - Safety calculation
**Mathematical Formula**: `score = exp(-3 * distance_ratio)` with BMI multipliers
**Issues Found**:
- Division by zero risk when `sus_min == abs_min` (line 188)
- BMI calculation assumes fixed height (1.67m default)
- Exponential decay can produce near-zero scores for edge weights

### 2. Plausibility Component Vulnerabilities
**Key File**: `quality_scorer.py:203-266` - Plausibility calculation  
**Mathematical Formula**: Z-score with trend adjustment
**Issues Found**:
- Linear regression can fail with < 2 weights (line 383)
- R² calculation has division by zero risk when `ss_tot == 0` (line 396)
- Trend detection amplifies variance for strong trends (line 233)
- Minimum std dev of 0.5kg may be too rigid for stable weights

### 3. Consistency Component Boundary Issues
**Key File**: `quality_scorer.py:267-316` - Consistency calculation
**Time-based Thresholds**:
- < 6 hours: 3.0 kg max
- 6-24 hours: 2.0 kg ideal, 4.0 kg acceptable
- > 24 hours: Daily rate evaluation
**Issues Found**:
- No handling for negative `time_diff_hours`
- Percentage calculation risk when baseline approaches zero (line 296)
- Hard-coded physiological max (6.44 kg/day) may not suit all populations

### 4. Reliability Component Static Mapping
**Key File**: `quality_scorer.py:318-345` - Reliability calculation
**Source Profiles**: `constants.py:76-175`
**Issues Found**:
- Static outlier rates don't adapt to actual performance
- Unknown sources get moderate score (0.6) which may be too generous
- Outlier rate multipliers have abrupt transitions (5%, 20%, 50%)

### 5. Harmonic Mean Zero Vulnerability
**Key File**: `quality_scorer.py:347-361` - Harmonic mean
**Critical Issue**: Returns 0.0 if ANY component is zero (line 359)
**Impact**: Single failing component causes total rejection regardless of others

## Interaction Effects

### Quality Override Mechanism
**Location**: `outlier_detection.py:77-92`
- Measurements with quality > 0.7 are protected from outlier detection
- Creates feedback loop where high quality prevents statistical validation
- Can allow bad data through if source is trusted but malfunctioning

### Feature Toggle Dependencies
**Location**: `feature_manager.py:18-25`
- Quality scoring requires Kalman filtering
- Quality override requires both quality scoring AND outlier detection
- Disabling individual components returns perfect scores (1.0)

### Adaptive Weighting During Resets
**Location**: `processor.py:296-309`
- After resets, reliability weight increases to 30% (from 15%)
- Consistency weight decreases to 15% (from 25%)
- Can cause acceptance of questionable data from trusted sources

## Next Steps

1. **Add zero-division protection** in plausibility trend calculation (line 396)
2. **Implement adaptive outlier rates** instead of static profiles
3. **Add negative time handling** in consistency component
4. **Consider geometric mean** as alternative to harmonic mean for less severe penalization
5. **Make height configurable** per user for accurate BMI calculations
6. **Add component score logging** for debugging rejected measurements

## Risks
- **Mathematical errors** causing crashes on edge cases
- **Over-rejection** due to harmonic mean severity
- **Under-validation** when quality overrides statistical checks
