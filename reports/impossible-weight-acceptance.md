# Investigation: 34.56kg Weight Accepted Despite Being Humanly Impossible

## Bottom Line
**Root Cause**: BMI threshold too permissive at 10.0 (allows weights as low as ~29kg for average height)
**Fix Location**: `src/constants.py:line containing IMPOSSIBLE_LOW`
**Confidence**: High

## What's Happening
The system accepted a 34.56 kg weight for user e751ebe4-3e13-423d-bf50-88a9dd13f132 (height: 170.18 cm). This weight produces a BMI of 11.93, which passes the "impossible" threshold but is medically dangerous and likely erroneous.

## Why It Happens
**Primary Cause**: BMI validation threshold too low
**Trigger**: `src/processing/validation.py:712` - BMI check uses `IMPOSSIBLE_LOW = 10.0`
**Decision Point**: `src/constants.py` - `BMI_LIMITS['IMPOSSIBLE_LOW'] = 10.0`

The validation logic correctly calculates BMI (11.93) but only rejects if BMI < 10.0. A BMI of 10.0 allows:
- 29 kg for 1.70m person (severely underweight)
- 25 kg for 1.60m person (life-threatening)
- 34 kg for 1.85m person (dangerously low)

## Evidence
- **Key File**: `src/constants.py` - `BMI_LIMITS['IMPOSSIBLE_LOW'] = 10.0`
- **Search Used**: `rg "e751ebe4-3e13-423d-bf50-88a9dd13f132" data/2025-09-05_nocon.csv | rg "2025-04-10"` - Found 34.56 kg entry
- **Validation**: `src/processing/validation.py:712-714` - Only rejects if BMI < 10.0
- **User Height**: `data/2025-09-11_height_values_latest.csv` - User height is 170.18 cm

## Next Steps
1. Increase `BMI_LIMITS['IMPOSSIBLE_LOW']` from 10.0 to 13.0 (WHO severe thinness threshold)
2. Consider dual validation: BMI check AND absolute weight minimum (e.g., 35 kg)
3. Add source-specific thresholds for patient-device data which may have scale errors

## Risks
- **Current Risk**: Accepting medically impossible weights could affect treatment decisions
- **Data Quality**: Other users may have similarly erroneous low weights being accepted
- **Scale Errors**: Common digital scale error pattern (half-weight when one foot off scale)