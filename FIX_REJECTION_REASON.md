# Fix: Rejection Reason Not Captured in JSON Output

## Issue
User `0093a653-476b-4401-bbec-33a89abc2b18` had a rejected measurement in the visualization but the JSON output showed `"rejection_reason": null`.

## Root Cause
Field name mismatch between `processor.py` and `main.py`:
- `processor.py` returns rejection reason in field named `"reason"`
- `main.py` was looking for field named `"rejection_reason"`

## Solution
Updated `main.py` to use the correct field name:
```python
# Before:
'rejection_reason': result.get('rejection_reason')

# After:
'rejection_reason': result.get('reason')
```

## Files Modified
- `main.py`: Fixed field name references (3 locations)

## Verification
Test confirms rejection reasons are now properly captured:
- Weight outside bounds: "Weight 25.0kg outside bounds [30.0, 400.0]"
- Multi-user detection: "Change of 40.0kg in 0.1h exceeds hydration/bathroom limit"
- Session variance: "Session variance 10.0kg exceeds threshold 5.0kg"

## Impact
All future processing will correctly capture and display rejection reasons in the JSON debug output, making it easier to understand why measurements were rejected.

---
*Fixed: 2025-09-10*
