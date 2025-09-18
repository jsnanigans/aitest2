# Investigation: Value Conversions in main.py Processing Pipeline

## Bottom Line
**Root Cause**: Multiple value conversions occur throughout the pipeline - unit conversions (lb→kg), BMI detection, string parsing, and float normalization
**Fix Location**: Primary conversions at `src/processing/validation.py:653-700` and `main.py:349`
**Confidence**: High

## What's Happening
The weight processing pipeline performs several types of value conversions: parsing strings to floats, converting units (pounds/stones to kg), detecting and converting BMI values masquerading as weights, and normalizing data types for Kalman filtering.

## Why It Happens
**Primary Cause**: Healthcare data comes from multiple sources with inconsistent formats
**Trigger**: `main.py:349` - Initial float conversion from CSV string
**Decision Point**: `src/processing/validation.py:679-687` - Unit detection and conversion logic

## Evidence
- **Key File**: `main.py:344-369` - Initial parsing and validation
  - Line 344-345: Strip whitespace and check for NULL values
  - Line 349: Convert string to float with exception handling
  - Line 364: Convert unit to lowercase and strip whitespace
  - Line 367: Filter out BSA measurements by checking source/unit

- **Key File**: `src/processing/validation.py:653-700` - Main conversion logic
  - Lines 680-687: Convert pounds/stones to kilograms
  - Lines 691-699: Detect BMI values and convert to weight using height²
  - Line 629-642: Convert height units to meters for BMI calculations

- **Search Used**: `rg "float\(|convert" -g "*.py"` - Found all conversion points

## Conversion Types Identified

### 1. String to Float (main.py:349)
- Raw CSV values parsed with `float(weight_str)`
- Validation for NaN, Inf, and range (0-1000 kg)
- Exception handling for malformed values

### 2. Unit Conversions (validation.py:680-687)
- Pounds to kg: `weight * 0.453592`
- Stones to kg: `weight * 6.35029`
- Automatic detection via unit field

### 3. BMI Detection (validation.py:691-699)
- Values 15-50 in "kg" units checked for BMI pattern
- Conversion: `weight_from_bmi = bmi * height²`
- Uses user-specific height data from CSV

### 4. Height Conversions (validation.py:629-642)
- cm → m: `value / 100.0`
- inches → m: `value * 0.0254`
- feet → m: `value * 0.3048`

### 5. Case Normalization
- Unit strings: `.lower().strip()` (main.py:364)
- Source strings: `.upper()` for BSA detection (main.py:367)
- NULL checks: `.upper() == "NULL"` (main.py:345)

## Next Steps
1. Add logging for all conversions to track data transformations
2. Validate BMI detection thresholds against actual patient data patterns
3. Consider adding conversion confidence scores to metadata

## Risks
- BMI misdetection could incorrectly multiply weight by height² (40kg → 110kg)
- Unit detection relies on string matching - typos could bypass conversion
- No validation that converted values are physiologically plausible post-conversion