# Strict Unit Validation Documentation

## Overview
As of this update, the weight processing system enforces **strict unit validation** with no assumptions or heuristics. This is a breaking change from previous versions.

## Key Changes

### 1. No BMI Detection
- **REMOVED**: The system no longer attempts to detect if a weight value is actually a BMI
- **REMOVED**: No automatic conversion from BMI to weight (e.g., 25 → 69.75 kg)
- Values are processed exactly as provided with their explicit units

### 2. No Unit Defaults
- **REMOVED**: The system no longer defaults to 'kg' when unit is missing
- **REQUIRED**: Every measurement must have an explicit unit field
- Missing or empty units will cause rejection

### 3. Strict Unit Whitelist
Only the following units are supported:

#### Metric Units
- `kg`, `kilogram`, `kilograms`
- `g`, `gram`, `grams`

#### Imperial Units
- `lb`, `lbs`, `pound`, `pounds`
- `st`, `stone`, `stones`

### 4. Unit Conversion Factors
When supported units are provided, the following conversions are applied:
- Pounds to kg: multiply by 0.453592
- Stones to kg: multiply by 6.35029
- Grams to kg: divide by 1000

## Rejection Reasons

Measurements will be rejected with clear reasons:

1. **Missing Unit**: `"Missing unit - cannot process without explicit unit"`
2. **Unsupported Unit**: `"Unsupported unit: [unit] - only {...} are supported"`
3. **Physiological Limits**: Still enforced based on implied BMI, but no conversion

## Migration Guide

### For Data Providers
1. Ensure all weight measurements include explicit unit fields
2. Use only supported units from the whitelist
3. Do not send BMI values as weight measurements

### For System Administrators
1. Review historical data for measurements that may be rejected
2. Monitor `rejected_units` in processing statistics
3. Consider data cleanup for common unit typos

## Statistics and Monitoring

The system now tracks:
- `unit_rejected`: Count of measurements rejected for unit issues
- `rejected_units`: Dictionary of rejected units and their frequencies

Example output:
```
Processing Complete:
  Measurements rejected for unsupported units: 150
  
Rejected Units Summary:
  'ounce': 45 measurements
  '<missing>': 38 measurements
  'kgs': 25 measurements
  'lbm': 20 measurements
  'invalid_unit': 22 measurements
```

## Testing

Run unit validation tests:
```bash
uv run python tests/test_unit_validation.py
```

Test with sample data:
```bash
uv run python main.py data/test_unit_validation.csv --config test_config.toml
```

## Rollback Instructions

If you need to temporarily revert to the old behavior:

1. Restore the old `src/processing/validation.py` from git
2. Remove `SUPPORTED_WEIGHT_UNITS` from `src/constants.py`
3. Remove unit validation from `main.py` lines 372-383

**Note**: This is not recommended as it reintroduces data quality issues.

## Support

For questions about rejected measurements or unit support, check:
1. The `rejected_units` summary in processing output
2. The `output/results_*.json` file for detailed rejection reasons
3. This documentation for the supported units list