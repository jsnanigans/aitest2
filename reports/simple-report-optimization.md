# Investigation: simple_report.py Performance Optimization

## Bottom Line
**Root Cause**: Inefficient per-user datetime parsing and redundant DataFrame operations
**Fix Location**: `simple_report.py:62-113` (get_closest_weight function)
**Confidence**: High

## What's Happening
The script processes ~970K weight measurements across 5,644 users, taking 18.94 seconds. The main bottleneck is calling `get_closest_weight()` 11,168 times (twice per user), each time re-parsing datetime strings and performing inefficient DataFrame operations.

## Why It Happens
**Primary Cause**: Repeated datetime parsing and non-vectorized operations
**Trigger**: `simple_report.py:243-244` - Calls get_closest_weight() in a loop for each user
**Decision Point**: `simple_report.py:94` - pd.to_datetime() called 22,338 times total

## Evidence
- **Profiling Data**: `get_closest_weight()` consumes 16.85s of 18.94s total (89%)
- **Key Bottleneck**: `pd.to_datetime()` called 22,338 times, taking 6.06s (32%)
- **Search Pattern**: `uv run python profile_simple_report.py` - Shows datetime parsing dominates
- **Optimization Result**: Reduced to 4.95s (74% improvement)

## Key Optimizations Applied

1. **Parse Dates Once During CSV Load** (saves ~6s)
   - Before: Parse datetime for each user lookup
   - After: `parse_dates=['effectiveDateTime']` during pd.read_csv()

2. **Vectorized Weight Lookup** (saves ~10s)
   - Before: Individual DataFrame operations per user
   - After: Single grouped operation for all users

3. **Efficient Data Filtering** (saves ~1s)
   - Before: Multiple .loc operations with copy warnings
   - After: Boolean indexing without intermediate copies

4. **NumPy for Statistics** (saves ~0.5s)
   - Before: Python list comprehensions
   - After: NumPy arrays for calculations

## Next Steps
1. Replace original with optimized version: `mv simple_report_optimized.py simple_report.py`
2. Add caching for employer lookups if same employer used repeatedly
3. Consider parallel processing for very large datasets (>10M rows)

## Risks
- Memory usage slightly higher due to upfront datetime parsing
- Requires pandas >= 2.0 for ISO8601 date format support
