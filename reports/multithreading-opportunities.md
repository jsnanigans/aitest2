# Investigation: Additional Multithreading Opportunities in Report Generation

## Bottom Line

**Root Cause**: Statistical calculations and dashboard generation contain multiple sequential loops processing independent users
**Fix Location**: `generate_statistical_report.py:43-259`, `generate_dashboard.py:374-400`
**Confidence**: High

## What's Happening

The report generation scripts already use ThreadPoolExecutor and ProcessPoolExecutor for major parallel operations, but statistical tests and visualization generation still process users sequentially in tight loops that could benefit from parallelization.

## Why It Happens

**Primary Cause**: Statistical independence of per-user calculations allows safe parallelization
**Trigger**: `generate_statistical_report.py:43` - Sequential normality tests across users
**Decision Point**: `generate_dashboard.py:388` - Sequential user journey plotting

## Evidence

- **Key File**: `generate_statistical_report.py:43-259` - 4 major loops iterating over users independently
- **Search Used**: `rg "for .* in .*:" generate_statistical_report.py` - Found 7 sequential loops
- **Parallel Already**: `run_analysis.py:168` - ThreadPoolExecutor(max_workers=3) for major steps
- **ProcessPool Usage**: `analyze_90_day.py:364` - ProcessPoolExecutor for batch processing

## Current Parallel Implementation

1. **run_analysis.py:116-188**: Runs Steps 1b, 2, 3 in parallel using ThreadPoolExecutor
2. **analyze_90_day.py:364**: ProcessPoolExecutor for user batch processing (>100 users)  
3. **generate_daily_analysis.py:137**: ProcessPoolExecutor for daily analysis batches
4. **generate_visualizations.py:949**: ThreadPoolExecutor(max_workers=4) for chart generation

## Identified Bottlenecks

### 1. Statistical Tests (generate_statistical_report.py)

**Location**: Lines 43, 84, 120, 186, 259
**Issue**: Sequential processing of per-user statistical tests
**Impact**: ~60-70% potential speedup with 8 cores

```python
# Current sequential approach at line 43
for user_id in sample_users:
    # Shapiro-Wilk test per user
    
# Could parallelize with ThreadPoolExecutor
```

### 2. Dashboard Individual Journeys (generate_dashboard.py)

**Location**: Line 388-400
**Issue**: Sequential plotting of 9 user weight journeys
**Impact**: ~50% speedup possible (matplotlib thread safety permitting)

### 3. Data Cache Loading (data_cache.py)

**Location**: Lines 131-133
**Issue**: Sequential CSV loading during preload
**Impact**: ~30% speedup for initial load with parallel I/O

## Optimization Opportunities

### High Impact (Implement First)

1. **Parallelize Statistical Tests** - `generate_statistical_report.py`
   - Lines 43-71 (normality tests)
   - Lines 84-107 (variance metrics) 
   - Lines 120-152 (smoothness metrics)
   - Estimate: 60-70% speedup on 8-core system
   - Risk: Low (independent calculations)

2. **Concurrent DataFrame Operations** - `analyze_90_day.py`
   - GroupBy operations could use Dask or parallel apply
   - Estimate: 20-30% improvement
   - Risk: Medium (pandas GIL limitations)

### Medium Impact

3. **Parallel Chart Generation** - `generate_dashboard.py`
   - Lines 388-400 (individual journeys)
   - Already partial in visualizations.py
   - Estimate: 30-40% speedup
   - Risk: Medium (matplotlib backend thread safety)

4. **Async I/O for File Writes**
   - Multiple CSV/PNG writes could be concurrent
   - Estimate: 10-20% improvement 
   - Risk: Low

### Low Impact

5. **Pipeline Parallelism**
   - Process different time windows simultaneously
   - Complex implementation for modest gains
   - Estimate: 10-15% improvement
   - Risk: High (synchronization complexity)

## Next Steps

1. Wrap statistical test loops in ThreadPoolExecutor with batch processing
2. Test matplotlib thread safety for dashboard generation parallelization  
3. Consider Dask for large DataFrame operations if dataset grows

## Risks

- **Matplotlib thread safety**: Must use Agg backend or serialize plot calls
- **Memory overhead**: Parallel processing increases memory usage ~2-3x
- **Debugging complexity**: Parallel errors harder to diagnose
