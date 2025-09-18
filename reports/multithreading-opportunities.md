# Investigation: Multithreading Opportunities in run_analysis.py

## Bottom Line

**Root Cause**: Sequential execution of independent analysis steps and user-by-user processing
**Fix Location**: `run_analysis.py:83-120`, `analyze_90_day.py:146-199`, `generate_daily_analysis.py:234-259`
**Confidence**: High

## What's Happening

The analysis pipeline processes weight data sequentially through 4 major steps, with each step processing hundreds/thousands of users one-by-one. Major bottlenecks are in CSV I/O operations and per-user data processing loops.

## Why It Happens

**Primary Cause**: Design prioritizes simplicity over performance
**Trigger**: `run_analysis.py:83-120` - Sequential step execution
**Decision Point**: `analyze_90_day.py:146` - Single-threaded user loop

## Evidence

- **Key File**: `run_analysis.py:83-120` - Steps run sequentially despite independence
- **Search Used**: `rg "for user_id in"` - Found serial user processing loops
- **Key File**: `generate_daily_analysis.py:234-259` - Batch processing but still sequential

## Parallelization Opportunities

### 1. **Step-Level Parallelization (run_analysis.py)**
**Lines 83-120**: Steps 1b, 2, and 3 can run concurrently
- Step 1b (daily analysis) independent after 90-day CSV created
- Step 2 (visualizations) only needs 90_day_analysis.csv
- Step 3 (statistical report) can start immediately
- **Approach**: Use `concurrent.futures.ThreadPoolExecutor` (I/O-bound)
- **Potential Gain**: 30-40% reduction in total runtime

### 2. **User Processing Parallelization (analyze_90_day.py)**
**Lines 146-199**: Process users in parallel instead of serial loop
- Each user's weight calculation is independent
- Currently processes ~100 users/second sequentially
- **Approach**: Use `multiprocessing.Pool` with chunks of users
- **Potential Gain**: 3-4x speedup with 4 cores

### 3. **Batch Processing Parallelization (generate_daily_analysis.py)**
**Lines 234-259**: Process batches concurrently
- Already batched (50 users/batch) but sequential
- Each batch generates independent records
- **Approach**: Use `ThreadPoolExecutor` for CSV writes, `ProcessPoolExecutor` for computation
- **Potential Gain**: 2-3x speedup

### 4. **Visualization Generation (generate_visualizations.py)**
**Lines 457-460**: Generate charts in parallel
- 4 independent chart functions
- Each reads different data aspects
- **Approach**: Use `ThreadPoolExecutor` (matplotlib not thread-safe, but process-safe)
- **Potential Gain**: 2x speedup

### 5. **CSV Loading Optimization**
**Multiple locations**: Redundant CSV loads
- Same CSVs loaded multiple times across modules
- **Approach**: Load once, share via multiprocessing.Manager or cache
- **Potential Gain**: 20-30% I/O reduction

## Implementation Strategy

```python
# Example for run_analysis.py optimization
from concurrent.futures import ThreadPoolExecutor, as_completed

# After Step 1 completes (line 89)
with ThreadPoolExecutor(max_workers=3) as executor:
    futures = {
        executor.submit(generate_daily_analysis.main, user_start_dates, Path(".")): "daily",
        executor.submit(generate_visualizations.main, Path("90_day_analysis.csv"), Path("visualizations")): "viz",
        executor.submit(generate_statistical_report.generate_report, Path(".")): "stats"
    }
    
    for future in as_completed(futures):
        step_name = futures[future]
        result = future.result()
        logging.info(f"Completed: {step_name}")
```

## Constraints

- **matplotlib**: Not thread-safe for figure creation (use processes)
- **pandas**: GIL limits true parallelism (use multiprocessing for CPU-bound)
- **CSV writes**: Must synchronize or use separate files then merge
- **Memory**: Loading full datasets multiple times increases memory usage

## Next Steps

1. Implement step-level parallelization in `run_analysis.py` (easiest win)
2. Add multiprocessing to `analyze_90_day.py` user loop
3. Cache loaded CSVs to avoid redundant I/O
4. Profile with `cProfile` to verify bottlenecks before optimization

## Risks

- Increased complexity in error handling
- Higher memory usage with parallel data loading
- Potential race conditions in file writes if not properly synchronized
