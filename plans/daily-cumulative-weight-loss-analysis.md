# Plan: Daily Cumulative Weight Loss Analysis with Detailed Report

## Decision
**Approach**: Generate comprehensive CSV report with daily weight snapshots using existing "closest value" logic
**Why**: Provides granular day-by-day comparison showing exactly how raw vs filtered data differs over time
**Risk Level**: Medium (large output files, performance considerations)

## Implementation Steps

1. **Create daily analysis module** - Add `create-report/generate_daily_analysis.py`
   - Import `get_weight_at_date` function from `analyze_90_day.py`
   - Generate daily snapshots from start_date to day 90 (or beyond)
   - Use same 20-day window for closest value selection

2. **Implement core daily calculation** - Main function structure:
   ```python
   def generate_daily_report(user_start_dates, output_days=180):
       """Generate day-by-day weight analysis for all users."""
       # For each user
       # For each day from start to output_days
       # Get closest raw weight using get_weight_at_date
       # Get closest filtered weight using get_weight_at_date
       # Calculate cumulative loss from start
       # Record all metrics
   ```

3. **Output detailed CSV structure** - Columns per row:
   - `user_id` - User identifier
   - `day_number` - Days since start (0, 1, 2, ..., 180)
   - `date` - Actual calendar date
   - `raw_weight` - Closest raw weight value (or NULL)
   - `raw_days_offset` - How many days away the raw measurement was
   - `filtered_weight` - Closest filtered weight value (or NULL)
   - `filtered_days_offset` - How many days away the filtered measurement was
   - `raw_cumulative_loss_kg` - Weight lost since start (raw)
   - `raw_cumulative_loss_pct` - Percentage lost since start (raw)
   - `filtered_cumulative_loss_kg` - Weight lost since start (filtered)
   - `filtered_cumulative_loss_pct` - Percentage lost since start (filtered)
   - `divergence_kg` - Difference between raw and filtered (kg)
   - `divergence_pct` - Difference between raw and filtered (%)
   - `has_raw_measurement` - Boolean: was there a raw measurement within window
   - `has_filtered_measurement` - Boolean: was there a filtered measurement within window

4. **Add summary statistics file** - Create companion `daily_analysis_summary.json`:
   - Total records generated
   - Users analyzed
   - Average data availability (% of days with measurements)
   - Maximum divergence observed
   - Days where raw/filtered disagree on gain/loss direction

5. **Integrate into pipeline** - Update `run_analysis.py`:
   ```python
   # After line 80 (90-day analysis)
   logging.info("STEP 1b: GENERATING DAILY DETAIL REPORT")
   generate_daily_analysis.main(user_start_dates, Path("."))
   ```

6. **Add batch processing** - Handle memory constraints:
   - Process users in batches of 50
   - Write to CSV incrementally
   - Show progress every 10 users
   - Estimate completion time

## Files to Change

- `create-report/run_analysis.py:80` - Add daily analysis step
- `create-report/analyze_90_day.py:72-100` - Export get_weight_at_date function
- NEW: `create-report/generate_daily_analysis.py` - Main daily analysis module
- `create-report/FINAL_REPORT.md` - Add section for daily analysis findings

## Code Structure

```python
# create-report/generate_daily_analysis.py
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional
import logging
import json
from analyze_90_day import get_weight_at_date, RAW_FILE, FILTERED_FILE

def generate_daily_report(
    user_start_dates: Dict[str, datetime],
    output_path: Path,
    max_days: int = 180,
    batch_size: int = 50
) -> Dict:
    """
    Generate detailed daily weight analysis for all users.
    
    Args:
        user_start_dates: Dict mapping user_id to start_date
        output_path: Directory to save output files
        max_days: Maximum days to analyze (default 180)
        batch_size: Users to process per batch
        
    Returns:
        Summary statistics dictionary
    """
    output_file = output_path / "daily_weight_analysis.csv"
    
    # Load data once
    logging.info("Loading weight data files...")
    df_raw = pd.read_csv(RAW_FILE)
    df_filtered = pd.read_csv(FILTERED_FILE)
    df_raw['effectiveDateTime'] = pd.to_datetime(df_raw['effectiveDateTime'])
    df_filtered['effectiveDateTime'] = pd.to_datetime(df_filtered['effectiveDateTime'])
    
    # Initialize CSV with headers
    first_batch = True
    total_records = 0
    
    # Process users in batches
    user_ids = list(user_start_dates.keys())
    for batch_start in range(0, len(user_ids), batch_size):
        batch_end = min(batch_start + batch_size, len(user_ids))
        batch_users = user_ids[batch_start:batch_end]
        
        logging.info(f"Processing users {batch_start+1}-{batch_end} of {len(user_ids)}")
        
        batch_records = []
        for user_id in batch_users:
            start_date = user_start_dates[user_id]
            user_raw = df_raw[df_raw['user_id'] == user_id]
            user_filtered = df_filtered[df_filtered['user_id'] == user_id]
            
            # Get start weights for reference
            start_raw = get_weight_at_date(user_raw, start_date)
            start_filtered = get_weight_at_date(user_filtered, start_date)
            
            # Generate daily records
            for day_num in range(max_days + 1):
                current_date = start_date + timedelta(days=day_num)
                
                # Get weights with timing info
                raw_weight, raw_offset = get_weight_with_offset(user_raw, current_date)
                filtered_weight, filtered_offset = get_weight_with_offset(user_filtered, current_date)
                
                # Calculate cumulative losses
                raw_loss_kg = (start_raw - raw_weight) if start_raw and raw_weight else None
                raw_loss_pct = (raw_loss_kg / start_raw * 100) if raw_loss_kg else None
                
                filtered_loss_kg = (start_filtered - filtered_weight) if start_filtered and filtered_weight else None
                filtered_loss_pct = (filtered_loss_kg / start_filtered * 100) if filtered_loss_kg else None
                
                # Calculate divergence
                div_kg = (filtered_weight - raw_weight) if filtered_weight and raw_weight else None
                div_pct = (filtered_loss_pct - raw_loss_pct) if filtered_loss_pct and raw_loss_pct else None
                
                batch_records.append({
                    'user_id': user_id,
                    'day_number': day_num,
                    'date': current_date.strftime('%Y-%m-%d'),
                    'raw_weight': raw_weight,
                    'raw_days_offset': raw_offset,
                    'filtered_weight': filtered_weight,
                    'filtered_days_offset': filtered_offset,
                    'raw_cumulative_loss_kg': raw_loss_kg,
                    'raw_cumulative_loss_pct': raw_loss_pct,
                    'filtered_cumulative_loss_kg': filtered_loss_kg,
                    'filtered_cumulative_loss_pct': filtered_loss_pct,
                    'divergence_kg': div_kg,
                    'divergence_pct': div_pct,
                    'has_raw_measurement': raw_weight is not None,
                    'has_filtered_measurement': filtered_weight is not None
                })
        
        # Write batch to CSV
        df_batch = pd.DataFrame(batch_records)
        df_batch.to_csv(output_file, mode='w' if first_batch else 'a', 
                       header=first_batch, index=False)
        first_batch = False
        total_records += len(batch_records)
        
        logging.info(f"  Written {len(batch_records)} records")
    
    # Generate summary statistics
    summary = generate_summary_stats(output_file, total_records, len(user_ids))
    
    # Save summary
    with open(output_path / "daily_analysis_summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    return summary

def get_weight_with_offset(df: pd.DataFrame, target_date: datetime, 
                          window_days: int = 20) -> Tuple[Optional[float], Optional[int]]:
    """
    Get weight and days offset from target date.
    Returns (weight, days_offset) or (None, None).
    """
    if df.empty:
        return None, None
    
    df = df.copy()
    df['time_diff'] = (df['effectiveDateTime'] - target_date).dt.days
    df['abs_diff'] = abs(df['time_diff'])
    
    window = timedelta(days=window_days)
    df_window = df[df['abs_diff'] <= window_days]
    
    if df_window.empty:
        return None, None
    
    closest_idx = df_window['abs_diff'].idxmin()
    return df_window.loc[closest_idx, 'weight'], df_window.loc[closest_idx, 'time_diff']
```

## Acceptance Criteria

- [ ] Uses exact same `get_weight_at_date` logic as existing 90-day analysis
- [ ] Generates CSV with one row per user per day (up to 180 days)
- [ ] Includes offset information showing how far measurements were from target date
- [ ] Handles missing data gracefully (NULL values, not interpolation)
- [ ] Processes 1000 users × 180 days in under 60 seconds
- [ ] Output CSV is properly formatted and loadable in Excel/pandas
- [ ] Summary JSON includes key statistics and data quality metrics

## Risks & Mitigations

**Main Risk**: Output file size (1000 users × 180 days = 180,000 rows)
**Mitigation**: Incremental CSV writing, option to limit days or users

**Secondary Risk**: Memory usage loading all weight data
**Mitigation**: Filter data to target users immediately after loading

**Third Risk**: Performance with many date lookups
**Mitigation**: Consider caching or pre-indexing by user_id and date

## Out of Scope

- Interpolation between measurements
- Statistical significance testing
- Visualization of individual trajectories
- Real-time updates
- Compression of output files

## Next Steps

1. Review and approve this detailed plan
2. Implement `generate_daily_analysis.py` with batch processing
3. Test with small subset (10 users, 30 days)
4. Validate that closest value logic matches existing implementation
5. Run full analysis and review output format
6. Add summary statistics and documentation