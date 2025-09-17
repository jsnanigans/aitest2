# Plan: Weight Loss Interval Analysis Report System

## Executive Summary
This system analyzes weight loss progress by comparing raw measurements against Kalman-filtered data at 30-day intervals. The primary goal is to quantify the impact of outlier rejection and provide insights into data quality's effect on weight loss tracking accuracy.

## Decision
**Approach**: Create standalone analysis module that processes database snapshots at 30-day intervals, comparing raw vs filtered data impact
**Why**: Leverages existing Kalman/quality infrastructure while maintaining separation of concerns for analytical reporting
**Risk Level**: Low

## Rationale & Design Philosophy

### Why This Approach?
1. **Clinical Validity**: 30-day intervals align with standard medical weight loss assessment periods
2. **Statistical Significance**: ±7 day windows provide sufficient data density while accommodating real-world measurement patterns
3. **Quality Assessment**: Comparing raw vs filtered reveals the value of the Kalman/outlier system
4. **Actionable Insights**: Identifies users where data quality most affects outcomes

### Key Design Principles
- **Non-invasive**: Analysis layer doesn't modify existing processing pipeline
- **Reproducible**: Point-in-time snapshots ensure consistent results
- **Scalable**: Streaming architecture handles arbitrary user counts
- **Interpretable**: Clear metrics that non-technical stakeholders understand

## Implementation Steps

### Phase 1: Core Analysis Module
1. **Create `src/analysis/interval_analyzer.py`** - Main analysis engine
   - Extract user signup timestamps from database (initial-questionnaire)
   - Calculate 30-day interval boundaries from signup
   - Select weights within ±7 day windows for each interval
   - Compare raw vs filtered (Kalman) values

2. **Create `src/analysis/weight_selector.py`** - Interval weight selection
   - Implement sliding window logic (target_date ± 7 days)
   - Priority: closest to target date wins
   - Handle missing intervals gracefully
   - Return both raw and filtered values with metadata

3. **Create `src/analysis/statistics.py`** - Statistical comparisons
   - Calculate weight loss percentages (raw vs filtered)
   - Compute outlier impact metrics (difference magnitude)
   - Rank users by raw/filtered divergence
   - Generate summary statistics per interval

### Phase 2: Data Access Layer
4. **Update `src/database/database.py:450`** - Add analysis queries
   - `get_user_signup_date(user_id)` - Extract initial timestamp
   - `get_measurements_in_window(user_id, start, end)` - Fetch range
   - `get_all_users_with_counts()` - List users with measurement counts

5. **Create `src/analysis/data_loader.py`** - Batch data loading
   - Efficient bulk loading for top 200 users
   - Memory-optimized streaming for large datasets
   - Cache interval calculations to avoid recomputation

### Phase 3: Output Generation
6. **Create `src/analysis/csv_generator.py`** - Generate 4 output files
   - `interval_weights.csv` - All interval weights (raw and filtered)
   - `weight_loss_summary.csv` - Per-user loss percentages
   - `outlier_impact.csv` - Outlier rejection statistics
   - `top_200_divergence.csv` - Users with highest raw/filtered difference

7. **Create `src/viz/analysis_charts.py`** - Visualization generation
   - Weight loss distribution histograms
   - Outlier impact scatter plots
   - Time series comparison charts for top users
   - Aggregate statistics dashboard

### Phase 4: Integration & CLI
8. **Create `scripts/generate_analysis_report.py`** - Main entry point
   - Parse command-line arguments (date range, user count)
   - Orchestrate analysis pipeline
   - Handle errors and generate logs
   - Output progress indicators

9. **Update `main.py:850`** - Add analysis mode flag
   - `--analysis-report` flag to trigger report generation
   - `--top-n` parameter for user count (default 200)
   - `--interval-days` to override 30-day default

## Files to Change
- `src/database/database.py:450` - Add analysis query methods
- `main.py:850` - Add CLI flags for analysis mode
- `config.toml:175` - Add `[analysis]` section for parameters

## New Files to Create
- `src/analysis/interval_analyzer.py` - Core analysis logic
- `src/analysis/weight_selector.py` - Window selection algorithm
- `src/analysis/statistics.py` - Statistical computations
- `src/analysis/data_loader.py` - Efficient data loading
- `src/analysis/csv_generator.py` - CSV output formatting
- `src/viz/analysis_charts.py` - Analysis visualizations
- `scripts/generate_analysis_report.py` - Standalone runner
- `tests/test_interval_analysis.py` - Unit tests

## Acceptance Criteria
- [ ] Generate accurate 30-day intervals from signup date
- [ ] Select weights within ±7 day windows correctly
- [ ] Handle missing intervals without crashes
- [ ] Calculate raw vs filtered differences accurately
- [ ] Output 4 CSV files with correct schemas
- [ ] Generate visualization charts for top 200 users
- [ ] Process 10,000+ users in under 60 seconds
- [ ] Memory usage stays under 2GB for large datasets

## Algorithm Details

### Interval Weight Selection Algorithm
```python
def select_interval_weight(measurements, target_date, window_days=7):
    """
    Selects the most appropriate weight measurement for a given interval.

    Rationale:
    - ±7 days provides 15-day window (clinically acceptable for weight tracking)
    - Prioritizes proximity to target date over source reliability
    - Returns both raw and filtered values for comparison
    """
    # Filter to window
    in_window = [m for m in measurements
                 if abs((m.timestamp - target_date).days) <= window_days]

    if not in_window:
        return None

    # Sort by multiple criteria
    in_window.sort(key=lambda m: (
        abs(m.timestamp - target_date),  # Primary: closest to target
        -m.quality_score,                 # Secondary: higher quality
        m.source != 'patient-device'      # Tertiary: prefer device data
    ))

    return {
        'raw_weight': in_window[0].raw_weight,
        'filtered_weight': in_window[0].filtered_weight,
        'timestamp': in_window[0].timestamp,
        'quality_score': in_window[0].quality_score,
        'source': in_window[0].source,
        'distance_days': abs((in_window[0].timestamp - target_date).days)
    }
```

### Statistical Analysis Methods

#### Weight Loss Calculation
```python
def calculate_weight_loss(initial, current):
    """Calculate absolute and percentage weight loss"""
    if initial is None or current is None:
        return {'absolute': None, 'percentage': None}

    absolute_loss = initial - current
    percentage_loss = (absolute_loss / initial) * 100

    return {
        'absolute': absolute_loss,
        'percentage': percentage_loss,
        'is_gain': absolute_loss < 0
    }
```

#### Outlier Impact Metrics
```python
def calculate_outlier_impact(raw_weights, filtered_weights):
    """
    Quantify the effect of outlier rejection on weight tracking

    Returns:
    - total_deviation: Sum of all differences
    - avg_deviation: Mean difference per measurement
    - max_deviation: Largest single difference
    - deviation_pattern: 'consistent', 'sporadic', or 'systematic'
    """
    deviations = []
    for raw, filtered in zip(raw_weights, filtered_weights):
        if raw is not None and filtered is not None:
            deviations.append(abs(raw - filtered))

    if not deviations:
        return {'impact': 'no_data'}

    return {
        'total_deviation': sum(deviations),
        'avg_deviation': np.mean(deviations),
        'max_deviation': max(deviations),
        'std_deviation': np.std(deviations),
        'outlier_rate': len([d for d in deviations if d > 2]) / len(deviations),
        'pattern': classify_deviation_pattern(deviations)
    }
```

#### User Divergence Scoring
```python
def calculate_divergence_score(user_data):
    """
    Score users by how much filtering affects their results

    Components:
    - Magnitude: Average absolute difference
    - Consistency: Variance of differences
    - Direction: Whether filtering consistently adds/removes weight
    - Clinical Impact: Does it change weight loss classification?
    """
    raw_trajectory = calculate_trajectory(user_data['raw'])
    filtered_trajectory = calculate_trajectory(user_data['filtered'])

    score = {
        'magnitude': np.mean(np.abs(raw_trajectory - filtered_trajectory)),
        'consistency': 1 / (1 + np.var(raw_trajectory - filtered_trajectory)),
        'direction_bias': np.mean(raw_trajectory - filtered_trajectory),
        'changes_outcome': clinical_outcome_changed(raw_trajectory, filtered_trajectory),
        'data_quality': user_data['quality_scores'].mean()
    }

    # Weighted composite score
    score['composite'] = (
        score['magnitude'] * 0.4 +
        (1 - score['consistency']) * 0.2 +
        abs(score['direction_bias']) * 0.2 +
        score['changes_outcome'] * 0.2
    )

    return score
```

## CSV Schemas

### 1. filtered_data.csv
**Purpose**: Clean dataset with all rejected measurements removed
**Use Case**: Baseline for filtered analysis
```csv
user_id,weight,source,timestamp,bmi,quality_score
USR001,185.2,patient-device,2024-01-15 08:30:00,28.3,0.92
USR001,184.8,care-team-upload,2024-01-22 09:15:00,28.2,0.95
```

### 2. user_progress_raw.csv / user_progress_filtered.csv
**Purpose**: Weight values at each 30-day interval per user
**Use Case**: Individual trajectory analysis
```csv
user_id,baseline_weight,baseline_date,day_30,day_60,day_90,day_120,day_150,day_180,...
USR001,195.5,2024-01-01,193.2,191.8,189.5,NULL,186.2,184.1
USR002,220.0,2024-01-05,218.5,215.3,212.1,210.0,208.5,NULL
```

### 3. interval_statistics.csv
**Purpose**: Aggregate statistics for box-and-whisker plots
**Use Case**: Population-level comparison visualization
```csv
interval,dataset,mean_change,median_change,std_dev,q1,q3,min,max,n_users,outlier_count
30,raw,-2.3,-2.1,1.8,-3.2,-1.2,-8.5,1.2,450,12
30,filtered,-2.5,-2.4,1.5,-3.4,-1.5,-7.2,0.8,450,8
60,raw,-4.7,-4.5,3.2,-6.1,-2.8,-15.3,2.1,425,15
60,filtered,-5.1,-4.9,2.8,-6.5,-3.2,-13.8,1.5,425,10
```

### 4. user_comparison.csv
**Purpose**: Rank users by difference between raw and filtered results
**Use Case**: Identify users most affected by data quality
```csv
user_id,total_diff_lbs,avg_diff_per_interval,max_single_diff,consistency_score,divergence_rank
USR042,12.3,2.05,4.8,0.73,1
USR089,11.8,1.97,3.9,0.81,2
USR156,10.2,1.70,5.2,0.65,3
```

## Performance Optimizations
- Use database indexes on (user_id, timestamp) for window queries
- Batch load measurements by user chunks (100 users at a time)
- Parallel process visualization generation (4-8 workers)
- Stream CSV writing to avoid memory buildup
- Cache interval calculations in memory during processing

## Error Handling
- Skip users with < 2 measurements
- Log and continue on individual user failures
- Validate date ranges before processing
- Handle timezone inconsistencies
- Graceful degradation if visualization fails

## Risks & Mitigations
**Main Risk**: Memory exhaustion with large user populations
**Mitigation**: Stream processing with configurable batch sizes

**Secondary Risk**: Incorrect interval calculations across timezones
**Mitigation**: Normalize all timestamps to UTC before processing

## Visualization Specifications

### Summary Charts (All Users)

#### 1. Weight Loss Distribution Box Plot
**File**: `reports/visualizations/weight_loss_distribution.png`
**Layout**: Side-by-side box plots for each interval
**Data**: Raw vs Filtered comparison at 30, 60, 90... days
```python
fig, axes = plt.subplots(1, len(intervals), figsize=(20, 8))
for i, interval in enumerate(intervals):
    axes[i].boxplot([raw_data[interval], filtered_data[interval]],
                    labels=['Raw', 'Filtered'])
    axes[i].set_title(f'Day {interval}')
    axes[i].set_ylabel('Weight Change (lbs)')
```

#### 2. Outlier Impact Heatmap
**File**: `reports/visualizations/outlier_impact_heatmap.png`
**Purpose**: Show which intervals have highest filtering impact
**Layout**: Users (y-axis) vs Intervals (x-axis), color = difference magnitude

#### 3. Data Quality vs Weight Loss Correlation
**File**: `reports/visualizations/quality_correlation.png`
**Purpose**: Scatter plot showing relationship between data quality and success
**Axes**: X = Average Quality Score, Y = Weight Loss %, Color = Outlier Count

### Top 200 User Visualizations

#### Individual User Charts (200 files)
**Directory**: `reports/visualizations/top_200_users/`
**Naming**: `user_{user_id}_comparison.png`
**Layout**: 2x2 grid per user
- Top-left: Raw vs Filtered weight timeline
- Top-right: Daily difference plot
- Bottom-left: Quality score over time
- Bottom-right: Interval weight change bars

```python
def generate_user_comparison(user_id, data):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Timeline comparison
    ax1.plot(data['dates'], data['raw'], 'o-', label='Raw', alpha=0.7)
    ax1.plot(data['dates'], data['filtered'], 's-', label='Filtered', alpha=0.7)
    ax1.fill_between(data['dates'], data['raw'], data['filtered'],
                     alpha=0.3, color='red', label='Difference')

    # Difference plot
    differences = data['raw'] - data['filtered']
    ax2.bar(data['dates'], differences, color=['red' if d > 0 else 'blue' for d in differences])
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # Quality scores
    ax3.scatter(data['dates'], data['quality_scores'],
                c=data['quality_scores'], cmap='RdYlGn', vmin=0, vmax=1)

    # Interval changes
    intervals = [30, 60, 90, 120, 150, 180]
    raw_changes = [data[f'raw_day_{i}'] for i in intervals]
    filtered_changes = [data[f'filtered_day_{i}'] for i in intervals]
    x = np.arange(len(intervals))
    width = 0.35
    ax4.bar(x - width/2, raw_changes, width, label='Raw', alpha=0.7)
    ax4.bar(x + width/2, filtered_changes, width, label='Filtered', alpha=0.7)
```

#### Aggregate Top 200 Dashboard
**File**: `reports/visualizations/top_200_summary.png`
**Layout**: 3x3 grid showing:
1. Distribution of divergence scores
2. Average impact per interval
3. Source reliability breakdown
4. BMI category shifts
5. Success rate comparison (>5% loss)
6. Outlier frequency histogram
7. Time to first significant loss
8. Consistency score distribution
9. Data completeness chart

### Interactive HTML Report
**File**: `reports/analysis_report.html`
**Contents**:
- Executive summary with key findings
- Embedded charts with tooltips
- Sortable data tables
- User search/filter functionality
- Export buttons for each chart

## Report.py Structure

```python
#!/usr/bin/env python3
"""
Weight Loss Interval Analysis Report Generator

This script analyzes weight loss progress at 30-day intervals,
comparing raw measurements against Kalman-filtered data to
quantify the impact of outlier rejection.
"""

import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple

from src.database import Database
from src.config import Config
from src.analysis.interval_analyzer import IntervalAnalyzer
from src.analysis.weight_selector import WeightSelector
from src.analysis.statistics import StatisticsCalculator
from src.analysis.csv_generator import CSVGenerator
from src.viz.analysis_charts import AnalysisVisualizer

class WeightLossReport:
    """Main orchestrator for weight loss analysis reporting"""

    def __init__(self, config_path: str, output_dir: str):
        self.config = Config(config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.db = Database(self.config.database_path)
        self.logger = self._setup_logging()

    def generate_report(self,
                       input_csv: str,
                       top_n: int = 200,
                       interval_days: int = 30,
                       window_days: int = 7) -> Dict:
        """
        Generate comprehensive weight loss analysis report

        Args:
            input_csv: Path to input weight measurements CSV
            top_n: Number of top divergent users to visualize
            interval_days: Interval between measurements
            window_days: ±window for selecting measurements

        Returns:
            Dictionary with report metadata and file paths
        """
        self.logger.info(f"Starting report generation for {input_csv}")

        # Phase 1: Load and prepare data
        raw_data = self._load_raw_data(input_csv)
        filtered_data = self._generate_filtered_dataset(raw_data)

        # Phase 2: Calculate intervals for each user
        interval_analyzer = IntervalAnalyzer(interval_days, window_days)
        user_intervals = interval_analyzer.calculate_all_users(raw_data, filtered_data)

        # Phase 3: Generate statistics
        stats_calc = StatisticsCalculator()
        statistics = stats_calc.calculate_all_statistics(user_intervals)

        # Phase 4: Identify top divergent users
        top_users = self._identify_top_divergent_users(statistics, top_n)

        # Phase 5: Generate CSV outputs
        csv_gen = CSVGenerator(self.output_dir / 'data')
        csv_files = csv_gen.generate_all_csvs(
            raw_data, filtered_data, user_intervals, statistics, top_users
        )

        # Phase 6: Generate visualizations
        viz = AnalysisVisualizer(self.output_dir / 'visualizations')
        viz_files = viz.generate_all_visualizations(
            user_intervals, statistics, top_users
        )

        # Phase 7: Generate HTML report
        html_report = self._generate_html_report(
            statistics, csv_files, viz_files
        )

        return {
            'status': 'success',
            'csv_files': csv_files,
            'visualizations': viz_files,
            'html_report': html_report,
            'statistics_summary': statistics['summary']
        }
```

## Out of Scope
- Real-time analysis updates
- Interactive web dashboard
- Modification of existing Kalman processing
- Database schema changes
- API endpoint creation