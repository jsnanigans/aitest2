# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Codebase Overview

High-performance streaming data processor for analyzing time-series weight readings. Processes CSV data line-by-line with minimal memory usage, analyzing each reading in real-time for outlier detection.

## Development Commands

```bash
# Install dependencies (including visualization libs)
uv pip install -r requirements.txt

# First, optimize your CSV file (one-time operation)
uv run python optimize_csv.py ./your-data.csv

# Run the main processor (includes baseline + Kalman + visualization)
uv run python main.py

# Generate comprehensive visualizations from baseline results
uv run python create_visualizations.py

# Configure via config.toml
# source_file = './2025-03-27_optimized.csv'
# baseline_window_days = 7  # Days after signup to establish baseline
# baseline_min_readings = 3  # Min readings for baseline
```

## Architecture & Key Patterns

### True Line-by-Line Streaming
- Reads CSV line-by-line without loading into memory
- Processes each user as their data arrives
- Switches context when new user_id encountered
- Real-time confidence scoring for outlier detection

### Performance Metrics
- **2-3 users/second** processing speed (with full Kalman + visualization)
- **100% Kalman coverage** on all readings
- **100% baseline establishment** (with intelligent fallbacks)
- Minimal memory footprint (only current user in memory)
- Scales to millions of rows effortlessly

### Core Components

#### Main Streaming Processor
- `main.py` - Line-by-line CSV streaming with real-time analysis
- `baseline_processor.py` - Enhanced processor with weight baseline establishment
- `optimize_csv.py` - Pre-sorts CSV by user_id and date for optimal streaming

#### Data Flow
```
Raw CSV → optimize_csv.py → Sorted CSV → main.py → JSON Results
                ↓                           ↓
          (external sort)            (line-by-line stream)
```

### Baseline Establishment (Enhanced)
- **Multiple Baselines**: Re-establishes after 30+ day gaps automatically
- **Intelligent Fallbacks**: Multiple strategies ensure 100% establishment
  - Primary: 7-day window from first reading
  - Fallback: First N readings if window fails
  - Gap-triggered: New baseline after data gaps
- **Retry Mechanism**: BASELINE_PENDING state for sparse data
- **Performance**: 100% of users have baselines established

### Key Processing Features

#### Robust Kalman Filtering (Default)
- **Immediate Processing**: Starts from first reading (no waiting)
- **2D State Tracking**: Weight + trend (kg/day)
- **Extreme Outlier Protection**: Automatically rejects values that would corrupt state
- **Adaptive Noise**: Adjusts based on source trust
- **Gap Handling**: Reinitializes after 30+ day gaps
- **100% Coverage**: All readings filtered from day one

#### Confidence Scoring Algorithm
Each reading gets a confidence score (0.0-1.0):
- **0.95+** - Normal variation (<3% from baseline/recent average)
- **0.90** - Small variation (3-5%)
- **0.75** - Moderate variation (5-10%)
- **0.60** - Significant variation (10-15%)
- **0.45** - Large variation (15-20%)
- **0.30** - Major variation (20-30%)
- **0.15** - Extreme variation (30-50%)
- **0.05** - Extreme outlier (>50% change)

## Key Files

- `main.py` - Unified stream processor with Kalman + baseline + visualization
- `optimize_csv.py` - CSV optimizer using external sorting
- `config.toml` - Configuration settings
- `src/processing/baseline_establishment.py` - Robust baseline with gap detection
- `src/processing/user_processor.py` - User data processing with retry logic
- `src/filters/custom_kalman_filter.py` - 2D Kalman with trend tracking
- `src/visualization/dashboard.py` - Enhanced dashboard with baseline markers

## Configuration

Edit `config.toml` for:
- `source_file` - Input CSV path (use optimized version)
- `min_readings_per_user` - Filter users with insufficient data
- `process_max_users` - Limit users processed (0 for unlimited)
- `baseline_window_days` - Days after signup to establish baseline (default: 7)
- `baseline_min_readings` - Minimum readings for baseline (default: 3)
- `logging.stdout_level` - Set to WARNING to minimize output

## Output Structure

- `output/app.log` - Detailed application logs
- `output/results_YYYYMMDD_HHMMSS.json` - Standard analysis results
- `output/baseline_results_YYYYMMDD_HHMMSS.json` - Enhanced results with visualization data:
  ```json
  {
    "user_id": {
      "total_readings": 42,
      "outliers": 3,
      "average_confidence": 0.875,
      "min_weight": 70.5,
      "max_weight": 75.2,
      "weight_range": 4.7,
      "first_date": "2024-01-01",
      "last_date": "2024-03-15",
      "signup_date": "2024-01-01",
      "signup_weight": 70.5,
      "baseline_weight": 70.8,
      "baseline_readings_count": 5,
      "baseline_confidence": "high",
      "average_deviation_from_baseline": 2.3,
      "time_series": [
        {"date": "2024-01-01", "weight": 70.5, "confidence": 0.95, "source": "internal-questionnaire"}
      ],
      "percentiles": {"p5": 69.8, "p25": 70.2, "p50": 70.8, "p75": 71.3, "p95": 72.1},
      "moving_averages": [70.5, 70.6, 70.7, ...],
      "deviation_from_baseline": [0.0, 0.4, -0.3, ...]
    }
  }
  ```
- `output/visualizations_YYYYMMDD_HHMMSS/` - Visualization outputs:
  - `overview_dashboard.png` - Aggregate statistics for all users
  - `dashboard_<user_id>.png` - Individual user dashboards with 7 charts:
    - Weight trajectory with confidence bands
    - Weight distribution histogram
    - Confidence score distribution
    - Temporal patterns (day of week analysis)
    - Deviation from baseline
    - Outlier analysis pie chart
    - Key statistics summary

## Design Principles

1. **Performance First** - Optimized for speed and memory efficiency
2. **True Streaming** - Process data as it arrives, no buffering
3. **Simple Foundation** - Clean code ready for algorithm enhancements
4. **No Heavy Dependencies** - Pure Python, no pandas/numpy needed

# Super important rules when coding:
- NEVER WRITE ANY COMMENTS THAT ARE NOT SUPER DUPER MEGA IMPORTANT
- Keep the code clean and simple
- Focus on solid foundations over complex features