# Output Structure Documentation

## Overview

The weight stream processor creates a well-organized output structure with separate folders for different types of outputs.

## Directory Structure

```
output/                          # Main output directory (configurable)
├── logs/
│   └── app.log                  # Detailed application logs (DEBUG level)
│
├── results/
│   ├── user_[id].json           # Individual user processing results
│   ├── summary_[timestamp].json # Processing summary statistics
│   └── all_results_[timestamp].json # Combined results for all users
│
└── visualizations/              # Configurable via visualization.output_dir
    ├── dashboard_[user1]_[timestamp].png
    ├── dashboard_[user2]_[timestamp].png
    └── ...                      # Up to max_users dashboards
```

## Configuration

### Output Directory Settings

```toml
# Main output directory
[output]
directory = "output"  # Base directory for all outputs

# Visualization output
[visualization]
enabled = true
max_users = 10  # Limit number of dashboards created
output_dir = "output/visualizations"  # Can be customized
```

## Output Files

### 1. Logs (`output/logs/`)

- **app.log**: Complete application log with timestamps
  - DEBUG level to file (everything)
  - INFO level to console (important messages)
  - Colored console output for better readability

### 2. Results (`output/results/`)

#### Individual User Files
- **Format**: `user_[user_id].json`
- **Contents**: Complete processing results for one user
  ```json
  {
    "user_id": "abc123",
    "initialized": true,
    "baseline": {...},
    "current_state": {...},
    "stats": {...},
    "time_series": [...]
  }
  ```

#### Summary File
- **Format**: `summary_YYYYMMDD_HHMMSS.json`
- **Contents**: Processing statistics
  ```json
  {
    "processing_stats": {
      "total_rows": 100000,
      "total_users": 500,
      "processing_time_seconds": 45.2
    },
    "data_quality": {
      "acceptance_rate": 0.87,
      "outlier_types": {...}
    }
  }
  ```

#### All Results File
- **Format**: `all_results_YYYYMMDD_HHMMSS.json`
- **Contents**: Combined results for all users in one file

### 3. Visualizations (`output/visualizations/`)

- **Format**: `dashboard_[user_id]_YYYYMMDD_HHMMSS.png`
- **Contents**: 7-panel dashboard per user
  - Weight trajectory with Kalman filtering
  - Trend analysis
  - Innovation distribution
  - Confidence levels
  - Outlier breakdown
  - Weekly patterns
  - Statistics summary
- **Limit**: Only creates dashboards for first `max_users` users

## Features

### Automatic Creation
- All directories are created automatically
- Missing parent directories are created with `parents=True`
- Existing directories are not overwritten (`exist_ok=True`)

### Configurability
- Base output directory configurable via `[output].directory`
- Visualization directory configurable via `[visualization].output_dir`
- Can use absolute or relative paths

### Organization Benefits
- **Separation**: Logs, results, and visualizations are separated
- **Timestamps**: Files include timestamps to avoid overwrites
- **Clean**: Each run creates new timestamped files
- **Scalable**: Structure works for 1 user or 10,000 users

## Usage Examples

### Default Configuration
```toml
[output]
directory = "output"

[visualization]
output_dir = "output/visualizations"
```

Creates:
```
output/
├── logs/
├── results/
└── visualizations/
```

### Custom Paths
```toml
[output]
directory = "/data/weight_analysis"

[visualization]
output_dir = "/data/weight_analysis/charts"
```

Creates:
```
/data/weight_analysis/
├── logs/
├── results/
└── charts/
```

### Separate Visualization Location
```toml
[output]
directory = "output"

[visualization]
output_dir = "/shared/dashboards"
```

Creates:
```
output/
├── logs/
└── results/

/shared/dashboards/
└── [visualization files]
```

## Verification

Run the test script to verify structure:
```bash
python test_output_structure.py
```

This will:
1. Create test directories
2. Verify all paths work correctly
3. Test visualization settings
4. Show the created structure
5. Clean up test files

## Best Practices

1. **Regular Cleanup**: Periodically archive old results
2. **Disk Space**: Monitor visualization folder (PNG files can be large)
3. **Permissions**: Ensure write permissions for output directories
4. **Backup**: Results directory contains all processing data

## Troubleshooting

### Missing Directories
- Check write permissions
- Verify paths in config.toml
- Look for error messages in console

### Visualization Not Created
- Check `[visualization].enabled = true`
- Verify `max_users` is > 0
- Check for matplotlib errors in logs

### Wrong Output Location
- Check config.toml paths
- Use absolute paths if relative paths cause issues
- Verify working directory is correct