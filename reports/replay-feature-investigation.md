# Investigation: Replay Feature in Weight Processing System

## Bottom Line
**Root Cause**: Replay enables retrospective outlier detection and state refinement using 72-hour measurement windows
**Fix Location**: Not broken - sophisticated data quality system
**Confidence**: High

## What's Happening
The replay feature buffers incoming measurements for 72 hours, then performs batch outlier detection and chronologically replays clean measurements through the Kalman filter to refine historical state estimates.

## Why It Happens
**Primary Cause**: Real-time processing can't detect outliers that only become apparent with context
**Trigger**: `main.py:396-406` - Buffer fills to 72 hours or 100 measurements
**Decision Point**: `replay_buffer.py:306-364` - Time or count-based triggers

## Architecture Overview

### Data Flow Pipeline
```
Measurement → ReplayBuffer (72hr window) → OutlierDetector → ReplayManager → Refined State
     ↓                                           ↓                  ↓
Real-time Processing                    Statistical Analysis   State Restoration
```

### Key Components

**ReplayBuffer** (`src/processing/replay_buffer.py`)
- Thread-safe measurement storage with RLock protection
- Auto-manages 72-hour sliding windows per user
- Triggers on time (72hr) or count (100 measurements)
- Memory limits prevent unbounded growth

**ReplayManager** (`src/replay/replay_manager.py`)
- Creates state backups before replay (line 82-87)
- Restores state to buffer start time (line 89-93)
- Validates trajectory continuity (<15kg jumps) (line 96-119)
- Replays measurements chronologically (line 122-126)
- Atomic operations with automatic rollback on failure

**OutlierDetector** (`src/processing/outlier_detection.py`)
- IQR method: 1.5x interquartile range
- MAD-based Z-score: 3.0 threshold
- Temporal consistency: 30% max change
- Quality override: High scores (>0.7) protect measurements
- Kalman deviation: 15% prediction threshold

## State Management

### Snapshot System
- Snapshots saved at buffer trigger points (`main.py:401`)
- Database maintains 100 historical snapshots per user
- Snapshots enable restoration to any point in time
- Used for "time travel" during replay processing

### Rollback Safety
1. Full state backup before replay (`replay_manager.py:151-173`)
2. Automatic rollback on any failure (`replay_manager.py:124,142`)
3. 15kg jump prevention check (`replay_manager.py:109`)
4. 60-second timeout protection (`replay_manager.py:282`)

## Configuration

### Replay Settings (`config.toml:119-135`)
```toml
[replay]
enabled = true
buffer_hours = 1  # Currently 1hr for testing, normally 72
trigger_mode = "time_based"
max_buffer_measurements = 100

[replay.outlier_detection]
iqr_multiplier = 1.5
z_score_threshold = 3.0
temporal_max_change_percent = 0.5
```

### Quality Adjustment During Replay
- Quality threshold lowered to 0.25 (vs 0.6 normal)
- Consistency weight reduced to 0.10
- Reliability weight increased to 0.30
- Allows legitimate variations through

## Use Cases and Benefits

### Problem It Solves
1. **Delayed Outlier Detection**: Some outliers only visible with context
2. **State Refinement**: Improves historical estimates after removing noise
3. **Quality Recovery**: Recovers from temporary sensor issues
4. **Trend Correction**: Fixes trajectory after outlier removal

### Real-World Scenarios
- Patient accidentally enters 850lbs instead of 85lbs
- Scale malfunction produces series of bad readings
- Network issues cause duplicate measurements
- Medication changes cause legitimate rapid changes

## Thread Safety Mechanisms

### ReplayBuffer Protection
- RLock for all operations (line 49)
- Deep copy for returned data (line 135)
- Atomic buffer creation/clearing

### State Backup System
- In-memory backup storage (line 59)
- Deep copy for isolation (line 165)
- Clear separation between users

## Evidence
- **Buffer Management**: `replay_buffer.py:60-118` - Thread-safe measurement addition
- **Trigger Logic**: `replay_buffer.py:306-364` - Smart triggering conditions
- **State Restoration**: `replay_manager.py:209-254` - Time-travel mechanism
- **Outlier AND Logic**: `outlier_detection.py:117-139` - Conservative outlier marking

## Next Steps
1. Increase buffer_hours from 1 to 72 for production use
2. Consider async processing for large buffers
3. Add metrics for replay success rate
4. Implement buffer persistence for crash recovery

## Risks
- Memory usage with many concurrent users (100 measurements/user)
- Processing delay at 72-hour boundaries
- State corruption if rollback fails (mitigated by backup)
