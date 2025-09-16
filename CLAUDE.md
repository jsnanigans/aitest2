# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

All Python commands must be run with `uv`:

```bash
# Main processing
uv run python main.py data/weights.csv                    # Process CSV file
uv run python main.py data/weights.csv --config config.toml  # With custom config
uv run python main.py data/weights.csv --visualize        # Generate visualizations

# Testing
uv run python -m pytest tests/                            # Run all tests
uv run python -m pytest tests/test_processor.py -xvs      # Run specific test with verbose output
uv run python tests/test_processor.py                     # Run individual test file directly

# Performance
uv run python scripts/measure_performance.py              # Run performance benchmark

# Utilities
uv run python scripts/batch_visualize.py                  # Batch visualization generation
uv run python scripts/generate_index.py                   # Generate visualization index
```

## Architecture Overview

This is a **weight measurement processing system** with Kalman filtering and intelligent outlier detection. The system processes weight measurements from various sources with different reliability levels.

### Core Processing Pipeline

```
main.py → processor.py → kalman.py → quality_scorer.py → database.py
                ↓
         validation.py → outlier_detection.py
```

### Key Components

**src/processor.py**: Main processing pipeline that orchestrates all components
- Handles measurement flow through validation, Kalman filtering, and quality scoring
- Manages state persistence via database

**src/kalman_adaptive.py**: Adaptive Kalman filter implementation
- Three reset types: INITIAL (first measurements), HARD (after 30+ day gaps), SOFT (manual entries)
- Adaptive noise parameters that decay over time after resets
- Source-specific noise multipliers based on data reliability

**src/quality_scorer.py**: Multi-factor quality assessment
- Safety score (physiological limits)
- Plausibility score (BMI detection, trend analysis)
- Consistency score (change rate validation)
- Reliability score (source-based)

**src/outlier_detection.py**: Statistical outlier detection with quality override
- Uses MAD-based detection with dynamic thresholds
- Can be overridden by high quality scores
- Extreme threshold: 15% deviation from Kalman prediction

**src/reset_manager.py**: Manages Kalman filter resets
- Detects gaps and triggers appropriate reset types
- Tracks adaptation state and decay

### Configuration System

**config.toml**: Runtime parameters (thresholds, Kalman parameters, reset settings)
**src/constants.py**: Hard-coded safety limits and physiological boundaries

### Data Source Reliability

Sources are ranked by noise multiplier (lower = more reliable):
- `care-team-upload`: 0.5 (most reliable)
- `patient-upload`: 0.7
- `questionnaire`: 0.8
- `patient-device`: 1.0
- `connectivehealth.io`: 1.5
- `iglucose.com`: 3.0 (requires extra validation)

## Key Implementation Details

### Adaptive Kalman System
The Kalman filter adapts after resets by temporarily increasing process and observation noise, allowing it to quickly converge to new weight ranges. The adaptation decays over time or measurements.

### Quality Override System
High-quality measurements (score > 0.8) can override outlier detection, allowing legitimate weight changes through while still filtering noise.

### State Management
User states are persisted in SQLite database with full Kalman state, including adaptation parameters and buffer history.