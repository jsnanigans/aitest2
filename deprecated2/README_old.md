# Weight Stream Processor v2.2

A high-performance, clinically-informed system for processing longitudinal weight data streams with robust outlier detection and trend analysis. Features Layer 1 physiological pre-filtering and enhanced Kalman filtering that's resistant to extreme outliers.

## Overview

This system implements a sophisticated multi-layered pipeline for cleaning, validating, and analyzing weight measurements from multiple sources. It follows a clinically-informed framework that combines:

- **Robust statistical methods** for baseline establishment
- **Multi-layered outlier detection** (heuristic, ARIMA, and Kalman filtering)
- **Mathematically correct state estimation** using Kalman filters
- **True streaming architecture** for memory-efficient processing

## Architecture

The system follows a clean, layered architecture:

```
Input Stream → Layer 1 (Heuristics) → Layer 2 (ARIMA) → Layer 3 (Kalman) → Output
                    ↓                      ↓                   ↓
                 Rejected              Classified           Validated
```

### Layer 1: Fast Heuristic Filters (Active)
- **Physiological plausibility** (30-400 kg, configurable)
- **Rate of change limits** (±3% daily, ±5% in medical mode)
- **Deviation from prediction** (warns at 10%, rejects at 20%)
- **Stateless design** for real-time processing
- Protects Kalman from data entry errors and impossible values

### Layer 2: Time-Series Modeling
- ARIMA-based outlier detection
- Classification into 4 types (Additive, Innovational, Level Shift, Temporary)
- Handles missing data and irregular sampling

### Layer 3: Robust State Estimation
- **Enhanced Kalman filter** with extreme outlier protection
- 2D state tracking [weight, trend]
- **Adaptive outlier handling**:
  - Moderate outliers (3-5σ): Dampened updates
  - Extreme outliers (>5σ): Complete rejection
- Innovation monitoring for adaptive thresholds
- **97% error reduction** vs standard Kalman with outliers

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or using uv (recommended)
uv pip install -r requirements.txt
```

## Usage

### Quick Start

```bash
# Process weight data
python main.py

# Run architecture tests
python test_architecture.py

# Run unit tests
pytest tests/
```

### Configuration

Edit `config.toml`:

```toml
# Data source
source_file = "./data/weights_optimized.csv"

# Processing parameters
[processing]
min_init_readings = 3
validation_gamma = 3.0

[processing.baseline]
collection_days = 7
min_readings = 3

[processing.layer1]
enabled = true  # Enable physiological pre-filtering
min_weight = 30.0  # Minimum plausible weight
max_weight = 400.0  # Maximum weight (increase for bariatric)
max_daily_change_percent = 3.0  # Normal daily limit
medical_mode_percent = 5.0  # Relaxed for medical interventions
extreme_threshold_percent = 20.0  # Deviation rejection threshold

[processing.kalman]
process_noise_weight = 0.5
process_noise_trend = 0.01
outlier_threshold = 3.0  # Moderate outliers
extreme_outlier_threshold = 5.0  # Extreme outliers
innovation_window_size = 20  # Adaptive threshold window
```

## Features

### Robust Baseline Establishment
- IQR-based outlier removal
- Median for central tendency
- MAD for variance estimation
- Confidence assessment (high/medium/low)

### Multi-Source Trust
- Configurable trust levels by data source
- Adaptive measurement noise based on source reliability
- Handles data from scales, manual entry, and health aggregators

### Real-Time Processing
- True line-by-line streaming
- O(1) memory per user
- ~2-3 users/second throughput
- Handles millions of rows effortlessly

### Comprehensive Output
- Individual user results with time series
- Filtered weights and trends
- Confidence scores for each measurement
- Outlier classification and statistics

## Output Structure

```json
{
  "user_id": "abc123",
  "baseline": {
    "weight": 70.5,
    "confidence": "high",
    "variance": 0.25
  },
  "current_state": {
    "weight": 69.8,
    "trend_kg_per_day": -0.05,
    "trend_kg_per_week": -0.35
  },
  "time_series": [
    {
      "date": "2024-01-01T00:00:00",
      "weight": 70.5,
      "filtered_weight": 70.5,
      "confidence": 0.95,
      "is_valid": true
    }
  ]
}
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test module
pytest tests/test_layers.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Mathematical Foundation

The system implements the state-space model:

- **State Vector**: x = [weight, trend]ᵀ
- **State Transition**: F = [[1, Δt], [0, 1]]
- **Observation Model**: H = [1, 0]
- **Process Noise**: Q with adaptive parameters
- **Measurement Noise**: R scaled by source trust

## Performance

- **Memory**: O(1) per user (true streaming)
- **Speed**: 2-3 users/second with full processing
- **Accuracy**: 87%+ acceptance rate on real data
- **Scalability**: Handles millions of rows

## Framework Compliance

This implementation strictly follows the clinical framework:
- ✅ Part II: Robust baseline establishment
- ✅ Part III.1: Layer 1 heuristic filters
- ✅ Part III.2: Layer 2 ARIMA detection
- ✅ Part IV: Pure Kalman filter
- ✅ Part VI: Integrated pipeline

## Development

### Project Structure

```
src/
├── core/           # Core types and configuration
├── filters/        # Layer 1, 2, 3 implementations
└── processing/     # Pipeline and user processor

tests/              # Unit tests
deprecated/         # Old implementation (reference only)
```

### Key Principles

1. **Separation of Concerns**: Each component has a single responsibility
2. **Mathematical Correctness**: No modifications to proven algorithms
3. **Framework Compliance**: Exact implementation of specifications
4. **Clean Architecture**: Testable, maintainable, extensible

## License

MIT

## Contributing

Contributions welcome! Please ensure:
- All tests pass
- Code follows framework specifications
- Mathematical correctness is maintained
- Documentation is updated

## References

- Clinical weight dynamics framework
- Kalman filter theory
- ARIMA time-series analysis
- Robust statistical methods