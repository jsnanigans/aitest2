# Weight Stream Processor

A high-performance, production-ready weight measurement processing system with Kalman filtering and intelligent outlier detection.

## Performance

- **Processing Speed**: 0.21ms per measurement (14x faster than requirements)
- **Code Size**: 3,472 lines (40% reduction from original)
- **Architecture**: Clean, linear processing pipeline

## Features

- **Kalman Filtering**: Adaptive noise-based filtering for smooth weight tracking
- **Source-Specific Processing**: Intelligent handling based on data source reliability
- **BMI Detection**: Automatic detection and conversion of BMI values
- **Physiological Validation**: Comprehensive validation against human limits
- **Gap Detection**: Automatic reset after extended measurement gaps
- **Structured Logging**: Production-ready logging and metrics

## Installation

```bash
# Install dependencies
uv pip install -r requirements.txt
```

## Usage

### Basic Processing

```bash
# Process a CSV file
uv run python main.py data/weights.csv

# With configuration
uv run python main.py data/weights.csv --config config.toml

# Generate visualizations
uv run python main.py data/weights.csv --visualize
```

### Performance Testing

```bash
# Run performance benchmark
uv run python scripts/measure_performance.py
```

### Running Tests

```bash
# Run all tests
uv run python -m pytest tests/

# Run specific test
uv run python tests/test_processor.py
```

## Architecture

```
main.py                 # Entry point and CSV processing
├── src/
│   ├── processor.py    # Core processing pipeline
│   ├── kalman.py       # Kalman filter implementation
│   ├── validation.py   # Validation and data quality
│   ├── database.py     # State management
│   ├── constants.py    # Configuration constants
│   ├── visualization.py # Data visualization
│   └── logging_utils.py # Structured logging
└── tests/              # Test suite
```

## Configuration

The system uses a combination of hard-coded safety limits (in `constants.py`) and configurable parameters (in `config.toml`).

### Key Parameters

- **Kalman Filter**: Optimized parameters for weight tracking
- **Source Profiles**: Reliability and noise characteristics per source
- **Physiological Limits**: Safety boundaries for human weight
- **Processing Thresholds**: Adaptive thresholds based on time gaps

## Data Sources

The system intelligently handles different data sources with varying reliability:

- `patient-upload`: Most reliable (noise multiplier: 0.7)
- `care-team-upload`: Excellent reliability (noise multiplier: 0.5)
- `questionnaire`: Good reliability (noise multiplier: 0.8)
- `patient-device`: Moderate reliability (noise multiplier: 1.0)
- `connectivehealth.io`: Lower reliability (noise multiplier: 1.5)
- `iglucose.com`: Requires extra validation (noise multiplier: 3.0)

## Performance Metrics

Current performance (100 measurements test):
- Average: 0.21ms
- Median: 0.21ms
- Min: 0.20ms
- Max: 0.25ms
- Target: <3ms ✅

## Development

### Code Style
- Python 3.11+
- Type hints optional (pyright mode: off)
- No comments unless critical
- Single-purpose functions
- Clear module boundaries

### Testing
- Unit tests for each module
- Integration tests for full pipeline
- Performance benchmarks
- Golden dataset regression tests

## License

[Your License Here]