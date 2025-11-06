# Weight Processor Library

Core Python library for processing weight measurements using adaptive Kalman filtering and multi-component quality scoring. This library is infrastructure-agnostic and can be used in various contexts (AWS Lambda, local applications, batch processing, etc.).

## Overview

This library provides robust weight measurement processing with:

- **Adaptive Kalman Filtering**: Tracks weight trends and velocity with intelligent noise adaptation
- **Multi-Component Quality Scoring**: Comprehensive quality assessment including:
  - Plausibility checks (physiological limits)
  - Temporal consistency validation
  - Statistical validation
  - Source reliability weighting
  - Kalman fit analysis
- **Smart Reset Management**: Handles significant weight changes and state resets
- **State Persistence**: Abstract storage interface with DynamoDB implementation
- **Buffered Replay**: Reprocesses measurements after state changes
- **Circuit Breaker**: Prevents cascading failures in production systems

## Installation

### Basic Installation

```bash
pip install -e .
```

### With AWS Support (DynamoDB)

```bash
pip install -e ".[aws]"
```

### Development Installation

```bash
pip install -e ".[dev]"
```

### All Features

```bash
pip install -e ".[all]"
```

## Quick Start

### Using In-Memory Storage (Recommended for Testing)

```python
from weight_processor_lib.core.processing.processor import process_measurement
from weight_processor_lib.core.database import get_state_db
from datetime import datetime, timezone

# Use in-memory storage (no persistence, great for testing)
state_store = get_state_db(backend="memory")

# Process a measurement
result = process_measurement(
    user_id="user123",
    weight=75.5,
    timestamp=datetime.now(timezone.utc),
    source="connected_scale",
    config={},
    db=state_store
)

# Access results
if result["accepted"]:
    print(f"Filtered weight: {result['filtered_weight']:.2f} kg")
    print(f"Quality score: {result['quality_score']:.2f}")
```

### Using DynamoDB (Production)

```python
from weight_processor_lib.core.processing.processor import process_measurement
from weight_processor_lib.core.database import get_state_db
from datetime import datetime, timezone

# Use DynamoDB (requires boto3)
state_store = get_state_db(backend="dynamodb")

# Or use environment variable:
# export DB_BACKEND=dynamodb
# state_store = get_state_db()

# Process a measurement
result = process_measurement(
    user_id="user123",
    weight=75.5,
    timestamp=datetime.now(timezone.utc),
    source="connected_scale",
    config={},
    db=state_store
)
```

## Architecture

### Core Components

```
weight_processor_lib/
├── core/
│   ├── processing/          # Core processing logic
│   │   ├── processor.py           # Main processing orchestrator
│   │   ├── kalman.py              # Kalman filter implementation
│   │   ├── unified_quality_scorer.py  # Quality scoring system
│   │   ├── validation.py          # Input validation
│   │   ├── reset_manager.py       # Reset logic
│   │   └── circuit_breaker.py     # Failure protection
│   ├── database/            # Storage abstraction
│   │   ├── base.py               # Abstract StateStore interface
│   │   ├── memory_store.py       # In-memory implementation
│   │   └── dynamodb_store.py     # DynamoDB implementation
│   ├── constants.py         # Configuration constants
│   ├── exceptions.py        # Custom exceptions
│   └── utils.py             # Shared utilities
```

### Key Abstractions

#### StateStore Interface

The `StateStore` abstract base class allows plugging in different storage backends:

```python
from weight_processor_lib.core.database.base import StateStore

class MyCustomStore(StateStore):
    def get_state(self, user_id: str) -> Optional[Dict[str, Any]]:
        # Your implementation
        pass

    def save_state(self, user_id: str, state: Dict[str, Any]) -> None:
        # Your implementation
        pass
```

#### Process Measurement Function

Main entry point for processing:

```python
def process_measurement(
    user_id: str,
    measured_at: datetime,
    raw_weight_kg: float,
    source: str,
    db: StateStore,
    device_id: Optional[str] = None,
    user_height_m: Optional[float] = None,
    quality_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Process a single weight measurement.

    Returns dict with:
        - accepted: bool
        - processed_weight: float
        - quality_score: float
        - reset_occurred: bool
        - And many more fields...
    """
```

## Quality Scoring

The system uses a sophisticated multi-component quality scoring system:

### Components

1. **Plausibility Score** (30% weight)
   - Checks against physiological limits (40-300 kg)
   - Validates BMI if height provided
   - Source reliability weighting

2. **Temporal Consistency** (25% weight)
   - Checks for realistic rate of change
   - Accounts for time gaps
   - Adaptive thresholds

3. **Statistical Validation** (20% weight)
   - Z-score analysis
   - Outlier detection
   - Historical distribution fitting

4. **Kalman Fit** (15% weight)
   - Measures prediction vs actual
   - Chi-squared goodness of fit
   - Adaptive based on measurement frequency

5. **Trend Alignment** (10% weight)
   - Consistency with recent trends
   - Velocity correlation
   - Directional alignment

### Quality Thresholds

- **≥ 0.7**: Excellent quality (always accepted)
- **0.5-0.7**: Good quality (accepted with caution)
- **< 0.5**: Poor quality (may be rejected)

## Kalman Filtering

### Adaptive Noise Model

The Kalman filter adapts its noise parameters based on:
- Measurement frequency (more frequent = lower noise)
- Time gaps (longer gaps = higher noise)
- Source reliability (scales vary by reliability)
- Recent measurement variance

### State Vector

```
x = [weight, velocity]
```

Where:
- `weight`: Current estimated true weight (kg)
- `velocity`: Rate of change (kg/day)

### Reset Conditions

The system automatically resets the Kalman filter when:
1. **Large weight change**: > 5 kg sudden change
2. **Trend reversal**: Significant direction change
3. **Long time gap**: > 90 days without measurements
4. **Manual reset**: Via API request

## Testing

### Run All Tests

```bash
pytest tests/
```

### Run with Coverage

```bash
pytest tests/ --cov=src/weight_processor_lib --cov-report=html
```

### Run Specific Test Module

```bash
pytest tests/processing/test_processor.py -v
```

## Configuration

Core constants are defined in `constants.py`:

```python
KALMAN_DEFAULTS = {
    "initial_weight_variance": 4.0,
    "initial_velocity_variance": 0.01,
    "base_process_noise": 0.01,
    "base_measurement_noise": 0.25,
}

PHYSIOLOGICAL_LIMITS = {
    "min_weight_kg": 40.0,
    "max_weight_kg": 300.0,
    "max_daily_change_kg": 5.0,
}
```

## Storage Backends

The library provides flexible storage backends through the `StateStore` abstract interface.

### InMemoryStore (Testing & Development)

Fast, thread-safe in-memory storage. Perfect for testing, development, and prototyping. **No persistence** - data is lost when the process ends.

```python
from weight_processor_lib.core.database import InMemoryStore, get_state_db

# Option 1: Direct instantiation
db = InMemoryStore()

# Option 2: Via get_state_db
db = get_state_db(backend="memory")

# Option 3: Via environment variable
# export DB_BACKEND=memory
db = get_state_db()
```

**Features**:
- Thread-safe operations
- Zero external dependencies
- Fast performance
- Snapshot support for replay functionality
- Useful helper methods: `clear_all()`, `list_users()`, `get_snapshot_count()`

**Example usage in tests**:

```python
import pytest
from weight_processor_lib.core.database import InMemoryStore, set_state_db

@pytest.fixture
def db():
    """Provide clean database for each test."""
    store = InMemoryStore()
    set_state_db(store)
    yield store
    store.clear_all()

def test_process_measurement(db):
    result = process_measurement(
        user_id="test_user",
        weight=75.5,
        timestamp=datetime.now(timezone.utc),
        source="test",
        config={},
        db=db
    )
    assert result["accepted"]
```

### DynamoDB (Production)

Persistent storage using AWS DynamoDB. Ideal for production deployments.

```python
from weight_processor_lib.core.database import get_state_db

# Option 1: Explicit backend
db = get_state_db(backend="dynamodb")

# Option 2: Default (DynamoDB)
# export DB_BACKEND=dynamodb  # or omit to use default
db = get_state_db()
```

**Environment Variables**:
- `DYNAMODB_ENDPOINT`: DynamoDB endpoint (for local development, e.g., `http://localhost:8000`)
- `DYNAMODB_TABLE_NAME`: Table name (default: `weight-processor-state`)
- `AWS_REGION`: AWS region (default: `us-east-1`)
- `AWS_ACCESS_KEY_ID`: AWS credentials (or use IAM roles)
- `AWS_SECRET_ACCESS_KEY`: AWS credentials (or use IAM roles)

**Local Development with DynamoDB Local**:

```bash
# Start DynamoDB Local
docker run -p 8000:8000 amazon/dynamodb-local

# Set environment
export DYNAMODB_ENDPOINT=http://localhost:8000
export DB_BACKEND=dynamodb

# Use in code
db = get_state_db()
```

### Custom Backend

Implement the `StateStore` interface for custom storage:

```python
from weight_processor_lib.core.database.base import StateStore
from typing import Dict, Any, Optional
from datetime import datetime

class PostgresStore(StateStore):
    def __init__(self, connection_string: str):
        self.conn = psycopg2.connect(connection_string)

    def get_state(self, user_id: str) -> Optional[Dict[str, Any]]:
        # Query PostgreSQL
        cursor = self.conn.cursor()
        cursor.execute("SELECT state FROM weight_processor WHERE user_id = %s", (user_id,))
        result = cursor.fetchone()
        return json.loads(result[0]) if result else None

    def save_state(self, user_id: str, state: Dict[str, Any]) -> bool:
        # Save to PostgreSQL
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO weight_processor (user_id, state) VALUES (%s, %s) "
            "ON CONFLICT (user_id) DO UPDATE SET state = EXCLUDED.state",
            (user_id, json.dumps(state))
        )
        self.conn.commit()
        return True

    def delete_state(self, user_id: str) -> bool:
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM weight_processor WHERE user_id = %s", (user_id,))
        self.conn.commit()
        return cursor.rowcount > 0

    def create_initial_state(self) -> Dict[str, Any]:
        return {
            "kalman_params": None,
            "last_state": None,
            # ... other fields
        }

    # Implement remaining StateStore methods...

# Use custom backend
from weight_processor_lib.core.database import set_state_db

db = PostgresStore("postgresql://localhost/mydb")
set_state_db(db)
```

### Choosing a Backend

| Backend | Use Case | Pros | Cons |
|---------|----------|------|------|
| **InMemoryStore** | Testing, development | Fast, simple, no setup | No persistence |
| **DynamoDBStore** | Production AWS | Scalable, managed, HA | AWS-specific, cost |
| **Custom** | Special requirements | Full control | Implementation effort |

## Error Handling

### Circuit Breaker

Protects against cascading failures:

```python
from weight_processor_lib.core.processing.circuit_breaker import CircuitBreaker, CircuitOpenError

circuit_breaker = CircuitBreaker(
    failure_threshold=5,
    recovery_timeout=60,
    expected_exception=ValueError
)

try:
    with circuit_breaker:
        result = process_measurement(...)
except CircuitOpenError:
    # Circuit is open, service is degraded
    pass
```

### Custom Exceptions

```python
from weight_processor_lib.core.exceptions import (
    ValidationError,
    ProcessingError,
    StateError,
)
```

## Best Practices

### 1. Always Use Timezone-Aware Datetimes

```python
from datetime import datetime, timezone

measured_at = datetime.now(timezone.utc)  # Good
measured_at = datetime.now()  # Bad - ambiguous timezone
```

### 2. Handle Reset Events

```python
result = process_measurement(...)

if result["reset_occurred"]:
    # Log reset event
    # Notify user if appropriate
    # Update downstream systems
    pass
```

### 3. Monitor Quality Scores

```python
if result["quality_score"] < 0.5:
    # Flag for review
    # Request user confirmation
    # Log for analysis
    pass
```

### 4. Implement Proper Error Handling

```python
from weight_processor_lib.core.exceptions import ValidationError

try:
    result = process_measurement(...)
except ValidationError as e:
    # Handle validation errors (bad input)
    logger.error(f"Validation error: {e}")
except Exception as e:
    # Handle processing errors
    logger.error(f"Processing error: {e}")
```

## Performance Considerations

- **State Loading**: Cache state for batch processing to reduce database calls
- **Numpy Arrays**: State vectors use numpy for efficient computation
- **Measurement Frequency**: Higher frequency = better filter performance
- **Time Gaps**: Long gaps increase uncertainty and may trigger resets

## Development

### Code Style

```bash
# Format code
black src/ tests/

# Lint
ruff check src/ tests/

# Type check
mypy src/
```

### Running Tests

```bash
# All tests
pytest

# Specific module
pytest tests/processing/test_processor.py

# With coverage
pytest --cov=src/weight_processor_lib --cov-report=html
```

## Integration Examples

### AWS Lambda

See `be_implementation_service/` for a complete AWS Lambda implementation using this library.

### Batch Processing

```python
from weight_processor_lib.core.processing.processor import process_measurement
from weight_processor_lib.core.database import get_state_db

db = get_state_db()

for measurement in measurements:
    result = process_measurement(
        user_id=measurement["user_id"],
        measured_at=measurement["timestamp"],
        raw_weight_kg=measurement["weight"],
        source=measurement["source"],
        db=db
    )
    # Store results
```

### Real-Time Stream Processing

```python
# Process measurements as they arrive
def handle_weight_event(event):
    result = process_measurement(
        user_id=event["user_id"],
        measured_at=event["timestamp"],
        raw_weight_kg=event["weight"],
        source=event["source"],
        db=state_store
    )

    if result["accepted"]:
        emit_processed_weight(result)
```

## License

MIT

## Contributing

See main repository for contribution guidelines.
