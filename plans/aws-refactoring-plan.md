# AWS Migration Refactoring Plan

## Executive Summary

This document outlines the refactoring strategy to transform the current batch-processing codebase into an AWS-ready microservice architecture. The refactoring focuses on maintaining backward compatibility while introducing abstractions necessary for cloud deployment.

## Current Architecture Analysis

### Strengths (To Preserve)
- **Core Processing Logic**: `processor.py`, `kalman.py`, `unified_quality_scorer.py` are well-isolated
- **Modular Design**: Clear separation between processing, database, and replay modules
- **Configuration System**: Flexible config structure that can adapt to environment variables
- **State Management**: Clean state interface in `ProcessorStateDB`

### Challenges (To Address)
- **Tight CSV Coupling**: Main entry point assumes file-based batch processing
- **In-Memory Database**: State storage not suitable for distributed systems
- **Visualization Dependencies**: Processing mixed with visualization generation
- **Static Configuration**: Config loaded from TOML files rather than environment
- **Synchronous Design**: No async/await patterns for I/O operations
- **Missing API Layer**: No request/response models or error handling

## Refactoring Strategy

### Phase 1: Core Abstractions (Week 1)

#### 1.1 Database Abstraction Layer
Create an abstract base class that both in-memory and DynamoDB implementations can inherit from.

**File**: `src/database/base.py`
```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import datetime

class StateStore(ABC):
    """Abstract base class for state storage."""

    @abstractmethod
    def get_state(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve user state."""
        pass

    @abstractmethod
    def save_state(self, user_id: str, state: Dict[str, Any]) -> bool:
        """Save user state."""
        pass

    @abstractmethod
    def delete_state(self, user_id: str) -> bool:
        """Delete user state."""
        pass

    @abstractmethod
    def create_initial_state(self) -> Dict[str, Any]:
        """Create initial state structure."""
        pass

    @abstractmethod
    def save_state_snapshot(self, user_id: str, timestamp: datetime) -> bool:
        """Save state snapshot."""
        pass

    @abstractmethod
    def get_snapshot(self, user_id: str, timestamp: datetime) -> Optional[Dict[str, Any]]:
        """Get nearest snapshot before timestamp."""
        pass
```

**Refactoring Steps**:
1. Create `StateStore` abstract base class
2. Rename `ProcessorStateDB` to `InMemoryStateStore`
3. Make `InMemoryStateStore` inherit from `StateStore`
4. Create `DynamoDBStateStore` implementing `StateStore`
5. Add factory function `get_state_store(backend='memory')`

#### 1.2 Configuration Abstraction
Support both file-based and environment-based configuration.

**File**: `src/config/config_manager.py`
```python
import os
import tomllib
from typing import Dict, Any, Optional

class ConfigManager:
    """Manages configuration from multiple sources."""

    @staticmethod
    def load_config(source: str = 'auto') -> Dict[str, Any]:
        """
        Load configuration from file or environment.

        Args:
            source: 'file', 'env', or 'auto'
        """
        if source == 'env' or (source == 'auto' and os.getenv('AWS_LAMBDA_FUNCTION_NAME')):
            return ConfigManager._load_from_env()
        else:
            return ConfigManager._load_from_file()

    @staticmethod
    def _load_from_env() -> Dict[str, Any]:
        """Load configuration from environment variables."""
        return {
            'kalman': {
                'enabled': os.getenv('KALMAN_ENABLED', 'true').lower() == 'true',
                'adaptive': os.getenv('KALMAN_ADAPTIVE', 'true').lower() == 'true',
                'process_noise': float(os.getenv('KALMAN_PROCESS_NOISE', '1.0')),
                'observation_noise': float(os.getenv('KALMAN_OBS_NOISE', '4.0'))
            },
            'quality_scoring': {
                'enabled': os.getenv('QUALITY_SCORING_ENABLED', 'true').lower() == 'true',
                'weights': {
                    'kalman': float(os.getenv('QS_WEIGHT_KALMAN', '0.4')),
                    'temporal': float(os.getenv('QS_WEIGHT_TEMPORAL', '0.3')),
                    'source': float(os.getenv('QS_WEIGHT_SOURCE', '0.3'))
                }
            },
            'database': {
                'backend': os.getenv('DB_BACKEND', 'memory'),
                'table_name': os.getenv('DB_TABLE_NAME', 'weight-processor-state')
            }
        }

    @staticmethod
    def _load_from_file(path: str = 'config.toml') -> Dict[str, Any]:
        """Load configuration from TOML file."""
        with open(path, 'rb') as f:
            return tomllib.load(f)
```

### Phase 2: API Layer (Week 1-2)

#### 2.1 Request/Response Models
Create Pydantic models for API contracts.

**File**: `src/api/models.py`
```python
from pydantic import BaseModel, Field, validator
from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import UUID

class Measurement(BaseModel):
    """Weight measurement model."""
    uuid: UUID
    weight: float = Field(gt=0, le=1000)
    unit: str = Field(regex="^(kg|lbs|lb|g|oz)$")
    effective_date_time: datetime
    source: str
    metadata: Optional[Dict[str, Any]] = None

class ProcessRequest(BaseModel):
    """Request to process measurements."""
    measurements: List[Measurement]
    options: Optional[Dict[str, Any]] = {}

class MeasurementResult(BaseModel):
    """Result of processing a single measurement."""
    uuid: UUID
    accepted: bool
    quality_score: Optional[float] = None
    kalman_estimate: Optional[float] = None
    kalman_uncertainty: Optional[float] = None
    rejection_reason: Optional[str] = None
    reset_triggered: bool = False

class ProcessResponse(BaseModel):
    """Response from processing measurements."""
    status: str
    processed_count: int
    accepted_count: int
    rejected_count: int
    measurements: List[MeasurementResult]
    state_update: Optional[Dict[str, Any]] = None
```

#### 2.2 Service Layer
Create service classes that orchestrate processing.

**File**: `src/services/weight_processor_service.py`
```python
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from ..api.models import Measurement, MeasurementResult, ProcessResponse
from ..processing.processor import process_measurement
from ..database.base import StateStore
from ..config.config_manager import ConfigManager

logger = logging.getLogger(__name__)

class WeightProcessorService:
    """Service layer for weight processing operations."""

    def __init__(self, state_store: StateStore = None, config: Dict[str, Any] = None):
        self.state_store = state_store or self._get_default_store()
        self.config = config or ConfigManager.load_config()

    def process_batch(self, user_id: str, measurements: List[Measurement]) -> ProcessResponse:
        """
        Process a batch of measurements for a user.

        Args:
            user_id: User identifier
            measurements: List of measurements to process

        Returns:
            ProcessResponse with results for all measurements
        """
        # Check for historical conflicts
        conflict = self._check_historical_conflict(user_id, measurements)
        if conflict:
            raise HistoricalConflictError(conflict)

        # Sort measurements chronologically
        sorted_measurements = sorted(measurements, key=lambda m: m.effective_date_time)

        # Process each measurement
        results = []
        accepted_count = 0
        rejected_count = 0

        for measurement in sorted_measurements:
            try:
                result = self._process_single(user_id, measurement)
                results.append(result)

                if result.accepted:
                    accepted_count += 1
                else:
                    rejected_count += 1

            except Exception as e:
                logger.error(f"Error processing {measurement.uuid}: {e}")
                results.append(MeasurementResult(
                    uuid=measurement.uuid,
                    accepted=False,
                    rejection_reason=str(e)
                ))
                rejected_count += 1

        return ProcessResponse(
            status="processed",
            processed_count=len(results),
            accepted_count=accepted_count,
            rejected_count=rejected_count,
            measurements=results
        )

    def _process_single(self, user_id: str, measurement: Measurement) -> MeasurementResult:
        """Process a single measurement."""
        result = process_measurement(
            user_id=user_id,
            weight=measurement.weight,
            timestamp=measurement.effective_date_time,
            source=measurement.source,
            unit=measurement.unit,
            config=self.config,
            db=self.state_store
        )

        return MeasurementResult(
            uuid=measurement.uuid,
            accepted=result.get('accepted', False),
            quality_score=result.get('quality_score'),
            kalman_estimate=result.get('kalman_estimate'),
            kalman_uncertainty=result.get('kalman_uncertainty'),
            rejection_reason=result.get('reason'),
            reset_triggered=result.get('reset_triggered', False)
        )
```

### Phase 3: Dependency Injection (Week 2)

#### 3.1 Factory Pattern for Components
Create factories for major components to enable testing and flexibility.

**File**: `src/factories/component_factory.py`
```python
from typing import Dict, Any
import os

from ..database.base import StateStore
from ..database.memory_store import InMemoryStateStore
from ..database.dynamodb_store import DynamoDBStateStore
from ..config.config_manager import ConfigManager

class ComponentFactory:
    """Factory for creating application components."""

    _state_stores = {}

    @classmethod
    def get_state_store(cls, backend: str = None) -> StateStore:
        """
        Get or create a state store instance.

        Args:
            backend: 'memory', 'dynamodb', or None for auto-detection
        """
        if backend is None:
            backend = os.getenv('DB_BACKEND', 'memory')

        if backend not in cls._state_stores:
            if backend == 'dynamodb':
                cls._state_stores[backend] = DynamoDBStateStore()
            else:
                cls._state_stores[backend] = InMemoryStateStore()

        return cls._state_stores[backend]

    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Get configuration."""
        return ConfigManager.load_config()

    @classmethod
    def reset(cls):
        """Reset all cached instances (for testing)."""
        cls._state_stores.clear()
```

#### 3.2 Processor Refactoring
Update processor to use dependency injection.

**Updates to**: `src/processing/processor.py`
```python
def process_measurement(
    user_id: str,
    weight: float,
    timestamp: datetime,
    source: str,
    config: Dict[str, Any],
    unit: str = "kg",
    db: StateStore = None,  # Changed from db=None
) -> Dict[str, Any]:
    """Process a single weight measurement."""

    # Use factory if no database provided
    if db is None:
        from ..factories.component_factory import ComponentFactory
        db = ComponentFactory.get_state_store()

    # Rest of the function remains the same...
```

### Phase 4: Separation of Concerns (Week 2-3)

#### 4.1 Extract CSV Processing
Move CSV-specific logic to a separate module.

**File**: `src/batch/csv_processor.py`
```python
import csv
from pathlib import Path
from typing import List, Dict, Any

from ..services.weight_processor_service import WeightProcessorService
from ..api.models import Measurement

class CSVBatchProcessor:
    """Handles CSV file processing."""

    def __init__(self, service: WeightProcessorService = None):
        self.service = service or WeightProcessorService()

    def process_file(self, csv_path: str, output_dir: str, config: Dict[str, Any]):
        """Process CSV file (legacy interface)."""
        # Move logic from main.py stream_process() here
        pass

    def _parse_csv_row(self, row: Dict[str, Any]) -> Measurement:
        """Convert CSV row to Measurement model."""
        # Parsing logic here
        pass
```

#### 4.2 Extract Visualization
Move visualization to a separate, optional module.

**File**: `src/viz/viz_service.py`
```python
from typing import Dict, Any, List, Optional

class VisualizationService:
    """Optional visualization service."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def generate_if_enabled(self, user_results: Dict[str, List], output_dir: str):
        """Generate visualizations only if enabled."""
        if not self.enabled:
            return

        # Import only when needed
        from .visualization import create_weight_timeline

        # Visualization logic here
```

### Phase 5: Lambda Handler (Week 3)

#### 5.1 Lambda Entry Point
Create Lambda handler using the refactored components.

**File**: `src/lambda_handler.py`
```python
import json
from typing import Dict, Any
import logging

from .api.models import ProcessRequest, ProcessResponse
from .services.weight_processor_service import WeightProcessorService
from .factories.component_factory import ComponentFactory

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize service (reused across invocations)
service = None

def get_service() -> WeightProcessorService:
    """Get or create service instance."""
    global service
    if service is None:
        state_store = ComponentFactory.get_state_store()
        config = ComponentFactory.get_config()
        service = WeightProcessorService(state_store, config)
    return service

def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """AWS Lambda handler."""
    try:
        # Route based on HTTP method and path
        path = event.get('resource', '')
        method = event.get('httpMethod', '')

        if path.endswith('/process/{userId}') and method == 'POST':
            return handle_process(event)
        elif path.endswith('/cleanup/{userId}') and method == 'POST':
            return handle_cleanup(event)
        elif path.endswith('/replay/{userId}') and method == 'POST':
            return handle_replay(event)
        else:
            return error_response(404, "Not Found")

    except Exception as e:
        logger.exception("Unhandled error")
        return error_response(500, str(e))

def handle_process(event: Dict[str, Any]) -> Dict[str, Any]:
    """Handle process endpoint."""
    try:
        user_id = event['pathParameters']['userId']
        body = json.loads(event['body'])
        request = ProcessRequest(**body)

        service = get_service()
        response = service.process_batch(user_id, request.measurements)

        return {
            'statusCode': 200,
            'body': response.json()
        }

    except HistoricalConflictError as e:
        return {
            'statusCode': 409,
            'body': json.dumps(e.to_dict())
        }
    except Exception as e:
        logger.exception("Error in process handler")
        return error_response(400, str(e))
```

## Implementation Roadmap

### Week 1: Foundation
- [ ] Create abstract base classes
- [ ] Implement configuration manager
- [ ] Create API models
- [ ] Set up dependency injection

### Week 2: Service Layer
- [ ] Implement service classes
- [ ] Create factories
- [ ] Update processor for DI
- [ ] Write unit tests

### Week 3: AWS Integration
- [ ] Implement DynamoDB store
- [ ] Create Lambda handler
- [ ] Add error handling
- [ ] Integration tests

### Week 4: Migration & Testing
- [ ] Create migration scripts
- [ ] Performance testing
- [ ] Documentation
- [ ] Deployment pipeline

## Testing Strategy

### Unit Tests
```python
# tests/test_service.py
import pytest
from unittest.mock import Mock

from src.services.weight_processor_service import WeightProcessorService
from src.database.base import StateStore

class TestWeightProcessorService:

    @pytest.fixture
    def mock_store(self):
        """Create mock state store."""
        store = Mock(spec=StateStore)
        store.get_state.return_value = None
        store.create_initial_state.return_value = {...}
        return store

    def test_process_batch(self, mock_store):
        """Test batch processing."""
        service = WeightProcessorService(state_store=mock_store)
        measurements = [...]
        result = service.process_batch("user1", measurements)
        assert result.processed_count == len(measurements)
```

### Integration Tests
```python
# tests/integration/test_aws.py
import boto3
import pytest

from src.database.dynamodb_store import DynamoDBStateStore

@pytest.mark.integration
class TestDynamoDBIntegration:

    def test_state_persistence(self):
        """Test state persists in DynamoDB."""
        store = DynamoDBStateStore(table_name='test-table')

        # Save state
        state = {'test': 'data'}
        store.save_state('user1', state)

        # Retrieve state
        retrieved = store.get_state('user1')
        assert retrieved == state
```

## Backward Compatibility

### Maintaining CLI Interface
```python
# main.py (updated)
import argparse
from src.batch.csv_processor import CSVBatchProcessor

def main():
    """Legacy CLI interface."""
    parser = argparse.ArgumentParser()
    # ... existing arguments ...
    args = parser.parse_args()

    # Use new batch processor
    processor = CSVBatchProcessor()
    processor.process_file(args.csv_file, args.output, config)

if __name__ == "__main__":
    main()
```

### Environment Detection
```python
def is_lambda_environment() -> bool:
    """Detect if running in Lambda."""
    return bool(os.getenv('AWS_LAMBDA_FUNCTION_NAME'))

def get_appropriate_backend() -> str:
    """Choose backend based on environment."""
    if is_lambda_environment():
        return 'dynamodb'
    return 'memory'
```

## Migration Checklist

### Pre-Refactoring
- [x] Analyze current codebase
- [x] Identify dependencies
- [ ] Create test baseline
- [ ] Document current behavior

### During Refactoring
- [ ] Create feature branches
- [ ] Implement abstractions
- [ ] Update tests incrementally
- [ ] Maintain backward compatibility

### Post-Refactoring
- [ ] Run full test suite
- [ ] Performance benchmarks
- [ ] Update documentation
- [ ] Deploy to staging

## Risk Mitigation

### Technical Risks
1. **Breaking Changes**: Mitigated by maintaining backward compatibility
2. **Performance Regression**: Continuous benchmarking
3. **State Corruption**: Comprehensive testing of state operations
4. **Dependency Conflicts**: Use virtual environments and lock files

### Process Risks
1. **Scope Creep**: Stick to defined phases
2. **Testing Gaps**: 80% coverage minimum
3. **Documentation Lag**: Update as you go

## Success Criteria

1. **All existing tests pass**: No regression in functionality
2. **New tests added**: 80%+ coverage for new code
3. **Performance maintained**: No more than 10% slowdown
4. **AWS deployment works**: Successfully deployed to Lambda
5. **Backward compatible**: Existing CLI still works

## Next Steps

1. **Review & Approve**: Get team consensus on approach
2. **Create Branch**: `feature/aws-refactoring`
3. **Start Phase 1**: Begin with database abstraction
4. **Daily Progress**: Track against roadmap