# AWS Refactoring - Step-by-Step Implementation Guide

## Quick Start Commands

```bash
# Create new branch for refactoring
git checkout -b feature/aws-refactoring

# Create new directory structure
mkdir -p src/{api,services,config,factories,batch}
touch src/database/base.py
touch src/database/dynamodb_store.py
touch src/api/{__init__.py,models.py}
touch src/services/{__init__.py,weight_processor_service.py}
touch src/config/{__init__.py,config_manager.py}
touch src/factories/{__init__.py,component_factory.py}
```

## Step 1: Database Abstraction Layer

### 1.1 Create Abstract Base Class
**File**: `src/database/base.py`
```python
"""Abstract base class for state storage."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime


class StateStore(ABC):
    """Abstract interface for state storage backends."""

    @abstractmethod
    def get_state(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve state for a user."""
        pass

    @abstractmethod
    def save_state(self, user_id: str, state: Dict[str, Any]) -> bool:
        """Save state for a user."""
        pass

    @abstractmethod
    def delete_state(self, user_id: str) -> bool:
        """Delete state for a user."""
        pass

    @abstractmethod
    def create_initial_state(self) -> Dict[str, Any]:
        """Create an empty initial state."""
        pass

    @abstractmethod
    def save_state_snapshot(self, user_id: str, timestamp: datetime) -> bool:
        """Save a snapshot of current state."""
        pass

    @abstractmethod
    def restore_state_snapshot(self, user_id: str) -> bool:
        """Restore state from snapshot."""
        pass

    @abstractmethod
    def get_snapshot(self, user_id: str, timestamp: datetime) -> Optional[Dict[str, Any]]:
        """Get the nearest snapshot before the given timestamp."""
        pass

    @abstractmethod
    def check_and_restore_snapshot(self, user_id: str, buffer_start_time: datetime) -> dict:
        """Check if a snapshot exists and restore it atomically."""
        pass

    @abstractmethod
    def export_to_csv(self, filepath: str) -> int:
        """Export all states to CSV."""
        pass
```

### 1.2 Refactor Existing Database Class
**File**: `src/database/memory_store.py`
```python
"""In-memory implementation of StateStore."""

import copy
import csv
import json
import logging
from datetime import datetime
from typing import Dict, Optional, Any

import numpy as np

from .base import StateStore

logger = logging.getLogger(__name__)


class InMemoryStateStore(StateStore):
    """
    In-memory state storage for weight processor.
    This is the refactored version of ProcessorStateDB.
    """

    def __init__(self, storage_path: Optional[str] = None):
        """Initialize in-memory state database."""
        self.states = {}
        self._snapshots = {}
        self.storage_path = storage_path  # For future file persistence

    # Copy all methods from current ProcessorStateDB
    # Just add the StateStore interface methods

    def get_state(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve state for a user."""
        if user_id in self.states:
            return copy.deepcopy(self.states[user_id])
        return None

    def save_state(self, user_id: str, state: Dict[str, Any]) -> bool:
        """Save state for a user."""
        try:
            self.states[user_id] = copy.deepcopy(state)
            return True
        except Exception as e:
            logger.error(f"Error saving state: {e}")
            return False

    # ... rest of the methods from ProcessorStateDB ...
```

### 1.3 Create DynamoDB Implementation
**File**: `src/database/dynamodb_store.py`
```python
"""DynamoDB implementation of StateStore."""

import json
import logging
import os
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Any, Optional, List

import boto3
import numpy as np
from botocore.exceptions import ClientError

from .base import StateStore

logger = logging.getLogger(__name__)


class DynamoDBStateStore(StateStore):
    """DynamoDB-based state storage for AWS deployment."""

    def __init__(self, table_name: str = None, region: str = None):
        """
        Initialize DynamoDB state store.

        Args:
            table_name: DynamoDB table name
            region: AWS region
        """
        self.table_name = table_name or os.getenv('DYNAMODB_TABLE_NAME', 'weight-processor-state')
        self.region = region or os.getenv('AWS_REGION', 'us-east-1')

        # Initialize DynamoDB client
        self.dynamodb = boto3.resource('dynamodb', region_name=self.region)
        self.table = self.dynamodb.Table(self.table_name)

    def get_state(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve state for a user from DynamoDB."""
        try:
            response = self.table.get_item(
                Key={
                    'userId': user_id,
                    'stateType': 'current'
                }
            )

            if 'Item' not in response:
                return None

            # Deserialize the state
            state = self._deserialize_state(response['Item'])
            return state

        except ClientError as e:
            logger.error(f"Error getting state from DynamoDB: {e}")
            return None

    def save_state(self, user_id: str, state: Dict[str, Any]) -> bool:
        """Save state to DynamoDB."""
        try:
            # Serialize the state
            item = self._serialize_state(state)
            item.update({
                'userId': user_id,
                'stateType': 'current',
                'updatedAt': datetime.utcnow().isoformat(),
                'version': state.get('version', 0) + 1
            })

            # Save to DynamoDB
            self.table.put_item(Item=item)
            return True

        except ClientError as e:
            logger.error(f"Error saving state to DynamoDB: {e}")
            return False

    def delete_state(self, user_id: str) -> bool:
        """Delete state from DynamoDB."""
        try:
            # Delete current state
            self.table.delete_item(
                Key={
                    'userId': user_id,
                    'stateType': 'current'
                }
            )

            # Delete all snapshots
            self._delete_user_snapshots(user_id)
            return True

        except ClientError as e:
            logger.error(f"Error deleting state from DynamoDB: {e}")
            return False

    def create_initial_state(self) -> Dict[str, Any]:
        """Create an empty initial state."""
        return {
            'kalman_params': None,
            'last_state': None,
            'last_covariance': None,
            'last_timestamp': None,
            'last_accepted_timestamp': None,
            'last_source': None,
            'last_raw_weight': None,
            'measurement_history': [],
            'reset_events': [],
            'measurements_since_reset': 0,
            'adaptation_state': {},
            'version': 0
        }

    def save_state_snapshot(self, user_id: str, timestamp: datetime) -> bool:
        """Save a snapshot to DynamoDB."""
        try:
            # Get current state
            current_state = self.get_state(user_id)
            if not current_state:
                return False

            # Create snapshot item
            snapshot = self._serialize_state(current_state)
            snapshot.update({
                'userId': user_id,
                'stateType': f'snapshot_{timestamp.isoformat()}',
                'snapshotTime': timestamp.isoformat(),
                'ttl': int((timestamp + timedelta(days=7)).timestamp())  # 7-day retention
            })

            # Save to DynamoDB
            self.table.put_item(Item=snapshot)
            return True

        except ClientError as e:
            logger.error(f"Error saving snapshot to DynamoDB: {e}")
            return False

    def get_snapshot(self, user_id: str, timestamp: datetime) -> Optional[Dict[str, Any]]:
        """Get the nearest snapshot before the given timestamp."""
        try:
            # Query snapshots
            response = self.table.query(
                KeyConditionExpression='userId = :uid AND stateType < :st',
                ExpressionAttributeValues={
                    ':uid': user_id,
                    ':st': f'snapshot_{timestamp.isoformat()}'
                },
                ScanIndexForward=False,  # Descending order
                Limit=1
            )

            if not response.get('Items'):
                return None

            return self._deserialize_state(response['Items'][0])

        except ClientError as e:
            logger.error(f"Error getting snapshot from DynamoDB: {e}")
            return None

    def _serialize_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize state for DynamoDB storage."""
        serialized = {}

        for key, value in state.items():
            if value is None:
                continue  # Skip None values

            if isinstance(value, np.ndarray):
                # Convert numpy arrays to lists
                serialized[key] = value.tolist()
            elif isinstance(value, datetime):
                # Convert datetime to ISO string
                serialized[key] = value.isoformat()
            elif isinstance(value, float):
                # Convert float to Decimal for DynamoDB
                serialized[key] = Decimal(str(value))
            elif isinstance(value, dict):
                # Recursively serialize nested dicts
                serialized[key] = self._serialize_state(value)
            elif isinstance(value, list):
                # Handle lists
                serialized[key] = [
                    self._serialize_value(item) for item in value
                ]
            else:
                serialized[key] = value

        return serialized

    def _deserialize_state(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Deserialize state from DynamoDB."""
        state = {}

        for key, value in item.items():
            if key in ['userId', 'stateType', 'ttl']:
                continue  # Skip DynamoDB metadata

            if key in ['last_timestamp', 'last_accepted_timestamp']:
                # Convert ISO strings back to datetime
                if isinstance(value, str):
                    state[key] = datetime.fromisoformat(value)
                else:
                    state[key] = value
            elif key == 'kalman_params' and value:
                # Handle Kalman parameters
                state[key] = self._deserialize_kalman_params(value)
            elif isinstance(value, Decimal):
                # Convert Decimal back to float
                state[key] = float(value)
            else:
                state[key] = value

        return state

    def _deserialize_kalman_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Deserialize Kalman filter parameters."""
        deserialized = {}

        for key, value in params.items():
            if key in ['x', 'P'] and isinstance(value, list):
                # Convert lists back to numpy arrays
                deserialized[key] = np.array(value)
            elif isinstance(value, Decimal):
                deserialized[key] = float(value)
            else:
                deserialized[key] = value

        return deserialized

    def _serialize_value(self, value: Any) -> Any:
        """Serialize a single value."""
        if isinstance(value, float):
            return Decimal(str(value))
        elif isinstance(value, datetime):
            return value.isoformat()
        elif isinstance(value, dict):
            return self._serialize_state(value)
        return value

    # Implement remaining abstract methods...
```

## Step 2: Update Database Factory

### 2.1 Modify get_state_db Function
**File**: `src/database/__init__.py`
```python
"""Database module initialization."""

import os
from typing import Optional

from .base import StateStore
from .memory_store import InMemoryStateStore

# Singleton instance
_db_instance: Optional[StateStore] = None


def get_state_db(backend: str = None) -> StateStore:
    """
    Get or create state database instance.

    Args:
        backend: 'memory', 'dynamodb', or None for auto-detection

    Returns:
        StateStore instance
    """
    global _db_instance

    if _db_instance is None:
        if backend is None:
            backend = os.getenv('DB_BACKEND', 'memory')

        if backend == 'dynamodb':
            # Import only when needed to avoid AWS SDK dependency
            from .dynamodb_store import DynamoDBStateStore
            _db_instance = DynamoDBStateStore()
        else:
            _db_instance = InMemoryStateStore()

    return _db_instance


def reset_db_instance():
    """Reset the singleton instance (for testing)."""
    global _db_instance
    _db_instance = None


# For backward compatibility
ProcessorStateDB = InMemoryStateStore
```

## Step 3: Configuration Management

### 3.1 Create Config Manager
**File**: `src/config/config_manager.py`
```python
"""Configuration management for multiple environments."""

import os
import tomllib
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigManager:
    """Manages configuration from multiple sources."""

    _cached_config: Optional[Dict[str, Any]] = None

    @classmethod
    def load_config(cls, source: str = 'auto', config_path: str = None) -> Dict[str, Any]:
        """
        Load configuration from file or environment.

        Args:
            source: 'file', 'env', or 'auto'
            config_path: Path to config file (for 'file' source)

        Returns:
            Configuration dictionary
        """
        # Use cached config if available
        if cls._cached_config is not None:
            return cls._cached_config

        # Determine source
        if source == 'auto':
            if os.getenv('AWS_LAMBDA_FUNCTION_NAME'):
                source = 'env'
            else:
                source = 'file'

        # Load configuration
        if source == 'env':
            config = cls._load_from_env()
        else:
            config = cls._load_from_file(config_path or 'config.toml')

        # Cache the config
        cls._cached_config = config
        return config

    @classmethod
    def _load_from_env(cls) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        return {
            'data': {
                'max_users': int(os.getenv('MAX_USERS', '0')),
                'min_readings': int(os.getenv('MIN_READINGS', '0'))
            },
            'kalman': {
                'enabled': os.getenv('KALMAN_ENABLED', 'true').lower() == 'true',
                'adaptive': os.getenv('KALMAN_ADAPTIVE', 'true').lower() == 'true',
                'process_noise': float(os.getenv('KALMAN_PROCESS_NOISE', '1.0')),
                'observation_noise': float(os.getenv('KALMAN_OBS_NOISE', '4.0')),
                'adaptation': {
                    'enabled': os.getenv('KALMAN_ADAPTATION_ENABLED', 'true').lower() == 'true',
                    'initial_multiplier': float(os.getenv('KALMAN_ADAPTATION_MULTIPLIER', '10.0')),
                    'decay_rate': float(os.getenv('KALMAN_ADAPTATION_DECAY', '0.1'))
                },
                'resets': {
                    'hard_gap_days': int(os.getenv('KALMAN_HARD_GAP_DAYS', '30')),
                    'soft_sources': os.getenv('KALMAN_SOFT_SOURCES', 'questionnaire').split(',')
                }
            },
            'quality_scoring': {
                'enabled': os.getenv('QUALITY_SCORING_ENABLED', 'true').lower() == 'true',
                'weights': {
                    'kalman': float(os.getenv('QS_WEIGHT_KALMAN', '0.4')),
                    'temporal': float(os.getenv('QS_WEIGHT_TEMPORAL', '0.3')),
                    'source': float(os.getenv('QS_WEIGHT_SOURCE', '0.3'))
                },
                'thresholds': {
                    'outlier_override': float(os.getenv('QS_OUTLIER_OVERRIDE', '0.8')),
                    'acceptance': float(os.getenv('QS_ACCEPTANCE', '0.3'))
                }
            },
            'outlier_detection': {
                'enabled': os.getenv('OUTLIER_DETECTION_ENABLED', 'true').lower() == 'true',
                'iqr_multiplier': float(os.getenv('OUTLIER_IQR_MULTIPLIER', '1.5')),
                'mad_threshold': float(os.getenv('OUTLIER_MAD_THRESHOLD', '3.0'))
            },
            'replay': {
                'enabled': os.getenv('REPLAY_ENABLED', 'false').lower() == 'true',
                'buffer_hours': int(os.getenv('REPLAY_BUFFER_HOURS', '72'))
            },
            'database': {
                'backend': os.getenv('DB_BACKEND', 'memory'),
                'table_name': os.getenv('DB_TABLE_NAME', 'weight-processor-state'),
                'region': os.getenv('AWS_REGION', 'us-east-1')
            },
            'logging': {
                'level': os.getenv('LOG_LEVEL', 'INFO'),
                'verbose': os.getenv('LOG_VERBOSE', 'false').lower() == 'true'
            }
        }

    @classmethod
    def _load_from_file(cls, config_path: str) -> Dict[str, Any]:
        """Load configuration from TOML file."""
        path = Path(config_path)

        if not path.exists():
            # Return minimal defaults if file doesn't exist
            return cls._get_defaults()

        with open(path, 'rb') as f:
            return tomllib.load(f)

    @classmethod
    def _get_defaults(cls) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'data': {'max_users': 0, 'min_readings': 0},
            'kalman': {
                'enabled': True,
                'adaptive': True,
                'process_noise': 1.0,
                'observation_noise': 4.0
            },
            'quality_scoring': {'enabled': True},
            'database': {'backend': 'memory'},
            'logging': {'level': 'INFO'}
        }

    @classmethod
    def reset_cache(cls):
        """Reset cached configuration (for testing)."""
        cls._cached_config = None
```

## Step 4: API Models

### 4.1 Create Pydantic Models
**File**: `src/api/models.py`
```python
"""API request and response models."""

from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import UUID

from pydantic import BaseModel, Field, validator


class Measurement(BaseModel):
    """Weight measurement model."""
    uuid: UUID
    weight: float = Field(gt=0, le=1000, description="Weight value")
    unit: str = Field(regex="^(kg|lbs?|g|oz)$", description="Unit of measurement")
    effective_date_time: datetime = Field(alias="effectiveDateTime")
    source: str = Field(description="Data source")
    metadata: Optional[Dict[str, Any]] = None

    class Config:
        populate_by_name = True  # Allow both snake_case and camelCase

    @validator('weight')
    def validate_weight(cls, v, values):
        """Validate weight is within physiological bounds."""
        unit = values.get('unit', 'kg')

        # Convert to kg for validation
        weight_kg = v
        if unit in ['lb', 'lbs']:
            weight_kg = v * 0.453592
        elif unit == 'g':
            weight_kg = v / 1000
        elif unit == 'oz':
            weight_kg = v * 0.0283495

        if weight_kg < 10 or weight_kg > 500:
            raise ValueError(f"Weight {weight_kg}kg outside valid range (10-500kg)")

        return v


class UserProfile(BaseModel):
    """User profile for validation."""
    height: Optional[float] = None
    height_unit: Optional[str] = "cm"
    date_of_birth: Optional[str] = None
    gender: Optional[str] = None


class ProcessOptions(BaseModel):
    """Options for processing."""
    fail_on_historical_conflict: bool = True


class ProcessRequest(BaseModel):
    """Request to process measurements."""
    measurements: List[Measurement]
    options: Optional[ProcessOptions] = ProcessOptions()


class CleanupOptions(BaseModel):
    """Options for cleanup operation."""
    reset_state: bool = True
    include_quality_scores: bool = True
    include_debug_info: bool = False


class CleanupRequest(BaseModel):
    """Request for cleanup operation."""
    measurements: List[Measurement]
    user_profile: Optional[UserProfile] = None
    options: Optional[CleanupOptions] = CleanupOptions()


class ReplayOptions(BaseModel):
    """Options for replay operation."""
    use_snapshot: bool = True
    create_new_snapshot: bool = True


class ReplayRequest(BaseModel):
    """Request for replay operation."""
    replay_from_timestamp: datetime
    measurements: List[Measurement]
    options: Optional[ReplayOptions] = ReplayOptions()


class MeasurementResult(BaseModel):
    """Result of processing a single measurement."""
    uuid: UUID
    accepted: bool
    quality_score: Optional[float] = Field(None, ge=0, le=1)
    kalman_estimate: Optional[float] = None
    kalman_uncertainty: Optional[float] = None
    rejection_reason: Optional[str] = None
    stage: Optional[str] = None
    reset_triggered: bool = False
    components: Optional[Dict[str, float]] = None


class StateUpdate(BaseModel):
    """State update information."""
    previous_weight: Optional[float] = None
    current_weight: Optional[float] = None
    last_processed_timestamp: datetime


class ProcessResponse(BaseModel):
    """Response from processing measurements."""
    status: str
    processed_count: int = Field(ge=0)
    accepted_count: int = Field(ge=0)
    rejected_count: int = Field(ge=0)
    measurements: List[MeasurementResult]
    state_update: Optional[StateUpdate] = None


class FinalState(BaseModel):
    """Final state after processing."""
    current_weight: float
    uncertainty: float
    last_processed_timestamp: datetime
    total_measurements: int
    adaptation_state: str


class CleanupResponse(BaseModel):
    """Response from cleanup operation."""
    user_id: str
    processed_count: int
    accepted_count: int
    rejected_count: int
    measurements: List[MeasurementResult]
    final_state: Optional[FinalState] = None


class HistoricalConflictDetails(BaseModel):
    """Details about historical conflict."""
    earliest_measurement_timestamp: datetime
    last_processed_timestamp: datetime
    replay_required: bool = True
    replay_from_timestamp: datetime
    snapshot_available: Optional[datetime] = None
    conflicting_measurements: List[str]


class HistoricalConflictResponse(BaseModel):
    """Response when historical conflict is detected."""
    status: str = "historical_conflict"
    error: str
    details: HistoricalConflictDetails
```

## Step 5: Service Layer

### 5.1 Create Service Class
**File**: `src/services/weight_processor_service.py`
```python
"""Service layer for weight processing operations."""

import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

from ..api.models import (
    Measurement,
    MeasurementResult,
    ProcessResponse,
    CleanupResponse,
    StateUpdate,
    FinalState,
    HistoricalConflictDetails,
    HistoricalConflictResponse
)
from ..database.base import StateStore
from ..processing.processor import process_measurement
from ..config.config_manager import ConfigManager
from ..exceptions import HistoricalConflictError

logger = logging.getLogger(__name__)


class WeightProcessorService:
    """Service layer for weight processing operations."""

    def __init__(self, state_store: StateStore = None, config: Dict[str, Any] = None):
        """
        Initialize service.

        Args:
            state_store: State storage backend
            config: Configuration dictionary
        """
        # Use factory pattern if not provided
        if state_store is None:
            from ..database import get_state_db
            state_store = get_state_db()

        self.state_store = state_store
        self.config = config or ConfigManager.load_config()

    def process_batch(self, user_id: str, measurements: List[Measurement]) -> ProcessResponse:
        """
        Process a batch of measurements for a user.

        Args:
            user_id: User identifier
            measurements: List of measurements to process

        Returns:
            ProcessResponse with results for all measurements

        Raises:
            HistoricalConflictError: If measurements are before last processed timestamp
        """
        # Sort measurements chronologically
        sorted_measurements = sorted(measurements, key=lambda m: m.effective_date_time)

        # Check for historical conflicts
        conflict = self._check_historical_conflict(user_id, sorted_measurements)
        if conflict:
            raise HistoricalConflictError(conflict)

        # Get initial state
        current_state = self.state_store.get_state(user_id)
        previous_weight = current_state.get('last_raw_weight') if current_state else None

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
                    rejection_reason=str(e),
                    stage="processing"
                ))
                rejected_count += 1

        # Get final state
        final_state = self.state_store.get_state(user_id)
        current_weight = final_state.get('last_raw_weight') if final_state else None

        # Create state update
        state_update = None
        if sorted_measurements:
            state_update = StateUpdate(
                previous_weight=previous_weight,
                current_weight=current_weight,
                last_processed_timestamp=sorted_measurements[-1].effective_date_time
            )

        return ProcessResponse(
            status="processed",
            processed_count=len(results),
            accepted_count=accepted_count,
            rejected_count=rejected_count,
            measurements=results,
            state_update=state_update
        )

    def cleanup(self, user_id: str, measurements: List[Measurement],
                reset_state: bool = True) -> CleanupResponse:
        """
        Perform one-time cleanup for a user.

        Args:
            user_id: User identifier
            measurements: All historical measurements
            reset_state: Whether to reset state before processing

        Returns:
            CleanupResponse with results for all measurements
        """
        # Reset state if requested
        if reset_state:
            self.state_store.delete_state(user_id)
            logger.info(f"Reset state for user {user_id}")

        # Sort measurements chronologically
        sorted_measurements = sorted(measurements, key=lambda m: m.effective_date_time)

        # Process all measurements
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
                    rejection_reason=str(e),
                    stage="processing"
                ))
                rejected_count += 1

        # Get final state
        final_state_data = self.state_store.get_state(user_id)
        final_state = None

        if final_state_data:
            final_state = FinalState(
                current_weight=final_state_data.get('last_raw_weight', 0),
                uncertainty=final_state_data.get('last_covariance', 1.0),
                last_processed_timestamp=final_state_data.get('last_timestamp', datetime.now()),
                total_measurements=len(results),
                adaptation_state="converged" if final_state_data.get('measurements_since_reset', 0) > 10 else "adapting"
            )

        return CleanupResponse(
            user_id=user_id,
            processed_count=len(results),
            accepted_count=accepted_count,
            rejected_count=rejected_count,
            measurements=results,
            final_state=final_state
        )

    def _process_single(self, user_id: str, measurement: Measurement) -> MeasurementResult:
        """Process a single measurement."""
        # Call the existing processor
        result = process_measurement(
            user_id=user_id,
            weight=measurement.weight,
            timestamp=measurement.effective_date_time,
            source=measurement.source,
            unit=measurement.unit,
            config=self.config,
            db=self.state_store
        )

        # Convert to API model
        return MeasurementResult(
            uuid=measurement.uuid,
            accepted=result.get('accepted', False),
            quality_score=result.get('quality_score'),
            kalman_estimate=result.get('kalman_estimate'),
            kalman_uncertainty=result.get('kalman_uncertainty'),
            rejection_reason=result.get('reason'),
            stage=result.get('stage'),
            reset_triggered=result.get('reset_triggered', False),
            components=result.get('quality_components')
        )

    def _check_historical_conflict(self, user_id: str,
                                  measurements: List[Measurement]) -> Optional[HistoricalConflictResponse]:
        """Check if any measurements are before last processed timestamp."""
        current_state = self.state_store.get_state(user_id)

        if not current_state or not current_state.get('last_timestamp'):
            return None  # No conflict if no previous state

        last_timestamp = current_state['last_timestamp']
        if isinstance(last_timestamp, str):
            last_timestamp = datetime.fromisoformat(last_timestamp)

        # Find conflicting measurements
        conflicting = [
            str(m.uuid) for m in measurements
            if m.effective_date_time < last_timestamp
        ]

        if not conflicting:
            return None  # No conflict

        # Get earliest measurement
        earliest = min(measurements, key=lambda m: m.effective_date_time)

        # Check for available snapshot
        snapshot = self.state_store.get_snapshot(user_id, earliest.effective_date_time)
        snapshot_time = None
        if snapshot and 'snapshotTime' in snapshot:
            snapshot_time = datetime.fromisoformat(snapshot['snapshotTime'])

        return HistoricalConflictResponse(
            error="One or more measurements are before last processed timestamp",
            details=HistoricalConflictDetails(
                earliest_measurement_timestamp=earliest.effective_date_time,
                last_processed_timestamp=last_timestamp,
                replay_from_timestamp=earliest.effective_date_time,
                snapshot_available=snapshot_time,
                conflicting_measurements=conflicting
            )
        )
```

## Step 6: Update Processor

### 6.1 Modify process_measurement Function
**File**: `src/processing/processor.py` (modifications)
```python
# Add at top of file
from ..database.base import StateStore

# Update function signature
def process_measurement(
    user_id: str,
    weight: float,
    timestamp: datetime,
    source: str,
    config: Dict[str, Any],
    unit: str = "kg",
    db: StateStore = None,  # Changed from db=None
) -> Dict[str, Any]:
    """
    Process a single weight measurement through the complete pipeline.
    """
    # Use factory if no database provided
    if db is None:
        from ..database import get_state_db
        db = get_state_db()

    # Rest of function remains the same...
```

## Step 7: Create Lambda Handler

### 7.1 Main Lambda Handler
**File**: `src/lambda_handler.py`
```python
"""AWS Lambda handler for weight processor service."""

import json
import logging
import os
from typing import Dict, Any

from .api.models import ProcessRequest, CleanupRequest, ReplayRequest
from .services.weight_processor_service import WeightProcessorService
from .services.replay_service import ReplayService
from .config.config_manager import ConfigManager
from .database import get_state_db
from .exceptions import HistoricalConflictError

# Configure logging
logger = logging.getLogger()
logger.setLevel(os.getenv('LOG_LEVEL', 'INFO'))

# Initialize services (reused across invocations)
_service = None
_replay_service = None


def get_service() -> WeightProcessorService:
    """Get or create service instance."""
    global _service
    if _service is None:
        state_store = get_state_db('dynamodb')
        config = ConfigManager.load_config('env')
        _service = WeightProcessorService(state_store, config)
    return _service


def get_replay_service() -> ReplayService:
    """Get or create replay service instance."""
    global _replay_service
    if _replay_service is None:
        _replay_service = ReplayService(get_service())
    return _replay_service


def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    AWS Lambda handler.

    Routes requests to appropriate handlers based on path and method.
    """
    try:
        # Log the event for debugging
        logger.debug(f"Received event: {json.dumps(event)}")

        # Extract routing information
        resource = event.get('resource', '')
        http_method = event.get('httpMethod', '')

        # Route to appropriate handler
        if resource == '/api/v1/process/{userId}' and http_method == 'POST':
            return handle_process(event)
        elif resource == '/api/v1/cleanup/{userId}' and http_method == 'POST':
            return handle_cleanup(event)
        elif resource == '/api/v1/replay/{userId}' and http_method == 'POST':
            return handle_replay(event)
        elif resource == '/api/v1/state/{userId}' and http_method == 'GET':
            return handle_get_state(event)
        elif resource == '/api/v1/state/{userId}' and http_method == 'DELETE':
            return handle_delete_state(event)
        else:
            return error_response(404, "Not Found")

    except Exception as e:
        logger.exception("Unhandled error in Lambda handler")
        return error_response(500, f"Internal server error: {str(e)}")


def handle_process(event: Dict[str, Any]) -> Dict[str, Any]:
    """Handle process endpoint."""
    try:
        # Extract user ID and request body
        user_id = event['pathParameters']['userId']
        body = json.loads(event['body'])

        # Parse and validate request
        request = ProcessRequest(**body)

        # Process measurements
        service = get_service()
        response = service.process_batch(user_id, request.measurements)

        return success_response(response.dict())

    except HistoricalConflictError as e:
        return conflict_response(e.to_dict())
    except ValueError as e:
        return error_response(400, f"Invalid request: {str(e)}")
    except Exception as e:
        logger.exception(f"Error processing measurements for user {user_id}")
        return error_response(500, f"Processing error: {str(e)}")


def handle_cleanup(event: Dict[str, Any]) -> Dict[str, Any]:
    """Handle cleanup endpoint."""
    try:
        # Extract user ID and request body
        user_id = event['pathParameters']['userId']
        body = json.loads(event['body'])

        # Parse and validate request
        request = CleanupRequest(**body)

        # Perform cleanup
        service = get_service()
        response = service.cleanup(
            user_id,
            request.measurements,
            request.options.reset_state
        )

        return success_response(response.dict())

    except ValueError as e:
        return error_response(400, f"Invalid request: {str(e)}")
    except Exception as e:
        logger.exception(f"Error in cleanup for user {user_id}")
        return error_response(500, f"Cleanup error: {str(e)}")


def handle_replay(event: Dict[str, Any]) -> Dict[str, Any]:
    """Handle replay endpoint."""
    try:
        # Extract user ID and request body
        user_id = event['pathParameters']['userId']
        body = json.loads(event['body'])

        # Parse and validate request
        request = ReplayRequest(**body)

        # Perform replay
        replay_service = get_replay_service()
        response = replay_service.replay_from_timestamp(
            user_id,
            request.replay_from_timestamp,
            request.measurements,
            request.options.use_snapshot,
            request.options.create_new_snapshot
        )

        return success_response(response)

    except ValueError as e:
        return error_response(400, f"Invalid request: {str(e)}")
    except Exception as e:
        logger.exception(f"Error in replay for user {user_id}")
        return error_response(500, f"Replay error: {str(e)}")


def handle_get_state(event: Dict[str, Any]) -> Dict[str, Any]:
    """Handle get state endpoint."""
    try:
        user_id = event['pathParameters']['userId']
        state_store = get_state_db('dynamodb')
        state = state_store.get_state(user_id)

        if state is None:
            return error_response(404, f"State not found for user {user_id}")

        return success_response(state)

    except Exception as e:
        logger.exception(f"Error getting state for user {user_id}")
        return error_response(500, f"Error retrieving state: {str(e)}")


def handle_delete_state(event: Dict[str, Any]) -> Dict[str, Any]:
    """Handle delete state endpoint."""
    try:
        user_id = event['pathParameters']['userId']
        state_store = get_state_db('dynamodb')
        success = state_store.delete_state(user_id)

        if success:
            return success_response({"message": f"State deleted for user {user_id}"})
        else:
            return error_response(404, f"State not found for user {user_id}")

    except Exception as e:
        logger.exception(f"Error deleting state for user {user_id}")
        return error_response(500, f"Error deleting state: {str(e)}")


def success_response(body: Any, status_code: int = 200) -> Dict[str, Any]:
    """Create successful response."""
    return {
        'statusCode': status_code,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
        },
        'body': json.dumps(body, default=str)
    }


def error_response(status_code: int, message: str) -> Dict[str, Any]:
    """Create error response."""
    return {
        'statusCode': status_code,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
        },
        'body': json.dumps({
            'error': message
        })
    }


def conflict_response(conflict_data: Dict[str, Any]) -> Dict[str, Any]:
    """Create conflict response."""
    return {
        'statusCode': 409,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
        },
        'body': json.dumps(conflict_data, default=str)
    }
```

## Implementation Checklist

### Immediate Actions (Day 1)
- [ ] Create feature branch: `git checkout -b feature/aws-refactoring`
- [ ] Create new directory structure
- [ ] Copy this implementation guide to the repo

### Phase 1: Database Abstraction (Days 2-3)
- [ ] Create `src/database/base.py`
- [ ] Refactor `ProcessorStateDB` to `InMemoryStateStore`
- [ ] Create `DynamoDBStateStore` stub
- [ ] Update `get_state_db` function
- [ ] Run existing tests to ensure no regression

### Phase 2: Configuration (Day 4)
- [ ] Create `ConfigManager`
- [ ] Update existing config loading
- [ ] Test with environment variables
- [ ] Verify backward compatibility

### Phase 3: API Models (Day 5)
- [ ] Install Pydantic: `pip install pydantic`
- [ ] Create all API models
- [ ] Write model validation tests

### Phase 4: Service Layer (Days 6-7)
- [ ] Create `WeightProcessorService`
- [ ] Implement batch processing
- [ ] Add historical conflict detection
- [ ] Write service tests

### Phase 5: Lambda Handler (Day 8)
- [ ] Create Lambda handler
- [ ] Implement all endpoints
- [ ] Add error handling
- [ ] Local testing setup

### Phase 6: Testing (Days 9-10)
- [ ] Unit tests for new components
- [ ] Integration tests
- [ ] End-to-end testing
- [ ] Performance benchmarks

### Phase 7: Documentation (Day 11)
- [ ] Update README
- [ ] API documentation
- [ ] Deployment guide
- [ ] Migration guide

### Phase 8: Deployment (Day 12)
- [ ] Create Docker image
- [ ] Deploy to AWS Lambda
- [ ] Configure API Gateway
- [ ] Smoke tests

## Testing Commands

```bash
# Run tests for specific modules
python -m pytest tests/test_database.py
python -m pytest tests/test_service.py
python -m pytest tests/test_api.py

# Run with coverage
python -m pytest --cov=src --cov-report=term-missing

# Test Lambda handler locally
python -c "from src.lambda_handler import handler; print(handler({'resource': '/api/v1/state/test', 'httpMethod': 'GET', 'pathParameters': {'userId': 'test'}}, None))"

# Test with environment variables
DB_BACKEND=dynamodb python -m pytest tests/integration/

# Benchmark performance
python scripts/benchmark_refactored.py
```

## Notes for Implementation

1. **Maintain backward compatibility** - The CSV processing should still work
2. **Use feature flags** - Allow gradual rollout of new features
3. **Document breaking changes** - If any are absolutely necessary
4. **Keep PRs small** - One phase per PR for easier review
5. **Test continuously** - Run tests after each significant change