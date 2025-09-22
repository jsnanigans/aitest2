# AWS Refactoring - File Creation Priority List

## 📁 Files to Create/Modify (In Order)

### Priority 1: Database Abstraction (Day 1)

#### 1. CREATE: `src/database/base.py` (New)
```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import datetime

class StateStore(ABC):
    @abstractmethod
    def get_state(self, user_id: str) -> Optional[Dict[str, Any]]: pass

    @abstractmethod
    def save_state(self, user_id: str, state: Dict[str, Any]) -> bool: pass

    @abstractmethod
    def delete_state(self, user_id: str) -> bool: pass

    @abstractmethod
    def create_initial_state(self) -> Dict[str, Any]: pass
```

#### 2. RENAME: `src/database/database.py` → `src/database/memory_store.py`
```bash
git mv src/database/database.py src/database/memory_store.py
```

#### 3. MODIFY: `src/database/memory_store.py`
```python
# Add at top:
from .base import StateStore

# Change class definition:
class InMemoryStateStore(StateStore):  # was ProcessorStateDB
    # Keep all existing methods
```

#### 4. MODIFY: `src/database/__init__.py`
```python
from .base import StateStore
from .memory_store import InMemoryStateStore

_db_instance = None

def get_state_db(backend=None):
    global _db_instance
    if _db_instance is None:
        _db_instance = InMemoryStateStore()  # For now
    return _db_instance

# Backward compatibility
ProcessorStateDB = InMemoryStateStore
```

#### 5. MODIFY: `src/processing/processor.py` (1 line change)
```python
# Line ~36, change:
db=None,  # Old
# To:
db: StateStore = None,  # New
```

### Priority 2: API Models (Day 1-2)

#### 6. CREATE: `src/api/__init__.py` (New)
```python
"""API models and contracts."""
```

#### 7. CREATE: `src/api/models.py` (New)
```python
from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import UUID

class Measurement(BaseModel):
    uuid: UUID
    weight: float = Field(gt=0, le=1000)
    unit: str
    effective_date_time: datetime
    source: str

class ProcessRequest(BaseModel):
    measurements: List[Measurement]

class MeasurementResult(BaseModel):
    uuid: UUID
    accepted: bool
    quality_score: Optional[float] = None
    kalman_estimate: Optional[float] = None

class ProcessResponse(BaseModel):
    status: str
    processed_count: int
    accepted_count: int
    rejected_count: int
    measurements: List[MeasurementResult]
```

### Priority 3: Service Layer (Day 2)

#### 8. CREATE: `src/services/__init__.py` (New)
```python
"""Service layer for business logic."""
```

#### 9. CREATE: `src/services/weight_processor_service.py` (New)
```python
from typing import List
from ..api.models import *
from ..processing.processor import process_measurement
from ..database import get_state_db

class WeightProcessorService:
    def __init__(self):
        self.db = get_state_db()

    def process_batch(self, user_id: str, measurements: List[Measurement]) -> ProcessResponse:
        # Sort measurements
        sorted_measurements = sorted(measurements, key=lambda m: m.effective_date_time)

        # Process each
        results = []
        for m in sorted_measurements:
            result = process_measurement(
                user_id=user_id,
                weight=m.weight,
                timestamp=m.effective_date_time,
                source=m.source,
                unit=m.unit,
                config={},  # Load from config
                db=self.db
            )
            results.append(MeasurementResult(
                uuid=m.uuid,
                accepted=result.get('accepted', False),
                quality_score=result.get('quality_score')
            ))

        return ProcessResponse(
            status="processed",
            processed_count=len(results),
            accepted_count=sum(1 for r in results if r.accepted),
            rejected_count=sum(1 for r in results if not r.accepted),
            measurements=results
        )
```

### Priority 4: Lambda Handler (Day 2-3)

#### 10. CREATE: `src/lambda_handler.py` (New)
```python
import json
from .api.models import ProcessRequest
from .services.weight_processor_service import WeightProcessorService

service = WeightProcessorService()

def handler(event, context):
    try:
        # Parse request
        user_id = event['pathParameters']['userId']
        body = json.loads(event['body'])
        request = ProcessRequest(**body)

        # Process
        response = service.process_batch(user_id, request.measurements)

        # Return response
        return {
            'statusCode': 200,
            'body': response.json()
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
```

### Priority 5: Configuration (Day 3)

#### 11. CREATE: `src/config/__init__.py` (New)
```python
"""Configuration management."""
```

#### 12. CREATE: `src/config/config_manager.py` (New)
```python
import os
import tomllib
from typing import Dict, Any

class ConfigManager:
    @classmethod
    def load_config(cls) -> Dict[str, Any]:
        if os.getenv('AWS_LAMBDA_FUNCTION_NAME'):
            # Load from environment
            return {
                'kalman': {
                    'enabled': os.getenv('KALMAN_ENABLED', 'true') == 'true'
                }
            }
        else:
            # Load from file
            with open('config.toml', 'rb') as f:
                return tomllib.load(f)
```

### Priority 6: DynamoDB (Day 3-4)

#### 13. CREATE: `src/database/dynamodb_store.py` (New)
```python
import boto3
from .base import StateStore

class DynamoDBStateStore(StateStore):
    def __init__(self):
        self.dynamodb = boto3.resource('dynamodb')
        self.table = self.dynamodb.Table('weight-processor-state')

    def get_state(self, user_id: str):
        response = self.table.get_item(
            Key={'userId': user_id, 'stateType': 'current'}
        )
        return response.get('Item')

    def save_state(self, user_id: str, state):
        self.table.put_item(
            Item={'userId': user_id, 'stateType': 'current', **state}
        )
        return True
```

#### 14. UPDATE: `src/database/__init__.py`
```python
import os

def get_state_db(backend=None):
    if backend is None:
        backend = os.getenv('DB_BACKEND', 'memory')

    if backend == 'dynamodb':
        from .dynamodb_store import DynamoDBStateStore
        return DynamoDBStateStore()
    else:
        from .memory_store import InMemoryStateStore
        return InMemoryStateStore()
```

### Priority 7: Exceptions (Day 4)

#### 15. MODIFY: `src/exceptions.py`
```python
# Add:
class HistoricalConflictError(Exception):
    def __init__(self, details):
        self.details = details
        super().__init__("Historical conflict detected")

    def to_dict(self):
        return {'error': str(self), 'details': self.details}
```

### Priority 8: Testing (Day 4-5)

#### 16. CREATE: `tests/test_service.py` (New)
```python
import pytest
from src.services.weight_processor_service import WeightProcessorService
from src.api.models import Measurement
from datetime import datetime
from uuid import uuid4

def test_process_batch():
    service = WeightProcessorService()
    measurements = [
        Measurement(
            uuid=uuid4(),
            weight=75.0,
            unit="kg",
            effective_date_time=datetime.now(),
            source="test"
        )
    ]
    response = service.process_batch("user1", measurements)
    assert response.processed_count == 1
```

#### 17. CREATE: `tests/test_lambda.py` (New)
```python
import json
from src.lambda_handler import handler

def test_handler():
    event = {
        'pathParameters': {'userId': 'test'},
        'body': json.dumps({
            'measurements': [{
                'uuid': '550e8400-e29b-41d4-a716-446655440000',
                'weight': 75.0,
                'unit': 'kg',
                'effective_date_time': '2024-01-01T10:00:00Z',
                'source': 'test'
            }]
        })
    }
    response = handler(event, None)
    assert response['statusCode'] == 200
```

### Priority 9: Deployment Files (Day 5)

#### 18. CREATE: `Dockerfile` (New)
```dockerfile
FROM public.ecr.aws/lambda/python:3.11
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY src/ ${LAMBDA_TASK_ROOT}/src/
CMD ["src.lambda_handler.handler"]
```

#### 19. CREATE: `serverless.yml` (New)
```yaml
service: weight-processor
provider:
  name: aws
  runtime: python3.11
  environment:
    DB_BACKEND: dynamodb
    DYNAMODB_TABLE_NAME: weight-processor-state

functions:
  processor:
    handler: src.lambda_handler.handler
    events:
      - http:
          path: /api/v1/process/{userId}
          method: post
      - http:
          path: /api/v1/cleanup/{userId}
          method: post
```

#### 20. UPDATE: `requirements.txt`
```txt
# Add:
pydantic>=2.0.0
boto3>=1.26.0
```

## 📝 Git Commit Strategy

### Commit 1: Database Abstraction
```bash
git add src/database/base.py src/database/memory_store.py src/database/__init__.py
git commit -m "refactor: Add database abstraction layer for AWS compatibility"
```

### Commit 2: API Models
```bash
git add src/api/
git commit -m "feat: Add Pydantic models for API contracts"
```

### Commit 3: Service Layer
```bash
git add src/services/
git commit -m "feat: Add service layer for weight processing"
```

### Commit 4: Lambda Handler
```bash
git add src/lambda_handler.py
git commit -m "feat: Add AWS Lambda handler"
```

### Commit 5: DynamoDB Support
```bash
git add src/database/dynamodb_store.py
git commit -m "feat: Add DynamoDB state storage"
```

## ✅ Verification Steps After Each Phase

### After Database Abstraction:
```bash
python -m pytest tests/  # All existing tests should pass
```

### After API Models:
```python
# Test in Python REPL
from src.api.models import Measurement
from datetime import datetime
from uuid import uuid4

m = Measurement(
    uuid=uuid4(),
    weight=75.0,
    unit="kg",
    effective_date_time=datetime.now(),
    source="test"
)
print(m.json())  # Should serialize correctly
```

### After Service Layer:
```python
# Test service directly
from src.services.weight_processor_service import WeightProcessorService
service = WeightProcessorService()
# Should initialize without errors
```

### After Lambda Handler:
```bash
# Test locally
python -c "from src.lambda_handler import handler; print('Handler loads successfully')"
```

## 🚨 Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| Import errors | Update `__init__.py` files |
| Type errors | Add type hints gradually |
| Test failures | Run tests after each change |
| Config not found | Add defaults in ConfigManager |
| DynamoDB errors | Use local DynamoDB for testing |

## 📊 Progress Tracking

```markdown
## Refactoring Progress

### Day 1
- [x] Database abstraction
- [x] Memory store refactor
- [ ] Basic tests passing

### Day 2
- [ ] API models
- [ ] Service layer
- [ ] Lambda handler

### Day 3
- [ ] DynamoDB implementation
- [ ] Configuration manager
- [ ] Integration tests

### Day 4
- [ ] Exception handling
- [ ] Error responses
- [ ] End-to-end testing

### Day 5
- [ ] Documentation
- [ ] Deployment setup
- [ ] Final review
```

---

**Remember**: Test after each file change. Small, incremental changes are better than large rewrites.