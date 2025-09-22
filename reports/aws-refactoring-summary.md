# AWS Refactoring - Executive Summary & Priority Guide

## 🎯 Refactoring Goal

Transform the batch CSV processor into an AWS-ready microservice while maintaining backward compatibility and all existing functionality.

## 📊 Current State Assessment

### What Works Well (Keep)
- ✅ Core processing logic (`processor.py`, `kalman.py`)
- ✅ Quality scoring system
- ✅ Modular architecture
- ✅ State management interface

### What Needs Change
- ❌ In-memory database → DynamoDB support
- ❌ File-based config → Environment variables
- ❌ CSV-centric design → API-first approach
- ❌ Synchronous processing → Lambda-compatible
- ❌ No request/response models → Pydantic schemas

## 🚀 Minimal Viable Refactoring (MVP)

### Phase 1: Critical Path (3-4 days)
These changes are **essential** for AWS deployment:

1. **Database Abstraction** ⭐️
   ```python
   # src/database/base.py
   class StateStore(ABC):
       @abstractmethod
       def get_state(self, user_id: str) -> Optional[Dict[str, Any]]
       @abstractmethod
       def save_state(self, user_id: str, state: Dict[str, Any]) -> bool
   ```

2. **API Models** ⭐️
   ```python
   # src/api/models.py
   class Measurement(BaseModel):
       uuid: UUID
       weight: float
       unit: str
       effective_date_time: datetime
       source: str
   ```

3. **Service Layer** ⭐️
   ```python
   # src/services/weight_processor_service.py
   class WeightProcessorService:
       def process_batch(self, user_id: str, measurements: List[Measurement])
       def cleanup(self, user_id: str, measurements: List[Measurement])
   ```

4. **Lambda Handler** ⭐️
   ```python
   # src/lambda_handler.py
   def handler(event: Dict, context: Any) -> Dict
   ```

### Phase 2: AWS Integration (2-3 days)

5. **DynamoDB Store**
   - Implement StateStore interface
   - Handle serialization/deserialization
   - Snapshot management

6. **Configuration Manager**
   - Environment variable support
   - Backward compatible with TOML

7. **Error Handling**
   - HistoricalConflictError
   - Proper HTTP status codes

### Phase 3: Polish (1-2 days)

8. **Testing**
   - Unit tests for new components
   - Integration tests
   - End-to-end validation

9. **Documentation**
   - API documentation
   - Deployment guide

## 📝 Quick Implementation Checklist

### Day 1: Setup & Database
```bash
# Morning
□ git checkout -b feature/aws-refactoring
□ mkdir -p src/{api,services,config}
□ Create src/database/base.py (abstract class)

# Afternoon
□ Refactor ProcessorStateDB → InMemoryStateStore
□ Update get_state_db() function
□ Run tests to verify no regression
```

### Day 2: API Layer
```bash
# Morning
□ pip install pydantic
□ Create src/api/models.py
□ Write Measurement, ProcessRequest models

# Afternoon
□ Create src/services/weight_processor_service.py
□ Implement process_batch method
□ Add historical conflict detection
```

### Day 3: Lambda Handler
```bash
# Morning
□ Create src/lambda_handler.py
□ Implement handle_process endpoint
□ Add error handling

# Afternoon
□ Implement handle_cleanup endpoint
□ Local testing with mock events
```

### Day 4: DynamoDB
```bash
# Morning
□ Create src/database/dynamodb_store.py
□ Implement get_state/save_state
□ Handle numpy serialization

# Afternoon
□ Test with local DynamoDB
□ Implement snapshot methods
```

### Day 5: Testing & Deploy
```bash
# Morning
□ Write unit tests
□ Integration testing
□ Fix any issues

# Afternoon
□ Create Dockerfile
□ Deploy to Lambda
□ End-to-end testing
```

## 🔧 Minimal Code Changes

### Change 1: Update processor.py (1 line)
```python
# FROM:
def process_measurement(..., db=None):
    if db is None:
        db = get_state_db()

# TO:
def process_measurement(..., db: StateStore = None):
    if db is None:
        db = get_state_db()
```

### Change 2: Factory Pattern (New File)
```python
# src/database/__init__.py
def get_state_db(backend=None):
    if backend == 'dynamodb':
        from .dynamodb_store import DynamoDBStateStore
        return DynamoDBStateStore()
    else:
        from .memory_store import InMemoryStateStore
        return InMemoryStateStore()
```

### Change 3: Keep main.py Working
```python
# main.py - No changes needed!
# Continue using stream_process() as before
```

## 🎨 Architecture Comparison

### Before (CSV Batch)
```
CSV File → main.py → processor.py → InMemory DB → Results
```

### After (AWS API)
```
API Request → Lambda → Service → processor.py → DynamoDB → API Response
     ↓
CSV File → main.py → Service → processor.py → InMemory DB → Results
```

## 🚦 Go/No-Go Criteria

### Must Have (Go)
- ✅ All existing tests pass
- ✅ CSV processing still works
- ✅ Lambda handler responds to requests
- ✅ State persists in DynamoDB

### Nice to Have (Can defer)
- ⏸ Async processing
- ⏸ Caching layer
- ⏸ Advanced monitoring
- ⏸ Performance optimizations

## 💡 Key Decisions

### Decision 1: Database Interface
**Choice**: Abstract base class
**Why**: Allows both in-memory and DynamoDB without changing core logic

### Decision 2: API Framework
**Choice**: Pydantic models + raw Lambda
**Why**: Lightweight, type-safe, no heavy framework dependencies

### Decision 3: Backward Compatibility
**Choice**: Keep main.py unchanged
**Why**: Existing users can continue using CLI interface

## 📈 Success Metrics

1. **No Regression**: All existing tests pass
2. **API Works**: Can process via HTTP requests
3. **State Persists**: DynamoDB stores user states
4. **Performance**: < 200ms latency for single measurement

## 🔴 Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Breaking existing functionality | Keep all tests, run continuously |
| DynamoDB costs | Use on-demand pricing initially |
| Lambda cold starts | Keep warm with scheduled pings |
| State corruption | Implement snapshots from day 1 |

## 📚 Resources Needed

### Dependencies
```txt
# requirements.txt additions
pydantic>=2.0.0
boto3>=1.26.0
```

### AWS Resources
- DynamoDB table: `weight-processor-state`
- Lambda function: `weight-processor`
- API Gateway: REST API

### Environment Variables
```bash
DB_BACKEND=dynamodb
DYNAMODB_TABLE_NAME=weight-processor-state
AWS_REGION=us-east-1
LOG_LEVEL=INFO
```

## ⚡ Quick Wins

1. **Start with models**: Define API contracts first
2. **Use factories**: Dependency injection from the start
3. **Test continuously**: Run tests after every change
4. **Mock AWS services**: Use moto for local testing
5. **Keep PRs small**: One component per PR

## 📋 Final Checklist Before Production

- [ ] All unit tests pass
- [ ] Integration tests with real DynamoDB
- [ ] API documentation complete
- [ ] Error handling tested
- [ ] Performance benchmarked
- [ ] Rollback plan ready
- [ ] Monitoring configured
- [ ] Team review completed

## 🎯 Next Immediate Action

1. **Create branch**: `git checkout -b feature/aws-refactoring`
2. **Create abstract base**: `src/database/base.py`
3. **Run tests**: Ensure nothing breaks
4. **Commit**: "Add database abstraction layer"

---

**Time Estimate**: 5-8 days for MVP
**Complexity**: Medium
**Risk**: Low (with proper testing)
**ROI**: High (enables cloud deployment)