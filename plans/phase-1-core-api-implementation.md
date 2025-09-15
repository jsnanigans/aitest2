# Plan: Phase 1 - Core API Implementation

## Summary
Implement the HTTP API layer that wraps the weight validation service, providing RESTful endpoints for weight measurement validation with quality scoring.

## Context
- Prerequisites: Phase 0 (Architectural Preparation) completed
- Current state: Service layer ready, no HTTP interface
- Target state: Fully functional REST API

## Requirements
- REST API with JSON request/response
- Lambda-compatible handler function
- Request validation and error handling
- Response formatting with quality scores
- Support for single and batch processing

## Alternatives

### Option A: Native Lambda Handler
**Approach**: Direct Lambda function with manual routing
- Pros: Minimal dependencies, fastest cold start
- Cons: Manual request parsing, limited features
- Risks: Reinventing the wheel

### Option B: FastAPI with Mangum
**Approach**: FastAPI app with Lambda adapter
- Pros: Auto-documentation, validation, async support
- Cons: Heavier dependencies, slower cold start
- Risks: Complexity for Lambda deployment

### Option C: Flask/Chalice
**Approach**: Lightweight framework with Lambda support
- Pros: Simple, well-documented, AWS-native (Chalice)
- Cons: Less modern than FastAPI, manual validation
- Risks: Limited async support

## Recommendation
**Option A: Native Lambda Handler** initially, with clear upgrade path to FastAPI

## High-Level Design

### API Structure
```
handler.py (Lambda entry point)
    ↓
request_validator.py
    ↓
service.py (from Phase 0)
    ↓
response_formatter.py
```

### Endpoint Design
```
POST /validate
- Single or batch weight validation
- Returns quality scores and acceptance

GET /health
- Service health check
- Returns version and status

GET /users/{user_id}/state (optional)
- Retrieve current Kalman state
- For debugging/monitoring
```

## Implementation Plan (No Code)

### Step 1: Create Lambda Handler
**File**: `src/api/handler.py` (new)
- Main Lambda handler function:
  - Parse API Gateway event
  - Extract HTTP method and path
  - Route to appropriate handler
  - Format response with status codes
- Error handling:
  - Catch all exceptions
  - Return appropriate HTTP status
  - Include correlation ID in errors
- Support both v1 and v2 API Gateway events

### Step 2: Implement Request Models
**File**: `src/api/models.py` (new)
- Create Pydantic models (or dataclasses):
  - `MeasurementRequest`:
    - user_id: str (required)
    - measurement_id: str (required)
    - timestamp: datetime (required)
    - weight: float (required)
    - unit: Literal['kg', 'lbs'] (default: 'kg')
    - source: str (required)
  - `BatchValidationRequest`:
    - measurements: List[MeasurementRequest]
    - options: Dict (optional)
  - `ValidationResponse`:
    - user_id: str
    - measurement_id: str
    - quality_score: float
    - accepted: bool
    - components: Dict[str, float]
    - kalman_prediction: Optional[float]
    - filtered_weight: Optional[float]
    - reason: Optional[str]
  - `BatchValidationResponse`:
    - results: List[ValidationResponse]
    - summary: ProcessingSummary

### Step 3: Create Request Validator
**File**: `src/api/validator.py` (new)
- Validate incoming requests:
  - Check required fields
  - Validate data types
  - Verify timestamp formats
  - Convert units if needed
  - Check weight ranges (30-400 kg)
- Sanitize inputs:
  - Normalize user_id format
  - Clean source strings
  - Handle timezone conversions
- Return validation errors:
  - Field-level error messages
  - Suggested corrections
  - Error codes for client handling

### Step 4: Implement Response Formatter
**File**: `src/api/formatter.py` (new)
- Format service results to API responses:
  - Map internal fields to API fields
  - Round numerical values appropriately
  - Convert numpy types to JSON-safe
  - Add metadata (version, timestamp)
- Handle partial failures:
  - Success/failure per measurement
  - Aggregate statistics
  - Clear error messages
- Compression for large responses:
  - Gzip responses over threshold
  - Set appropriate headers

### Step 5: Create API Router
**File**: `src/api/router.py` (new)
- Route handlers for each endpoint:
  - `/validate` → validation_handler()
  - `/health` → health_handler()
  - `/users/{user_id}/state` → state_handler()
- Method validation:
  - Only POST for /validate
  - Only GET for /health and /state
  - Return 405 for wrong methods
- Path parameter extraction:
  - Parse user_id from path
  - Validate format

### Step 6: Add Health Check
**File**: `src/api/health.py` (new)
- Implement health check endpoint:
  - Check database connectivity
  - Verify configuration loaded
  - Return service version
  - Include deployment metadata
- Readiness vs liveness:
  - Liveness: service is running
  - Readiness: can process requests
- Cache health status briefly

### Step 7: Implement Batch Processing
**File**: `src/api/batch.py` (new)
- Batch processing logic:
  - Group measurements by user_id
  - Sort by timestamp within user
  - Process sequentially per user
  - Collect results maintaining order
- Optimization strategies:
  - Load user states in batch
  - Reuse Kalman filter instances
  - Batch save state updates
- Error isolation:
  - One measurement failure doesn't fail batch
  - Track success/failure counts
  - Return partial results

### Step 8: Add Correlation and Tracing
**File**: `src/api/tracing.py` (new)
- Request correlation:
  - Generate or extract correlation ID
  - Pass through all layers
  - Include in logs and responses
- AWS X-Ray integration:
  - Trace segments for operations
  - Annotate with user_id, batch size
  - Capture processing time
- Logging context:
  - Structured JSON logs
  - Include correlation ID
  - Log request/response (sanitized)

## Validation & Testing

### Test Strategy
- Unit tests for each component
- Integration tests with mock service
- End-to-end tests with real Lambda
- Load tests for batch processing

### Test Cases
- [ ] Valid single measurement
- [ ] Valid batch measurements
- [ ] Invalid request formats
- [ ] Missing required fields
- [ ] Out-of-range values
- [ ] Mixed success/failure batch
- [ ] Large batch handling
- [ ] Concurrent requests

## Risks & Mitigations

### Risk 1: Request Size Limits
- **Impact**: Large batches fail
- **Mitigation**: Implement pagination/chunking
- **Monitoring**: Track request sizes

### Risk 2: Timeout Issues
- **Impact**: Long batches timeout
- **Mitigation**: Implement async processing option
- **Monitoring**: Track processing times

### Risk 3: Memory Constraints
- **Impact**: Lambda runs out of memory
- **Mitigation**: Stream processing for large batches
- **Monitoring**: Memory usage metrics

## Acceptance Criteria
- [ ] Lambda handler processes requests
- [ ] Single measurement validation works
- [ ] Batch processing maintains order
- [ ] Error responses are informative
- [ ] Health check endpoint operational
- [ ] All responses are JSON-valid
- [ ] Correlation IDs tracked
- [ ] Tests achieve 90% coverage

## Configuration
```yaml
# Environment Variables
API_VERSION: "1.0.0"
MAX_BATCH_SIZE: 100
REQUEST_TIMEOUT: 30
ENABLE_TRACING: true
LOG_LEVEL: INFO
CORS_ORIGINS: "*"
```

## API Documentation

### POST /validate
```json
Request:
{
  "measurements": [
    {
      "user_id": "user123",
      "measurement_id": "meas456",
      "timestamp": "2025-09-15T10:30:00Z",
      "weight": 75.5,
      "unit": "kg",
      "source": "patient-device"
    }
  ]
}

Response (200):
{
  "results": [
    {
      "user_id": "user123",
      "measurement_id": "meas456",
      "quality_score": 0.85,
      "accepted": true,
      "components": {
        "safety": 0.95,
        "plausibility": 0.88,
        "consistency": 0.82,
        "reliability": 0.75
      },
      "kalman_prediction": 75.2,
      "filtered_weight": 75.4,
      "reason": null
    }
  ],
  "summary": {
    "total": 1,
    "accepted": 1,
    "rejected": 0
  }
}

Response (400):
{
  "error": "ValidationError",
  "message": "Invalid request format",
  "details": [
    {
      "field": "measurements[0].weight",
      "error": "Value must be between 30 and 400"
    }
  ],
  "correlation_id": "req-123"
}
```

## Dependencies
- Phase 0 must be complete
- No external API dependencies
- AWS Lambda Python 3.11 runtime

## Next Steps
- Phase 2: State Persistence (DynamoDB)
- Phase 3: Idempotency & Ordering
- Phase 4: Batch Optimization