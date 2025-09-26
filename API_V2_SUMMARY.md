# Weight Processor API v2 - Full Migration Complete

## Summary

The Weight Processor API has been fully migrated to v2 with improved contracts, consistency, and error handling. All v1 code has been removed.

## Key Improvements Implemented

### 1. ✅ Consistent Field Naming
- `processed_count` → `measurements_processed`
- `accepted_count` → `measurements_accepted`
- `rejected_count` → `measurements_rejected`
- `uuid` → `measurement_id`
- `weight` → `weight_value`
- `unit` → `weight_unit`
- `effectiveDateTime` → `measured_at`

### 2. ✅ Standardized Response Format
```json
{
  "success": true,
  "data": { /* actual response data */ },
  "meta": {
    "timestamp": "2024-01-01T00:00:00Z",
    "version": "2.0.0",
    "request_id": "req_abc123"
  }
}
```

### 3. ✅ Improved Error Responses
```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Clear error message",
    "field": "measurements[0].weight_value",
    "suggestion": "How to fix it",
    "documentation": "https://api.docs/errors#validation_error"
  }
}
```

### 4. ✅ Fixed Bugs
- **NoneType comparison in outlier detection** - Now properly handled with error messages
- **502 errors for time gaps** - Returns 422 with helpful message
- **Replay field name** - Uses correct `replay_from_timestamp`
- **Cleanup endpoint** - No longer requires unnecessary `measurements` field

### 5. ✅ Enhanced Features
- **Stones (st) unit support** - Now accepts stones as a valid weight unit
- **User ID in state responses** - State endpoint now includes user_id
- **Request IDs** - Every response includes tracking ID
- **Better validation** - Clear messages for invalid data

## Files Changed

### Core Files
- `src/aws/lambda_handler.py` - Complete v2 implementation
- `src/aws/api/models.py` - v2 models with validation
- `lambda/api/models.py` - v2 models (mirror)
- `sam-template.yaml` - Added API_VERSION=v2

### Test Files
- `test_lambda_api.py` - Updated for v2 testing only
- `run_tests.sh` - Works with v2 API

### Removed Files (v1/compatibility)
- `src/aws/lambda_handler_wrapper.py`
- `src/aws/lambda_handler_v2.py`
- `lambda/api/models_v2.py`
- `API_MIGRATION_GUIDE.md`

### Backup Files (for reference)
- `src/aws/lambda_handler_v1_backup.py`
- `src/aws/api/models_v1_backup.py`
- `lambda/api/models_v1_backup.py`
- `test_lambda_api_v1_backup.py`

## API Endpoints

All endpoints now use v2 contracts:

| Endpoint | Method | Changes |
|----------|--------|---------|
| `/api/v1/health` | GET | Returns standard format with success/data/meta |
| `/api/v1/process/{userId}` | POST | Field names updated, better errors |
| `/api/v1/state/{userId}` | GET | Now includes user_id in response |
| `/api/v1/state/{userId}` | DELETE | Better response format |
| `/api/v1/replay/{userId}` | POST | Fixed field name: `replay_from_timestamp` |
| `/api/v1/cleanup/{userId}` | POST | No longer requires `measurements` field |

## Testing

Run the updated test suite:
```bash
# Standard test run
python test_lambda_api.py

# Or use the runner script
./run_tests.sh
```

## Deployment

Deploy with SAM:
```bash
sam build
sam deploy
```

The API will automatically use v2 format (configured in template.yaml).

## Breaking Changes

Since this is a complete migration without backward compatibility:

1. **All clients must be updated** to use new field names
2. **Response parsing must handle** the success/data/meta structure
3. **Error handling must parse** the new error format
4. **Request fields must use** correct names (e.g., `replay_from_timestamp`)

## Benefits Achieved

1. ✅ **Consistency** - All responses follow the same structure
2. ✅ **Clarity** - Field names clearly indicate purpose
3. ✅ **Debuggability** - Request IDs and better errors
4. ✅ **Reliability** - Fixed critical bugs (NoneType, 502 errors)
5. ✅ **Usability** - Helpful error messages with suggestions
6. ✅ **Standards** - Follows REST best practices
7. ✅ **Simplicity** - No compatibility layer needed

## Next Steps

1. Update all client applications to use v2 contracts
2. Update API documentation
3. Monitor for any issues in production
4. Consider implementing remaining nice-to-have features:
   - Pagination
   - Field filtering
   - Batch operations
   - Rate limiting