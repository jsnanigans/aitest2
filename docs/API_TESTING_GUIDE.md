# API Testing Guide

This directory contains test files for the Weight Processor Lambda API that can be used with various API testing tools.

## Available Test Files

### 1. Postman Collection (`weight-processor-api.postman_collection.json`)

A comprehensive Postman collection with all API endpoints and test scenarios.

#### How to Use:
1. Open Postman (download from https://www.postman.com if needed)
2. Click "Import" in the top left
3. Select the `weight-processor-api.postman_collection.json` file
4. The collection will appear in your Collections sidebar
5. Update the `base_url` variable in the collection variables:
   - For local testing: `http://localhost:5448` (default)
   - For AWS: Your API Gateway URL

#### Features:
- Organized into folders by functionality
- Pre-configured test scripts for response validation
- Dynamic timestamp generation for testing
- Environment variables for easy configuration
- Examples for all use cases including edge cases

### 2. HTTP File (`api-tests.http`)

A simple HTTP file that can be used with VS Code REST Client extension or similar tools.

#### How to Use with VS Code:
1. Install the "REST Client" extension by Huachao Mao
2. Open `api-tests.http` in VS Code
3. Click "Send Request" above any request to execute it
4. Update the `@baseUrl` variable at the top for your environment

#### How to Use with IntelliJ IDEA / WebStorm:
1. Open `api-tests.http` in your JetBrains IDE
2. Click the green arrow next to any request to execute it
3. Update variables at the top of the file

## Test Scenarios Covered

### Basic Operations
- ✅ Health check
- ✅ Process single measurement
- ✅ Process multiple measurements
- ✅ Process measurements in different units (kg/lbs)
- ✅ Get user state
- ✅ Delete user state

### Advanced Features
- ✅ Replay measurements from timestamp
- ✅ Replay with state rollback
- ✅ Cleanup with Kalman reset
- ✅ Cleanup buffer only

### Data Source Testing
- ✅ High reliability sources (care-team-upload)
- ✅ Medium reliability sources (patient-device)
- ✅ Low reliability sources (iglucose.com)

### Edge Cases & Validation
- ✅ Invalid weight (too low/high)
- ✅ BMI value detection
- ✅ Missing required fields
- ✅ Invalid UUID format
- ✅ Empty measurements array
- ✅ Outlier detection
- ✅ Long gap Kalman reset

### Performance Testing
- ✅ Batch processing examples
- ✅ Multiple source types in single request

## Quick Start Testing Sequence

1. **Start the local API:**
   ```bash
   make local
   ```

2. **Test basic health:**
   ```bash
   make local-health
   ```

3. **Run through test scenarios:**
   - Import the Postman collection or open the HTTP file
   - Start with Health Check
   - Process a single measurement
   - Get user state to verify processing
   - Try edge cases and error scenarios

## Expected Responses

### Successful Processing
```json
{
  "status": "processed",
  "processed_count": 1,
  "accepted_count": 1,
  "rejected_count": 0,
  "measurements": [
    {
      "uuid": "550e8400-e29b-41d4-a716-446655440001",
      "accepted": true,
      "quality_score": 0.95,
      "kalman_estimate": 75.3,
      "kalman_uncertainty": 2.1
    }
  ]
}
```

### Validation Error
```json
{
  "error": "Invalid request: validation error details..."
}
```

### Outlier Rejection
```json
{
  "measurements": [
    {
      "uuid": "...",
      "accepted": false,
      "rejection_reason": "Outlier detected: weight outside acceptable range"
    }
  ]
}
```

## Testing Different Environments

### Local Development
```
base_url: http://localhost:5448
```

### AWS Development
```
base_url: https://your-api-id.execute-api.region.amazonaws.com/dev
```

### AWS Production
```
base_url: https://your-api-id.execute-api.region.amazonaws.com/prod
```

## Troubleshooting

### Connection Refused
- Ensure SAM local is running: `make local`
- Check the port is correct (default: 5448)

### 400 Bad Request
- Check all required fields are present
- Verify UUID format is valid
- Ensure timestamps are in ISO format

### 500 Internal Server Error
- Check Lambda logs: `make local-logs`
- Verify dependencies are built: `make build-local`

## Performance Benchmarking

For performance testing with larger datasets, consider using:
- Postman's Collection Runner with data files
- Apache JMeter for load testing
- Custom scripts using the provided examples as templates

## Contributing

When adding new test cases:
1. Add to both Postman collection and HTTP file
2. Document expected behavior
3. Include both success and failure scenarios
4. Update this guide with new test categories