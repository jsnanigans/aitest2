#!/bin/bash

# Test the local API endpoints without AWS authentication
echo "🧪 Testing Local API (no AWS auth required)"
echo ""

# Test health endpoint
echo "1. Testing health endpoint..."
curl -s http://localhost:5448/api/v1/health | jq .
echo ""

# Test process endpoint with sample data
echo "2. Testing process endpoint with sample data..."
curl -X POST http://localhost:5448/api/v1/process/test-user \
    -H "Content-Type: application/json" \
    -d '{
        "measurements": [
            {
                "uuid": "550e8400-e29b-41d4-a716-446655440000",
                "weight": 75.5,
                "unit": "kg",
                "effectiveDateTime": "2024-01-01T10:00:00Z",
                "source": "patient-device"
            }
        ]
    }' | jq .
echo ""

# Get user state
echo "3. Getting user state..."
curl -s http://localhost:5448/api/v1/state/test-user | jq .
echo ""

echo "✅ Local API tests complete"