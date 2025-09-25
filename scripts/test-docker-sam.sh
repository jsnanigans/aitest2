#!/bin/bash

# Test script for Docker SAM environment

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "🧪 Testing Docker SAM Environment"
echo "=================================="

# Function to print results
print_result() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✅ $2${NC}"
    else
        echo -e "${RED}❌ $2${NC}"
        return 1
    fi
}

# Test 1: Check Docker
echo -n "1. Checking Docker... "
if docker info > /dev/null 2>&1; then
    print_result 0 "Docker is running"
else
    print_result 1 "Docker is not running"
    exit 1
fi

# Test 2: Check Docker Compose
echo -n "2. Checking Docker Compose... "
if docker-compose version > /dev/null 2>&1; then
    print_result 0 "Docker Compose is installed"
else
    print_result 1 "Docker Compose is not installed"
    exit 1
fi

# Test 3: Build Docker image
echo -n "3. Building SAM Docker image... "
if docker-compose -f docker-compose.sam.yml build sam-builder > /dev/null 2>&1; then
    print_result 0 "Image built successfully"
else
    print_result 1 "Failed to build image"
    exit 1
fi

# Test 4: Start environment
echo -n "4. Starting Docker environment... "
if docker-compose -f docker-compose.sam.yml up -d > /dev/null 2>&1; then
    print_result 0 "Environment started"
    sleep 5
else
    print_result 1 "Failed to start environment"
    exit 1
fi

# Test 5: Check DynamoDB
echo -n "5. Testing DynamoDB connectivity... "
if curl -s http://localhost:8000 > /dev/null 2>&1; then
    print_result 0 "DynamoDB is accessible"
else
    print_result 1 "DynamoDB is not accessible"
fi

# Test 6: Check DynamoDB Admin
echo -n "6. Testing DynamoDB Admin UI... "
if curl -s http://localhost:8001 > /dev/null 2>&1; then
    print_result 0 "DynamoDB Admin UI is accessible"
else
    print_result 1 "DynamoDB Admin UI is not accessible"
fi

# Test 7: Test SAM CLI in container
echo -n "7. Testing SAM CLI in container... "
if docker-compose -f docker-compose.sam.yml exec -T sam-builder sam --version > /dev/null 2>&1; then
    print_result 0 "SAM CLI is working"
else
    print_result 1 "SAM CLI is not working"
fi

# Test 8: Test Python in container
echo -n "8. Testing Python in container... "
if docker-compose -f docker-compose.sam.yml exec -T sam-builder python3.12 --version > /dev/null 2>&1; then
    print_result 0 "Python 3.12 is available"
else
    print_result 1 "Python 3.12 is not available"
fi

# Test 9: Test network connectivity between containers
echo -n "9. Testing container networking... "
if docker-compose -f docker-compose.sam.yml exec -T sam-builder curl -s http://dynamodb-local:8000 > /dev/null 2>&1; then
    print_result 0 "Containers can communicate"
else
    print_result 1 "Container networking issue"
fi

# Test 10: Test volume mounts
echo -n "10. Testing volume mounts... "
if docker-compose -f docker-compose.sam.yml exec -T sam-builder ls -la /workspace/aws > /dev/null 2>&1; then
    print_result 0 "Project files are mounted"
else
    print_result 1 "Volume mount issue"
fi

echo ""
echo "=================================="
echo -e "${GREEN}🎉 All tests passed!${NC}"
echo ""
echo "You can now:"
echo "  • Run 'make -f Makefile.docker sam-api' to start the API"
echo "  • Run 'make -f Makefile.docker docker-shell' to work in the container"
echo "  • Visit http://localhost:8001 for DynamoDB Admin UI"
echo ""
echo "To stop the environment: docker-compose -f docker-compose.sam.yml down"