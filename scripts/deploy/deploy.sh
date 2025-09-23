#!/bin/bash
# Deploy Weight Processor to AWS Lambda

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
ENVIRONMENT="dev"
BUILD_ONLY=false
SKIP_BUILD=false
AUTO_CONFIRM=false

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --env|--environment) ENVIRONMENT="$2"; shift ;;
        --build-only) BUILD_ONLY=true ;;
        --skip-build) SKIP_BUILD=true ;;
        --auto-confirm|-y) AUTO_CONFIRM=true ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --env, --environment ENV    Deploy to specific environment (dev/staging/prod)"
            echo "  --build-only                Only build, don't deploy"
            echo "  --skip-build                Skip build step, deploy existing package"
            echo "  --auto-confirm, -y          Auto confirm deployment"
            echo "  --help, -h                  Show this help message"
            exit 0
            ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

echo -e "${GREEN}Weight Processor Deployment Script${NC}"
echo "=================================="
echo "Environment: ${ENVIRONMENT}"
echo ""

# Check for required tools
check_tool() {
    if ! command -v $1 &> /dev/null; then
        echo -e "${RED}Error: $1 is not installed${NC}"
        echo "Please install $1 and try again"
        exit 1
    fi
}

echo "Checking required tools..."
check_tool "sam"
check_tool "aws"
check_tool "python3"

# Validate environment
if [[ ! "$ENVIRONMENT" =~ ^(dev|staging|prod)$ ]]; then
    echo -e "${RED}Error: Invalid environment '$ENVIRONMENT'${NC}"
    echo "Valid environments: dev, staging, prod"
    exit 1
fi

# Build Lambda package
if [ "$SKIP_BUILD" = false ]; then
    echo ""
    echo -e "${YELLOW}Building Lambda package...${NC}"

    # Clean build directory
    rm -rf .aws-sam

    # Build with SAM
    sam build \
        --use-container \
        --template template.yaml \
        --parallel

    if [ $? -ne 0 ]; then
        echo -e "${RED}Build failed!${NC}"
        exit 1
    fi

    echo -e "${GREEN}Build successful!${NC}"
fi

# Exit if build-only
if [ "$BUILD_ONLY" = true ]; then
    echo ""
    echo -e "${GREEN}Build complete. Skipping deployment.${NC}"
    exit 0
fi

# Deploy
echo ""
echo -e "${YELLOW}Deploying to AWS...${NC}"

# Set confirm flag
CONFIRM_FLAG=""
if [ "$AUTO_CONFIRM" = true ]; then
    CONFIRM_FLAG="--no-confirm-changeset"
fi

# Deploy with SAM
sam deploy \
    --config-env "$ENVIRONMENT" \
    --no-fail-on-empty-changeset \
    $CONFIRM_FLAG

if [ $? -ne 0 ]; then
    echo -e "${RED}Deployment failed!${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}Deployment successful!${NC}"

# Get stack outputs
echo ""
echo "Getting stack outputs..."
STACK_NAME="weight-processor-${ENVIRONMENT}"

API_ENDPOINT=$(aws cloudformation describe-stacks \
    --stack-name "$STACK_NAME" \
    --query "Stacks[0].Outputs[?OutputKey=='ApiEndpoint'].OutputValue" \
    --output text 2>/dev/null)

API_KEY_ID=$(aws cloudformation describe-stacks \
    --stack-name "$STACK_NAME" \
    --query "Stacks[0].Outputs[?OutputKey=='ApiKeyId'].OutputValue" \
    --output text 2>/dev/null)

if [ ! -z "$API_ENDPOINT" ]; then
    echo ""
    echo "API Endpoint: ${API_ENDPOINT}"

    if [ ! -z "$API_KEY_ID" ]; then
        # Get actual API key value
        API_KEY=$(aws apigateway get-api-key \
            --api-key "$API_KEY_ID" \
            --include-value \
            --query "value" \
            --output text 2>/dev/null)

        if [ ! -z "$API_KEY" ]; then
            echo "API Key: ${API_KEY}"
        fi
    fi
fi

echo ""
echo -e "${GREEN}Deployment complete!${NC}"
echo ""
echo "To test the API:"
echo "  curl -X POST ${API_ENDPOINT}/api/v1/process/test-user \\"
echo "    -H 'x-api-key: ${API_KEY}' \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d @test-payload.json"