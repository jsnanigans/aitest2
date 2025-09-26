#!/bin/bash

# Deployment Readiness Checklist
# This script verifies that the project is ready for deployment

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Counters
PASS=0
FAIL=0
WARN=0

echo "=========================================="
echo "  🚀 Deployment Readiness Check"
echo "=========================================="
echo ""

# Function to print results
check_pass() {
    echo -e "${GREEN}✅ $1${NC}"
    ((PASS++))
}

check_fail() {
    echo -e "${RED}❌ $1${NC}"
    ((FAIL++))
}

check_warn() {
    echo -e "${YELLOW}⚠️  $1${NC}"
    ((WARN++))
}

info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# 1. Check Project Structure
echo "📁 Checking Project Structure..."
if [ -d "src/core" ] && [ -d "src/aws" ] && [ -d "src/local" ]; then
    check_pass "Project structure is correct (core/aws/local separation)"
else
    check_fail "Project structure is incorrect"
fi

if [ -d "aws" ] && [ -f "aws/template.yaml" ]; then
    check_pass "AWS SAM templates are in correct location"
else
    check_fail "AWS SAM templates missing or in wrong location"
fi

# 2. Check Python Imports
echo ""
echo "🐍 Checking Python Imports..."
IMPORT_ERRORS=$(find src -name "*.py" -exec grep -l "from \.\.\." {} \; 2>/dev/null | wc -l)
if [ $IMPORT_ERRORS -eq 0 ]; then
    check_warn "Some files may have relative imports (verify manually)"
else
    info "Found $IMPORT_ERRORS files with relative imports"
fi

# Check for old import patterns
OLD_IMPORTS=$(grep -r "from src\.\(database\|processing\|services\|config\|api\)" src 2>/dev/null | wc -l)
if [ $OLD_IMPORTS -eq 0 ]; then
    check_pass "No old import patterns found"
else
    check_warn "Found $OLD_IMPORTS old import patterns - review needed"
fi

# 3. Check SAM Configuration
echo ""
echo "🔧 Checking SAM Configuration..."
if [ -f "aws/samconfig.toml" ]; then
    check_pass "SAM configuration file exists"

    # Check handler path in template
    if grep -q "Handler: src.aws.lambda_handler.handler" aws/template.yaml; then
        check_pass "Lambda handler path is correct"
    else
        check_fail "Lambda handler path may be incorrect"
    fi

    # Check CodeUri
    if grep -q "CodeUri: ../" aws/template.yaml; then
        check_pass "CodeUri points to project root"
    else
        check_fail "CodeUri may be incorrect"
    fi
else
    check_fail "SAM configuration file missing"
fi

# 4. Check Requirements Files
echo ""
echo "📦 Checking Requirements Files..."
if [ -f "requirements-lambda.txt" ]; then
    check_pass "Lambda requirements file exists"

    # Check for boto3 in lambda requirements (shouldn't be there)
    if grep -q "^boto3" requirements-lambda.txt; then
        check_warn "boto3 in requirements-lambda.txt (not needed for Lambda runtime)"
    else
        check_pass "Lambda requirements optimized (no boto3)"
    fi
else
    check_fail "Lambda requirements file missing"
fi

# 5. Check Environment Configuration
echo ""
echo "🌍 Checking Environment Configuration..."
if [ -f "config/local/config.toml" ]; then
    check_pass "Local configuration exists"
else
    check_warn "Local configuration missing"
fi


# 6. Check Critical Files
echo ""
echo "📄 Checking Critical Files..."
CRITICAL_FILES=(
    "src/aws/lambda_handler.py"
    "src/core/processing/kalman.py"
    "src/core/processing/processor.py"
    "src/core/database/base.py"
    "src/core/database/dynamodb_store.py"
)

for file in "${CRITICAL_FILES[@]}"; do
    if [ -f "$file" ]; then
        check_pass "$(basename $file) exists"
    else
        check_fail "$file missing"
    fi
done

# 7. Check for Common Issues
echo ""
echo "🔍 Checking for Common Issues..."

# Check for __pycache__ directories
PYCACHE_COUNT=$(find . -type d -name "__pycache__" | wc -l)
if [ $PYCACHE_COUNT -gt 0 ]; then
    check_warn "Found $PYCACHE_COUNT __pycache__ directories (run 'make clean')"
else
    check_pass "No __pycache__ directories"
fi

# Check for .pyc files
PYC_COUNT=$(find . -name "*.pyc" | wc -l)
if [ $PYC_COUNT -gt 0 ]; then
    check_warn "Found $PYC_COUNT .pyc files (run 'make clean')"
else
    check_pass "No .pyc files"
fi

# Check for AWS SAM build artifacts
if [ -d "aws/.aws-sam" ]; then
    info "SAM build artifacts present (OK for deployment)"
else
    info "No SAM build artifacts (will need to build)"
fi

# 8. Test Docker Environment
echo ""
echo "🐳 Checking Docker Environment..."
if docker info > /dev/null 2>&1; then
    check_pass "Docker is running"

    # Check if docker-compose is available
    if docker-compose version > /dev/null 2>&1; then
        check_pass "Docker Compose is available"
    else
        check_fail "Docker Compose not available"
    fi
else
    check_warn "Docker is not running (required for local testing)"
fi

# 9. Check Git Status
echo ""
echo "📝 Checking Git Status..."
if [ -d ".git" ]; then
    MODIFIED=$(git status --porcelain | wc -l)
    if [ $MODIFIED -eq 0 ]; then
        check_pass "No uncommitted changes"
    else
        check_warn "$MODIFIED files have uncommitted changes"
        echo "  Modified files:"
        git status --porcelain | head -5 | sed 's/^/    /'
        if [ $MODIFIED -gt 5 ]; then
            echo "    ... and $((MODIFIED - 5)) more"
        fi
    fi

    # Check current branch
    BRANCH=$(git branch --show-current)
    info "Current branch: $BRANCH"
else
    check_warn "Not a git repository"
fi

# 10. Check Documentation
echo ""
echo "📚 Checking Documentation..."
if [ -f "README.md" ]; then
    check_pass "README.md exists"
else
    check_fail "README.md missing"
fi

if [ -f "DOCKER_QUICKSTART.md" ]; then
    check_pass "Docker quickstart guide exists"
else
    check_warn "Docker quickstart guide missing"
fi

# Summary
echo ""
echo "=========================================="
echo "  📊 Summary"
echo "=========================================="
echo -e "${GREEN}Passed: $PASS${NC}"
echo -e "${YELLOW}Warnings: $WARN${NC}"
echo -e "${RED}Failed: $FAIL${NC}"
echo ""

if [ $FAIL -eq 0 ]; then
    echo -e "${GREEN}✨ Project is ready for deployment!${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Run tests: make test"
    echo "  2. Test locally: make -f Makefile.docker quick-start"
    echo "  3. Build for AWS: cd aws && sam build"
    echo "  4. Deploy to dev: cd aws && sam deploy --config-env default"
    exit 0
else
    echo -e "${RED}⚠️  Project has issues that need to be fixed${NC}"
    echo ""
    echo "Fix the failed checks above before deploying."
    exit 1
fi