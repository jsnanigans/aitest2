#!/bin/bash

# Script to check Lambda package size after build

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "🔍 Checking Lambda Package Size"
echo "================================"

# Build the package
echo ""
echo "📦 Building Lambda package..."
cd aws
sam build --template template.yaml --use-container

# Check build directory
BUILD_DIR=".aws-sam/build/WeightProcessorFunction"

if [ ! -d "$BUILD_DIR" ]; then
    echo -e "${RED}❌ Build directory not found${NC}"
    exit 1
fi

echo ""
echo "📊 Package Contents Analysis:"
echo ""

# Count Python files
PY_COUNT=$(find "$BUILD_DIR" -name "*.py" | wc -l)
echo "  Python files: $PY_COUNT"

# Check for unwanted directories
echo ""
echo "🔍 Checking for unnecessary inclusions:"

UNWANTED_DIRS=("tests" "docs" "scripts" "local" "data" "visualization" "__pycache__" ".git")
for dir in "${UNWANTED_DIRS[@]}"; do
    if [ -d "$BUILD_DIR/$dir" ] || [ -d "$BUILD_DIR/src/$dir" ]; then
        echo -e "${RED}  ❌ Found: $dir (should be excluded)${NC}"
    else
        echo -e "${GREEN}  ✅ Not found: $dir${NC}"
    fi
done

# Check for large libraries that shouldn't be there
echo ""
echo "🔍 Checking for unnecessary libraries:"

UNWANTED_LIBS=("matplotlib" "plotly" "pandas" "seaborn" "PIL" "boto3" "botocore")
for lib in "${UNWANTED_LIBS[@]}"; do
    if find "$BUILD_DIR" -type d -name "$lib" | grep -q .; then
        echo -e "${RED}  ❌ Found: $lib (should not be in package)${NC}"
    else
        echo -e "${GREEN}  ✅ Not found: $lib${NC}"
    fi
done

# Calculate sizes
echo ""
echo "📏 Size Analysis:"
echo ""

# Total size
TOTAL_SIZE=$(du -sh "$BUILD_DIR" | cut -f1)
echo "  Total package size: $TOTAL_SIZE"

# Size breakdown by directory
echo ""
echo "  Size by directory:"
du -sh "$BUILD_DIR"/* 2>/dev/null | sort -hr | head -10 | sed 's/^/    /'

# Check numpy size specifically (it's usually the largest)
if [ -d "$BUILD_DIR/numpy" ]; then
    NUMPY_SIZE=$(du -sh "$BUILD_DIR/numpy" | cut -f1)
    echo ""
    echo "  NumPy size: $NUMPY_SIZE"
fi

# Create a deployment package to check compressed size
echo ""
echo "🗜️ Creating deployment package..."
cd "$BUILD_DIR"
zip -qr9 ../deployment-package.zip .
COMPRESSED_SIZE=$(du -sh ../deployment-package.zip | cut -f1)
echo "  Compressed package size: $COMPRESSED_SIZE"

# Lambda limits check
echo ""
echo "📋 Lambda Limits Check:"
echo ""

# Convert compressed size to MB for comparison
COMPRESSED_MB=$(du -m ../deployment-package.zip | cut -f1)

if [ "$COMPRESSED_MB" -lt 50 ]; then
    echo -e "${GREEN}  ✅ Package size ($COMPRESSED_MB MB) is well under Lambda limit (50 MB for direct upload)${NC}"
elif [ "$COMPRESSED_MB" -lt 250 ]; then
    echo -e "${YELLOW}  ⚠️  Package size ($COMPRESSED_MB MB) requires S3 upload (over 50 MB)${NC}"
else
    echo -e "${RED}  ❌ Package size ($COMPRESSED_MB MB) exceeds Lambda limit (250 MB)${NC}"
fi

# Recommendations
echo ""
echo "💡 Optimization Tips:"
echo ""

if [ "$COMPRESSED_MB" -gt 30 ]; then
    echo "  Consider using Lambda Layers for dependencies like NumPy"
    echo "  This would reduce deployment package to < 5MB"
fi

echo "  Current dependencies in requirements-lambda.txt:"
grep -v "^#" ../../../requirements-lambda.txt | grep -v "^$" | sed 's/^/    - /'

echo ""
echo "✅ Analysis complete!"