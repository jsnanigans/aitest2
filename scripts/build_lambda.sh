#!/bin/bash
# Build Lambda deployment package

set -e

echo "Building Lambda deployment package..."

# Clean up previous build
rm -rf build/lambda lambda-deployment.zip 2>/dev/null || true

# Create build directory
mkdir -p build/lambda

echo "Copying Lambda code..."
cp -r lambda/* build/lambda/

echo "Copying core package..."
cp -r core build/lambda/

echo "Installing dependencies..."
cd build/lambda
pip install -r requirements.txt -t . --quiet

echo "Creating deployment package..."
zip -r ../../lambda-deployment.zip . -q

cd ../..

# Check package size
PACKAGE_SIZE=$(du -h lambda-deployment.zip | cut -f1)
echo "✓ Lambda package created: lambda-deployment.zip"
echo "  Package size: $PACKAGE_SIZE"

# Warn if package is too large
PACKAGE_SIZE_MB=$(du -m lambda-deployment.zip | cut -f1)
if [ $PACKAGE_SIZE_MB -gt 50 ]; then
    echo "⚠️  Warning: Package size exceeds 50MB (unzipped limit is 250MB)"
fi

echo "Done!"