#!/bin/bash
# Build Lambda package using uv

echo "Building Lambda package with uv..."

# Create a clean build directory
rm -rf .aws-sam/build
mkdir -p .aws-sam/build/WeightProcessorFunction

# Copy source code
cp -r src .aws-sam/build/WeightProcessorFunction/
cp -r requirements-lambda.txt .aws-sam/build/WeightProcessorFunction/

# Install dependencies with uv
cd .aws-sam/build/WeightProcessorFunction
uv pip install -r requirements-lambda.txt --target .

# Clean up unnecessary files
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete
rm -rf *.dist-info

echo "✅ Build complete"
echo "You can now run: sam local start-api --skip-pull-image"