#!/bin/bash
# Cleanup script for AWS MVP deployment
# This removes unnecessary files to reduce Lambda package size

echo "🧹 Cleaning up for AWS MVP deployment..."

# Remove visualization and analysis code
echo "Removing visualization and analysis directories..."
# rm -rf src/viz/
# rm -rf src/analysis/

# Remove non-essential scripts and tests
echo "Removing non-essential directories..."
# rm -rf presentation/
# rm -rf scripts/
rm -rf integration-tests/
rm -rf reports/visualizations/
rm -rf reports/test_quarterly/

# Remove documentation that's not needed for runtime
echo "Removing excessive documentation..."
rm -rf docs/images/
# rm -rf plans/

# Clean up data files (keep only essential test data)
echo "Cleaning up data files..."
# find data/ -name "*.csv" -size +1M -delete

# Remove __pycache__ directories
echo "Removing Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null

# Remove .pyc files
find . -name "*.pyc" -delete

# Create minimal requirements for Lambda
echo "Creating minimal requirements-lambda.txt..."
cat > requirements-lambda.txt << 'EOF'
# Minimal requirements for AWS Lambda deployment
# AWS Lambda runtime provides boto3

# Core dependencies
numpy==1.26.4
pydantic==2.5.3
toml==0.10.2
EOF

echo "✅ Cleanup complete!"
echo ""
echo "Package size comparison:"
du -sh . 2>/dev/null | awk '{print "Total size after cleanup: " $1}'
echo ""
echo "Next steps:"
echo "1. Review the changes with: git status"
echo "2. Test locally with: python -m pytest tests/"
echo "3. Deploy with: sam deploy --guided"
echo ""
echo "⚠️  Note: This script removes many files. Make sure you have a backup!"
