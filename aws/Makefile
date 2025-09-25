# AWS SAM Makefile for optimized Lambda builds

.PHONY: build build-optimized deploy check-size clean layer

# Build with standard settings
build:
	sam build --template template.yaml

# Build with container for consistent environment
build-container:
	sam build --template template.yaml --use-container

# Optimized build with size reduction
build-optimized:
	@echo "🚀 Building optimized Lambda package..."
	sam build --template template.yaml --use-container
	@echo ""
	@echo "🧹 Removing unnecessary files..."
	# Remove test files
	find .aws-sam/build -name "*test*.py" -delete 2>/dev/null || true
	find .aws-sam/build -name "*.pyc" -delete 2>/dev/null || true
	find .aws-sam/build -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	# Remove documentation
	find .aws-sam/build -name "*.md" -delete 2>/dev/null || true
	find .aws-sam/build -name "*.txt" -not -name "requirements*.txt" -delete 2>/dev/null || true
	# Remove unnecessary numpy components (saves ~10MB)
	rm -rf .aws-sam/build/*/numpy/tests 2>/dev/null || true
	rm -rf .aws-sam/build/*/numpy/*/tests 2>/dev/null || true
	rm -rf .aws-sam/build/*/numpy/doc 2>/dev/null || true
	@echo "✅ Optimization complete"

# Check package size
check-size:
	@../scripts/check-lambda-size.sh

# Deploy to AWS
deploy-dev:
	sam deploy --config-env default

deploy-prod:
	sam deploy --config-env prod

# Clean build artifacts
clean:
	rm -rf .aws-sam
	rm -f deployment-package.zip

# Create a Lambda Layer for dependencies (advanced optimization)
layer:
	@echo "📦 Creating Lambda Layer for dependencies..."
	mkdir -p layer/python
	pip install -t layer/python -r ../requirements-lambda.txt
	cd layer && zip -qr9 ../numpy-layer.zip .
	@echo "✅ Layer created: numpy-layer.zip"
	@echo "Upload this as a Lambda Layer to reduce deployment package size"

# Full optimized build and check
full: clean build-optimized check-size

# Help
help:
	@echo "AWS SAM Lambda Build Commands"
	@echo ""
	@echo "  make build           - Standard build"
	@echo "  make build-optimized - Build with size optimizations"
	@echo "  make check-size      - Analyze package size"
	@echo "  make deploy-dev      - Deploy to development"
	@echo "  make deploy-prod     - Deploy to production"
	@echo "  make layer           - Create Lambda Layer for dependencies"
	@echo "  make clean           - Remove build artifacts"
	@echo "  make full            - Clean, build optimized, and check size"