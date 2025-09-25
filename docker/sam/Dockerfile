# Use Amazon Linux 2023 as base (same as Lambda runtime environment)
FROM amazonlinux:2023

# Install system dependencies
RUN dnf update -y && \
    dnf install -y \
    python3.12 \
    python3.12-pip \
    python3.12-devel \
    gcc \
    gcc-c++ \
    make \
    curl \
    unzip \
    tar \
    gzip \
    git \
    which \
    shadow-utils \
    docker \
    && dnf clean all

# Install Node.js 18 (required for SAM CLI)
RUN curl -fsSL https://rpm.nodesource.com/setup_18.x | bash - && \
    dnf install -y nodejs

# Install AWS CLI v2
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && \
    ./aws/install && \
    rm -rf awscliv2.zip aws/

# Install AWS SAM CLI
RUN python3.12 -m pip install --upgrade pip && \
    python3.12 -m pip install aws-sam-cli

# Install uv for faster Python package management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:${PATH}"

# Create working directory
WORKDIR /workspace

# Set environment variables for local development
ENV AWS_ACCESS_KEY_ID=local \
    AWS_SECRET_ACCESS_KEY=local \
    AWS_DEFAULT_REGION=us-east-1 \
    AWS_SESSION_TOKEN=local \
    SAM_CLI_TELEMETRY=0 \
    DOCKER_HOST=unix:///var/run/docker.sock \
    PYTHONPATH=/workspace:${PYTHONPATH}

# Create entrypoint script
COPY <<'EOF' /usr/local/bin/entrypoint.sh
#!/bin/bash
set -e

# Function to wait for DynamoDB
wait_for_dynamodb() {
    echo "Waiting for DynamoDB to be ready..."
    for i in {1..30}; do
        if curl -s http://dynamodb-local:8000 > /dev/null 2>&1; then
            echo "✓ DynamoDB is ready"
            return 0
        fi
        echo -n "."
        sleep 1
    done
    echo ""
    echo "⚠ Warning: DynamoDB might not be ready"
    return 1
}

# If DynamoDB endpoint is set, wait for it
if [ ! -z "$DYNAMODB_ENDPOINT" ]; then
    wait_for_dynamodb
fi

# Execute the command
exec "$@"
EOF

RUN chmod +x /usr/local/bin/entrypoint.sh

# Expose ports for SAM Local API
EXPOSE 3000 3001

# Set entrypoint
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# Default command
CMD ["/bin/bash"]