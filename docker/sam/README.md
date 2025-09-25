# Docker-based SAM Development Environment

This directory contains the Docker configuration for running AWS SAM CLI in a containerized environment with DynamoDB integration.

## 🚀 Quick Start

```bash
# Start the environment and initialize database
make -f Makefile.docker quick-start

# Start the SAM Local API
make -f Makefile.docker sam-api

# Access the API
curl http://localhost:3000/api/v1/health
```

## 📋 Prerequisites

- Docker Desktop installed and running
- Docker Compose v2.0+
- Make (optional, but recommended)

## 🏗️ Architecture

The Docker environment consists of:

1. **SAM Builder Container**: Amazon Linux 2023 with SAM CLI, AWS CLI, Python 3.12
2. **DynamoDB Local**: Local DynamoDB instance for data persistence
3. **DynamoDB Admin**: Web UI for viewing DynamoDB data
4. **Network Bridge**: Enables communication between containers

```
┌─────────────────────────────────────────────────────────┐
│                    Host Machine                          │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │ SAM Container│  │  DynamoDB    │  │  DynamoDB    │ │
│  │              │◄─┤   Local      │◄─┤   Admin UI   │ │
│  │ Port: 3000   │  │  Port: 8000  │  │  Port: 8001  │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
│         ▲                  ▲                 ▲          │
│         └──────────────────┴─────────────────┘          │
│                     sam-network                          │
└─────────────────────────────────────────────────────────┘
```

## 📦 Container Features

### SAM Builder Container
- **Base Image**: Amazon Linux 2023 (matches Lambda runtime)
- **Installed Tools**:
  - Python 3.12
  - Node.js 18
  - AWS CLI v2
  - AWS SAM CLI
  - uv (fast Python package manager)
- **Volume Mounts**:
  - `/workspace`: Your project directory
  - `/var/run/docker.sock`: Docker socket for Lambda containers
- **Exposed Ports**:
  - `3000`: SAM Local API Gateway
  - `3001`: SAM Local Lambda endpoint

### DynamoDB Configuration
- Persistent data storage in `./data/dynamodb`
- Shared database mode for simplified access
- Health checks for service readiness
- Admin UI for data visualization

## 🛠️ Usage

### Using Make Commands

```bash
# Environment Management
make -f Makefile.docker docker-up      # Start environment
make -f Makefile.docker docker-down    # Stop environment
make -f Makefile.docker docker-status  # Check status
make -f Makefile.docker docker-clean   # Clean everything

# SAM Operations
make -f Makefile.docker sam-build      # Build Lambda
make -f Makefile.docker sam-api        # Start API (port 3000)
make -f Makefile.docker sam-test       # Run tests

# Development
make -f Makefile.docker docker-shell   # Open container shell
make -f Makefile.docker docker-logs    # View logs
```

### Using Helper Script

```bash
# Make script executable
chmod +x scripts/sam-docker.sh

# Start environment
./scripts/sam-docker.sh start

# Start API
./scripts/sam-docker.sh api

# Open shell
./scripts/sam-docker.sh shell
```

### Manual Docker Commands

```bash
# Start services
docker-compose -f docker-compose.sam.yml up -d

# Execute SAM commands
docker-compose -f docker-compose.sam.yml exec sam-builder \
  bash -c "cd aws && sam build"

# Stop services
docker-compose -f docker-compose.sam.yml down
```

## 🧪 Testing

### Test API Endpoints

```bash
# Health check
curl http://localhost:3000/api/v1/health

# Process measurements
curl -X POST http://localhost:3000/api/v1/process/test-user \
  -H "Content-Type: application/json" \
  -d '{
    "measurements": [{
      "value": 75.5,
      "unit": "kg",
      "timestamp": "2024-01-01T10:00:00Z",
      "source": "patient-device"
    }]
  }'
```

### Run Unit Tests

```bash
make -f Makefile.docker sam-test
```

## 🔧 Configuration

### Environment Variables

The SAM container is pre-configured with:
- `AWS_ACCESS_KEY_ID=local`
- `AWS_SECRET_ACCESS_KEY=local`
- `DYNAMODB_ENDPOINT=http://dynamodb-local:8000`
- `DYNAMODB_TABLE_NAME=weight-processor-state`

### Custom Configuration

Edit `docker-compose.sam.yml` to modify:
- Port mappings
- Environment variables
- Volume mounts
- Network configuration

## 🐛 Troubleshooting

### Container Won't Start
```bash
# Check Docker is running
docker info

# Check for port conflicts
lsof -i :3000 -i :8000 -i :8001

# View detailed logs
docker-compose -f docker-compose.sam.yml logs
```

### DynamoDB Connection Issues
```bash
# Test DynamoDB connectivity from SAM container
docker-compose -f docker-compose.sam.yml exec sam-builder \
  curl http://dynamodb-local:8000

# Recreate network
docker network rm sam-network
docker-compose -f docker-compose.sam.yml up -d
```

### SAM Build Failures
```bash
# Clean and rebuild
make -f Makefile.docker docker-clean
make -f Makefile.docker docker-build
make -f Makefile.docker docker-up
```

## 📚 Additional Resources

- [AWS SAM CLI Documentation](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/sam-cli-command-reference.html)
- [DynamoDB Local Documentation](https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/DynamoDBLocal.html)
- [Docker Compose Documentation](https://docs.docker.com/compose/)

## 🔒 Security Notes

- The container runs with local AWS credentials for development only
- Never use these configurations in production
- Docker socket mounting is required for SAM to create Lambda containers
- All services are configured for local development only