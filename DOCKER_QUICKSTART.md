# 🚀 Docker SAM Quick Start Guide

## One-Time Setup
```bash
# Test that everything works
./scripts/test-docker-sam.sh
```

## Daily Development Workflow

### 1️⃣ Start Environment
```bash
make -f Makefile.docker docker-up
```

### 2️⃣ Initialize Database (first time only)
```bash
make -f Makefile.docker docker-init-db
```

### 3️⃣ Start SAM API
```bash
make -f Makefile.docker sam-api
```
API available at: http://localhost:3000

### 4️⃣ View DynamoDB Data
```bash
open http://localhost:8001
```

### 5️⃣ Work in Container
```bash
make -f Makefile.docker docker-shell
```

## Common Commands

| Command | Description |
|---------|------------|
| `make -f Makefile.docker docker-status` | Check container status |
| `make -f Makefile.docker docker-logs` | View logs |
| `make -f Makefile.docker sam-build` | Build Lambda function |
| `make -f Makefile.docker sam-test` | Run tests |
| `make -f Makefile.docker docker-down` | Stop environment |
| `make -f Makefile.docker docker-clean` | Clean everything |

## Test API Endpoints

### Health Check
```bash
curl http://localhost:3000/api/v1/health
```

### Process Measurement
```bash
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

## Troubleshooting

### Port Already in Use
```bash
# Find what's using the port
lsof -i :3000
# Kill the process or change the port in docker-compose.sam.yml
```

### Container Won't Start
```bash
# Check logs
docker-compose -f docker-compose.sam.yml logs sam-builder

# Rebuild from scratch
make -f Makefile.docker docker-clean
make -f Makefile.docker docker-build
make -f Makefile.docker docker-up
```

### Can't Connect to DynamoDB
```bash
# Check DynamoDB is running
docker-compose -f docker-compose.sam.yml ps dynamodb-local

# Test from container
docker-compose -f docker-compose.sam.yml exec sam-builder \
  curl http://dynamodb-local:8000
```

## Architecture

```
Your Machine
├── http://localhost:3000  → SAM Local API
├── http://localhost:8000  → DynamoDB
├── http://localhost:8001  → DynamoDB Admin UI
└── Docker Network (sam-network)
    ├── sam-builder container
    ├── dynamodb-local container
    └── dynamodb-admin container
```

## Next Steps

- Read the full documentation: `docker/sam/README.md`
- Customize environment: Edit `docker-compose.sam.yml`
- Add more AWS services: Enable LocalStack profile