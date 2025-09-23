# Production Deployment Guide

## Lambda-Only Architecture for Production

This guide covers deploying the Weight Processor as a Lambda-only service (no API Gateway) for production use by your Java Spring Boot backend.

## Architecture

```
┌────────────────┐                    ┌─────────────────┐
│  Spring Boot   │─────────────────►  │     Lambda      │
│   Backend      │  Direct Invoke     │  Weight Processor│
└────────────────┘  (via IAM role)    └─────────────────┘
                                               │
                                               ▼
                                       ┌─────────────────┐
                                       │    DynamoDB     │
                                       │   State Store   │
                                       └─────────────────┘
```

## Prerequisites

1. **AWS CLI configured** with appropriate credentials
2. **SAM CLI installed** (`brew install aws-sam-cli`)
3. **Docker installed** for building Lambda packages
4. **IAM permissions** to create Lambda functions and DynamoDB tables

## Deployment Steps

### 1. Build the Lambda Package

```bash
# Build using the production template (no API Gateway)
make build-prod
```

### 2. Deploy to Production

```bash
# Deploy Lambda function and DynamoDB table
make deploy-prod

# Or manually:
sam deploy \
  --template template-prod.yaml \
  --stack-name weight-processor-prod \
  --parameter-overrides Environment=prod \
  --capabilities CAPABILITY_IAM \
  --confirm-changeset
```

### 3. Configure Spring Boot IAM Role

Your Spring Boot application's IAM role needs permission to invoke the Lambda:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": "lambda:InvokeFunction",
      "Resource": "arn:aws:lambda:us-east-1:YOUR_ACCOUNT:function:weight-processor-prod"
    }
  ]
}
```

Attach this policy to your:
- **EC2 instance role** (if running on EC2)
- **ECS task role** (if running on ECS)
- **EKS service account** (if running on EKS)

### 4. Update Spring Boot Configuration

**application-prod.yml:**
```yaml
aws:
  region: us-east-1
  lambda:
    weight-processor:
      function-name: weight-processor-prod
      timeout-seconds: 30
      max-retries: 3
```

### 5. Verify Deployment

Test the Lambda invocation from your Spring Boot application:

```java
// In your application
@Autowired
private WeightProcessorLambdaClient client;

// Test health check
HealthResponse health = client.checkHealth();
assert "healthy".equals(health.getStatus());
```

## Production Optimizations

### 1. Enable Provisioned Concurrency (Optional)

For consistent performance without cold starts:

```yaml
# Add to template-prod.yaml
ProvisionedConcurrencyConfig:
  ProvisionedConcurrentExecutions: 5
```

### 2. Configure Reserved Concurrency

Limit maximum concurrent executions:

```yaml
ReservedConcurrentExecutions: 100  # Already in template
```

### 3. VPC Configuration (If Required)

If your Lambda needs to access resources in a VPC:

```yaml
VpcConfig:
  SecurityGroupIds:
    - !Ref LambdaSecurityGroup
  SubnetIds:
    - !Ref PrivateSubnet1
    - !Ref PrivateSubnet2
```

### 4. Enable X-Ray Tracing

For distributed tracing:

```yaml
TracingConfig:
  Mode: Active
```

## Monitoring

### CloudWatch Metrics

Monitor these key metrics:
- **Invocations**: Total Lambda invocations
- **Errors**: Failed invocations
- **Duration**: Execution time
- **Throttles**: Rate limit hits
- **ConcurrentExecutions**: Active executions

### CloudWatch Alarms (Already Configured)

The template includes alarms for:
- Error rate > 10 errors in 5 minutes
- Throttling > 5 throttles in 5 minutes
- DynamoDB throttling

### Logs

View Lambda logs:
```bash
# Tail logs
sam logs -n weight-processor-prod --tail

# Or use AWS CLI
aws logs tail /aws/lambda/weight-processor-prod --follow
```

## Cost Optimization

### Estimated Monthly Costs

For 1 million invocations/month with 500ms average duration:
- **Lambda**: ~$20
- **DynamoDB**: ~$25 (on-demand)
- **CloudWatch Logs**: ~$5
- **Total**: ~$50/month

### Cost Saving Tips

1. **Use ARM architecture** (Graviton2):
   ```yaml
   Architectures:
     - arm64
   ```
   20% cheaper than x86

2. **Optimize memory allocation**:
   - Start with 1024 MB
   - Use AWS Lambda Power Tuning to find optimal setting

3. **Set log retention**:
   ```yaml
   RetentionInDays: 30  # Already configured
   ```

## Rollback Strategy

If issues occur:

```bash
# List stack versions
aws cloudformation list-stack-resources \
  --stack-name weight-processor-prod

# Rollback to previous version
aws cloudformation cancel-update-stack \
  --stack-name weight-processor-prod

# Or redeploy previous version
git checkout <previous-commit>
make deploy-prod
```

## Security Checklist

- [ ] Lambda function has minimal IAM permissions
- [ ] DynamoDB encryption at rest enabled
- [ ] CloudWatch Logs encrypted
- [ ] No hardcoded secrets (use environment variables)
- [ ] VPC security groups configured (if using VPC)
- [ ] Point-in-time recovery enabled for DynamoDB
- [ ] Java backend uses IAM role, not access keys

## Troubleshooting

### Lambda Not Found
```
Error: ResourceNotFoundException
```
**Solution**: Verify function name and region match your configuration

### Permission Denied
```
Error: AccessDeniedException
```
**Solution**: Check IAM role has `lambda:InvokeFunction` permission

### Timeout Errors
```
Error: Task timed out after 30.00 seconds
```
**Solution**: Increase timeout in template or optimize code

### High Latency
**Solutions**:
- Enable provisioned concurrency
- Increase Lambda memory
- Check for VPC cold starts
- Optimize DynamoDB queries

## Support

- **Lambda Logs**: CloudWatch Logs Group `/aws/lambda/weight-processor-prod`
- **Metrics**: CloudWatch Metrics Dashboard
- **Alarms**: CloudWatch Alarms for error/throttle alerts