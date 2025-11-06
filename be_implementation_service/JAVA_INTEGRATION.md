# Java App Integration Guide

## Security Architecture

The weight-processor Lambda is secured with:
1. **Resource-based policy** - Only the `LambdaInvokerRole` can invoke it
2. **ExternalId requirement** - Prevents confused deputy attacks
3. **VPC isolation** - Lambda runs in private subnets with no internet access

## Prerequisites

Your Java app needs AWS credentials (IAM user/role) that can assume the `weight-processor-invoker-{env}` role.

## Setup Steps

### 1. Grant Your Java App Permission to Assume the Role

Add this policy to your Java app's IAM user/role:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": "sts:AssumeRole",
      "Resource": "arn:aws:iam::387257169268:role/weight-processor-invoker-dev-us"
    }
  ]
}
```

### 2. Get Stack Outputs

After deployment, get the required values:

```bash
aws cloudformation describe-stacks \
  --stack-name weight-processor-dev \
  --region us-east-1 \
  --query 'Stacks[0].Outputs'
```

You need:
- `InvokerRoleArn` - The role to assume (e.g., `arn:aws:iam::387257169268:role/weight-processor-invoker-dev-us`)
- `ExternalId` - Required for AssumeRole (e.g., `weight-processor-dev-us`)
- `FunctionName` - Lambda function name (e.g., `weight-processor-dev-us`)

### 3. Java Code Example (AWS SDK v2)

```java
import software.amazon.awssdk.auth.credentials.AwsCredentialsProvider;
import software.amazon.awssdk.auth.credentials.StaticCredentialsProvider;
import software.amazon.awssdk.core.SdkBytes;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.lambda.LambdaClient;
import software.amazon.awssdk.services.lambda.model.InvokeRequest;
import software.amazon.awssdk.services.lambda.model.InvokeResponse;
import software.amazon.awssdk.services.sts.StsClient;
import software.amazon.awssdk.services.sts.model.AssumeRoleRequest;
import software.amazon.awssdk.services.sts.model.AssumeRoleResponse;
import software.amazon.awssdk.services.sts.model.Credentials;

public class WeightProcessorClient {

    private static final String INVOKER_ROLE_ARN = "arn:aws:iam::387257169268:role/weight-processor-invoker-dev-us";
    private static final String EXTERNAL_ID = "weight-processor-dev-us";
    private static final String FUNCTION_NAME = "weight-processor-dev-us";
    private static final Region REGION = Region.US_EAST_1;

    /**
     * Assume the invoker role and get temporary credentials
     */
    private Credentials assumeInvokerRole() {
        try (StsClient stsClient = StsClient.builder()
                .region(REGION)
                .build()) {

            AssumeRoleRequest assumeRoleRequest = AssumeRoleRequest.builder()
                .roleArn(INVOKER_ROLE_ARN)
                .roleSessionName("weight-processor-session-" + System.currentTimeMillis())
                .externalId(EXTERNAL_ID)
                .durationSeconds(3600) // 1 hour
                .build();

            AssumeRoleResponse response = stsClient.assumeRole(assumeRoleRequest);
            return response.credentials();
        }
    }

    /**
     * Invoke the weight processor Lambda
     */
    public String processWeight(String userId, String measurementJson) {
        // Step 1: Assume the invoker role
        Credentials credentials = assumeInvokerRole();

        // Step 2: Create Lambda client with temporary credentials
        AwsCredentialsProvider credentialsProvider = StaticCredentialsProvider.create(
            software.amazon.awssdk.auth.credentials.AwsSessionCredentials.create(
                credentials.accessKeyId(),
                credentials.secretAccessKey(),
                credentials.sessionToken()
            )
        );

        try (LambdaClient lambdaClient = LambdaClient.builder()
                .region(REGION)
                .credentialsProvider(credentialsProvider)
                .build()) {

            // Step 3: Build the payload
            String payload = String.format("""
                {
                  "action": "process",
                  "user_id": "%s",
                  "body": %s
                }
                """, userId, measurementJson);

            // Step 4: Invoke Lambda
            InvokeRequest invokeRequest = InvokeRequest.builder()
                .functionName(FUNCTION_NAME)
                .payload(SdkBytes.fromUtf8String(payload))
                .build();

            InvokeResponse response = lambdaClient.invoke(invokeRequest);

            // Step 5: Return response
            return response.payload().asUtf8String();
        }
    }

    /**
     * Example usage
     */
    public static void main(String[] args) {
        WeightProcessorClient client = new WeightProcessorClient();

        String measurementJson = """
            {
              "measurements": [{
                "uuid": "measurement-001",
                "weight": 185.5,
                "unit": "lb",
                "effectiveDateTime": "2025-10-02T10:00:00Z",
                "source": "smart_scale"
              }]
            }
            """;

        String result = client.processWeight("user-123", measurementJson);
        System.out.println("Response: " + result);
    }
}
```

## Payload Format

### Process Measurements
```json
{
  "action": "process",
  "user_id": "user-123",
  "body": {
    "measurements": [{
      "uuid": "measurement-001",
      "weight": 185.5,
      "unit": "lb",
      "effectiveDateTime": "2025-10-02T10:00:00Z",
      "source": "smart_scale"
    }],
    "user_height_m": 1.75
  }
}
```

### Get User State
```json
{
  "action": "get_state",
  "user_id": "user-123"
}
```

### Delete User State
```json
{
  "action": "delete_state",
  "user_id": "user-123"
}
```

### Cleanup (Reset Adaptive Parameters)
```json
{
  "action": "cleanup",
  "user_id": "user-123",
  "body": {
    "cleanup_type": "reset_adaptive"
  }
}
```

**Note:** Currently `reset_adaptive` deletes entire state (known issue). Use `clear_all` for complete reset.

## Response Format

All responses follow this structure:

**Success:**
```json
{
  "success": true,
  "data": { ... },
  "error": null,
  "meta": {
    "timestamp": "2025-10-02 10:00:00.123456",
    "version": "2.0.0",
    "request_id": "req_abc123def456"
  }
}
```

**Error:**
```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid request format",
    "field": "measurements",
    "details": null,
    "suggestion": "Check that all required fields are present and valid",
    "documentation": "https://api.docs/errors#validation_error"
  },
  "meta": {
    "timestamp": "2025-10-02 10:00:00.123456",
    "version": "2.0.0",
    "request_id": "req_abc123def456"
  }
}
```

## Security Notes

1. **ExternalId is required** - This prevents the confused deputy problem
2. **Credentials are temporary** - They expire after 1 hour (configurable)
3. **Cache the credentials** - Don't assume role on every request; reuse until expiration
4. **Only this role can invoke** - No other IAM principal can access the Lambda
5. **PII handling** - All logs are sanitized; user_id and weights are masked

## Troubleshooting

### "Access Denied" when assuming role
- Verify your Java app's IAM user/role has `sts:AssumeRole` permission for this role
- Check the ExternalId matches exactly

### "User is not authorized to perform: lambda:InvokeFunction"
- You forgot to assume the role first
- The temporary credentials expired

### Lambda timeout
- Check CloudWatch logs: `/aws/lambda/weight-processor-dev-us`
- VPC connectivity issues should be resolved (see DEPLOYMENT_USAGE.md)

## Cost Optimization

- **Cache STS credentials** for the full duration (up to 12 hours)
- **Reuse Lambda client** instances when possible
- **Batch measurements** when you have multiple for the same user

## Testing Integration

### Verify Setup

Before integrating, verify the Lambda is accessible:

```bash
# Test with the provided test script
./test-deployment.sh

# Or manually test assuming the role
aws sts assume-role \
  --role-arn arn:aws:iam::387257169268:role/weight-processor-invoker-dev-us \
  --role-session-name test-session \
  --external-id weight-processor-dev-us \
  --region us-east-1
```

### Unit Test Example

```java
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class WeightProcessorClientTest {

    @Test
    public void testHealthCheck() {
        WeightProcessorClient client = new WeightProcessorClient();

        String payload = "{\"action\": \"health\"}";
        String response = client.invokeFunction(payload);

        // Parse and verify response
        assertTrue(response.contains("\"success\": true"));
        assertTrue(response.contains("\"status\": \"healthy\""));
    }

    @Test
    public void testProcessMeasurement() {
        WeightProcessorClient client = new WeightProcessorClient();

        String measurementJson = """
            {
              "measurements": [{
                "uuid": "test-001",
                "weight": 185.5,
                "unit": "lb",
                "effectiveDateTime": "2025-10-02T10:00:00Z",
                "source": "smart_scale"
              }]
            }
            """;

        String response = client.processWeight("test-user-123", measurementJson);

        // Verify success
        assertTrue(response.contains("\"success\": true"));
        assertTrue(response.contains("\"measurements_processed\": 1"));
    }
}
```

### Check Logs

After invoking from Java:

```bash
# View recent logs
aws logs tail /aws/lambda/weight-processor-dev-us \
  --since 5m \
  --region us-east-1
```
