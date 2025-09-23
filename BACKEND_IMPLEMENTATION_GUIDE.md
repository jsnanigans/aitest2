# Backend Implementation Guide

## Lambda Direct Invocation for Java Spring Boot

## Required Dependencies
- `software.amazon.awssdk:lambda:2.20.26`
- `software.amazon.awssdk:apache-client:2.20.26`

## Configuration

**application.yml:**
```yaml
aws:
  region: us-east-1
  lambda:
    weight-processor:
      function-name: weight-processor-${spring.profiles.active}
      timeout-seconds: 30
```

## Lambda Client Implementation

```java
@Service
@RequiredArgsConstructor
public class WeightProcessorLambdaClient {

    private final ObjectMapper objectMapper;
    private final LambdaClient lambdaClient;

    @Value("${aws.lambda.weight-processor.function-name}")
    private String functionName;

    public ProcessResponse processMeasurements(String userId,
                                              List<WeightMeasurement> measurements) {
        Map<String, Object> event = Map.of(
            "resource", "/api/v1/process/{userId}",
            "httpMethod", "POST",
            "pathParameters", Map.of("userId", userId),
            "body", objectMapper.writeValueAsString(Map.of(
                "measurements", measurements
            ))
        );

        InvokeRequest request = InvokeRequest.builder()
            .functionName(functionName)
            .invocationType(InvocationType.REQUEST_RESPONSE)
            .payload(SdkBytes.fromUtf8String(objectMapper.writeValueAsString(event)))
            .build();

        InvokeResponse response = lambdaClient.invoke(request);

        // Parse response
        Map<String, Object> lambdaResponse = objectMapper.readValue(
            response.payload().asUtf8String(), Map.class);
        String body = (String) lambdaResponse.get("body");
        return objectMapper.readValue(body, ProcessResponse.class);
    }
}
```

## Required DTOs

```java
@Data
@Builder
public class WeightMeasurement {
    private UUID uuid;
    private Double weight;
    private String unit;
    @JsonProperty("effectiveDateTime")
    private Instant effectiveDateTime;
    private String source;
}

@Data
public class ProcessResponse {
    private String status;
    @JsonProperty("processed_count")
    private int processedCount;
    @JsonProperty("accepted_count")
    private int acceptedCount;
    @JsonProperty("rejected_count")
    private int rejectedCount;
    private List<MeasurementResult> measurements;
}
```

## IAM Permissions

Your application's IAM role needs:
```json
{
    "Statement": [{
        "Effect": "Allow",
        "Action": "lambda:InvokeFunction",
        "Resource": "arn:aws:lambda:us-east-1:*:function:weight-processor-*"
    }]
}
```

## Lambda Configuration

```java
@Configuration
public class LambdaConfig {

    @Bean
    public LambdaClient lambdaClient(@Value("${aws.region}") String region) {
        return LambdaClient.builder()
            .region(Region.of(region))
            .credentialsProvider(DefaultCredentialsProvider.create())
            .build();
    }
}
```

## Service Layer Example

```java
@Service
@RequiredArgsConstructor
public class WeightService {

    private final WeightProcessorLambdaClient lambdaClient;

    @Transactional
    public void processUserWeights(String userId, List<WeightEntity> weights) {
        List<WeightMeasurement> measurements = weights.stream()
            .map(this::toMeasurement)
            .toList();

        ProcessResponse response = lambdaClient.processMeasurements(userId, measurements);

        // Update your database with response.getMeasurements() results
    }
}
```

## Error Handling

```java
@Component
public class LambdaRetryConfig {

    @Bean
    public RetryTemplate lambdaRetryTemplate() {
        RetryTemplate template = new RetryTemplate();
        template.setRetryPolicy(new SimpleRetryPolicy(3));
        template.setBackOffPolicy(new ExponentialBackOffPolicy());
        return template;
    }
}

// Usage
retryTemplate.execute(context ->
    lambdaClient.processMeasurements(userId, measurements));
```

## Available Endpoints

All Lambda functions accept events formatted as API Gateway proxy events:

- **Process**: `/api/v1/process/{userId}` - Process weight measurements
- **Replay**: `/api/v1/replay/{userId}` - Replay from timestamp
- **Health**: `/api/v1/health` - Check service health

## Local Testing

For local development, run Lambda in Docker:
```bash
# In Lambda project
make docker-run  # Runs on localhost:5448
```

Then configure Spring Boot to use local endpoint:
```yaml
# application-local.yml
aws:
  lambda:
    weight-processor:
      use-local-endpoint: true
      local-endpoint: http://localhost:5448
```

## Production Deployment

Lambda function names by environment:
- Development: `weight-processor-dev`
- Staging: `weight-processor-staging`
- Production: `weight-processor-prod`

No API Gateway in production - only direct Lambda invocation.

## Monitoring

CloudWatch Log Groups:
- `/aws/lambda/weight-processor-{env}`

Key Metrics:
- Invocations, Errors, Duration, Throttles

## Common Issues

**AccessDeniedException**: Check IAM role has `lambda:InvokeFunction` permission

**Task timed out**: Increase `timeout-seconds` in config or optimize payload size

**ResourceNotFoundException**: Verify function name matches environment

**High latency**: Consider enabling Lambda provisioned concurrency