# Investigation: AWS Refactoring Implementation Issues

## Bottom Line

**Root Cause**: Template.yaml references a non-existent API key resource `WeightProcessorApiApiKey`
**Fix Location**: `template.yaml:215`
**Confidence**: High

## What's Happening

The AWS SAM template has a critical reference error where the Outputs section tries to export an API Key ID that was never created. Additionally, there are several minor issues with imports and potential runtime problems.

## Why It Happens

**Primary Cause**: Incomplete API Gateway configuration
**Trigger**: `template.yaml:215` - References `!Ref WeightProcessorApiApiKey`
**Decision Point**: `template.yaml:39-49` - Auth section creates usage plan but no explicit API key

## Evidence

- **Key File**: `template.yaml:215` - Invalid reference to undefined resource
- **Search Used**: Manual inspection of template.yaml
- **Missing Resource**: No `WeightProcessorApiApiKey` resource defined anywhere in template

## Additional Issues Found

### 1. Missing API Key Resource
The template enables API key authentication but doesn't create the actual API key resource. Need to add:
```yaml
WeightProcessorApiApiKey:
  Type: AWS::Serverless::ApiKey
  Properties:
    Name: !Sub weight-processor-key-${Environment}
    StageKeys:
      - RestApiId: !Ref WeightProcessorApi
        StageName: !Ref Environment
```

### 2. Requirements File Mismatch
- `template.yaml:61` references `requirements-lambda.txt`
- No such file exists in the repository
- Should either create this file or reference `requirements.txt`

### 3. DynamoDB Serialization Edge Cases
- `dynamodb_store.py:265` - Recursive serialization might fail on circular references
- No max recursion depth protection

### 4. Lambda Handler numpy Import Issue
- `lambda_handler.py:148` imports numpy inside function
- This should be at module level for better performance
- Numpy might not be in Lambda runtime without layer

### 5. CSV Processor Threading Issue
- `csv_processor.py:494` imports visualization module in subprocess
- Could fail if module dependencies aren't process-safe

## Next Steps

1. Add WeightProcessorApiApiKey resource to template.yaml
2. Create requirements-lambda.txt or update template reference
3. Add numpy to Lambda layer or requirements
4. Test DynamoDB serialization with complex nested states
5. Verify visualization module is process-safe

## Risks

- Deployment will fail due to invalid CloudFormation reference
- Runtime failures if numpy not available in Lambda
- Potential data corruption if circular references in state
