# Weight Processor Infrastructure Alignment Analysis

**Date:** 2025-10-06
**Scope:** Comparison of weight-processor configuration with main backend infrastructure patterns

## Executive Summary

This analysis compares the weight-processor SAM configuration against established patterns in `../infrastructure/*` to identify opportunities for improved consistency, maintainability, and compliance.

### Key Findings

1. **Monitoring Topic Reference:** Using hardcoded placeholder ARNs instead of CloudFormation exports
2. **Tagging Inconsistencies:** Missing Vanta compliance tags required for security compliance
3. **Naming Convention Gaps:** Environment naming doesn't fully align with main stack patterns
4. **Alarm Configuration:** Non-standard alarm thresholds compared to other Lambda functions
5. **Log Retention:** Hardcoded 30-day retention vs environment-specific patterns

---

## Current State vs. Infrastructure Patterns

### 1. Monitoring Alarm Topic Configuration

**Current State (samconfig.toml:22, 37, 52):**
```toml
parameter_overrides = "Environment=\"prod-us\" MonitoringAlarmNotificationTopic=\"arn:aws:sns:us-east-1:ACCOUNT_ID:TOPIC_NAME\""
```

**Infrastructure Pattern:**
- Topic is created in `LoggingStack` and exported as `${AWS::StackName}-MonitoringAlarmNotificationTopic`
- Other stacks reference via CloudFormation parameters and imports
- Actual ARN format: `arn:aws:sns:us-east-1:{AccountId}:MonitoringAlarmNotificationTopic-{StackName}`

**Example from careassistant/stack.template.yaml:7:**
```yaml
Parameters:
  MonitoringAlarmNotificationTopic:
    Description: The SNS topic for sending monitoring alarm notifications to.
    Type: String
```

**Exported Values Available:**
```
BackendDevUs-LoggingStack-NZRIHZGA2RI4-MonitoringAlarmNotificationTopic
  → arn:aws:sns:us-east-1:387257169268:MonitoringAlarmNotificationTopic-BackendDevUs-LoggingStack-NZRIHZGA2RI4
BackendQaUs-LoggingStack-MYC4XN0B0U4C-MonitoringAlarmNotificationTopic
  → arn:aws:sns:us-east-1:387257169268:MonitoringAlarmNotificationTopic-BackendQaUs-LoggingStack-MYC4XN0B0U4C
```

**Recommendation:**
- Update `sam-template-prod.yaml` to use `Fn::ImportValue` to reference the exported topic
- Update `samconfig.toml` to remove placeholder ARN from parameter_overrides
- Reference: infrastructure/root.template.yaml:66

---

### 2. Tagging Standards

**Current State (sam-template-prod.yaml:101-106):**
```yaml
Tags:
  Environment: !Ref Environment
  Service: weight-processor
  CostCenter: health-services
  DataClassification: PII
  ManagedBy: SAM
```

**Infrastructure Pattern (common.config.yaml:9-12):**
```yaml
tags:
  VantaOwner: bernhard.schandl@9am.health
  VantaContainsUserData: "true"
  VantaContainsEPHI: "true"
  VantaDescription: Backend infrastructure for 9amHealth services
```

**Environment-Specific (prod-us-east.config.yaml:9, dev-us.config.yaml:9):**
```yaml
tags:
  VantaNonProd: "false"  # prod
  VantaNonProd: "true"   # dev/qa
```

**Missing Tags:**
- `VantaOwner`: Required for compliance tracking
- `VantaContainsUserData`: Should be "true" (processes weight data)
- `VantaContainsEPHI`: Should be "true" (health information)
- `VantaDescription`: Descriptive text for auditing
- `VantaNonProd`: Environment classification
- Custom tag: `nineamhealth:backend:environment` (used in infrastructure)

**Recommendation:**
- Add all Vanta compliance tags to Lambda function and DynamoDB table
- Make `VantaNonProd` environment-specific via parameters
- Reference: infrastructure/common.config.yaml, infrastructure/prod-us-east.config.yaml

---

### 3. Naming Conventions

**Current State:**
- Stack names: `weight-processor-prod`, `weight-processor-dev`, `weight-processor-staging`
- Environment values: `prod-us`, `dev-us`, `qa-us`
- Function names: `weight-processor-${Environment}`

**Infrastructure Pattern:**
- Stack names: `BackendProdUsEast`, `BackendDevUs`, `BackendQaUs`
- Full environment identifiers in stack names, not parameters
- Consistent `{Service}{Environment}` pattern

**Observations:**
- `qa-us` in infrastructure vs. `staging` in weight-processor naming
- `BackendProdUsEast` vs `prod-us` - region included in stack name for multi-region prod

**Recommendation:**
- Consider renaming staging environment to `qa-us` for consistency
- Update stack names to follow pattern: `WeightProcessor{Environment}` (e.g., `WeightProcessorProdUsEast`)
- Align with infrastructure's environment naming: use `BackendProdUsEast` instead of just `prod-us`

---

### 4. CloudWatch Alarm Configuration

**Current State (sam-template-prod.yaml:156-173):**
```yaml
ProcessorErrorAlarm:
  Type: AWS::CloudWatch::Alarm
  Properties:
    Period: 300
    EvaluationPeriods: 1
    Threshold: 10
```

**Infrastructure Pattern (utils.include.yaml:9-27):**
```yaml
LambdaErrorAlarm:
  Type: AWS::CloudWatch::Alarm
  Properties:
    Threshold: 0
    EvaluationPeriods: 1
    Period: 60
    TreatMissingData: missing
```

**Differences:**
- Standard pattern: Alarm on ANY error (threshold: 0), 1-minute period
- Weight-processor: Alarm on 10+ errors, 5-minute period
- Standard uses `TreatMissingData: missing` vs `notBreaching`

**Recommendation:**
- Align with standard pattern: threshold 0, period 60 for faster alerting
- Consider creating a utility macro library for weight-processor if more alarms are needed
- Use `TreatMissingData: missing` for consistency

---

### 5. CloudWatch Logs Retention

**Current State (sam-template-prod.yaml:153):**
```yaml
ProcessorLogGroup:
  Type: AWS::Logs::LogGroup
  Properties:
    RetentionInDays: 30
```

**Infrastructure Pattern (prod-us-east.config.yaml:31, dev-us.config.yaml:31):**
```yaml
# Production
CloudwatchLogRetentionDays: 3653  # 10 years

# Development
CloudwatchLogRetentionDays: 30
```

**Recommendation:**
- Make log retention environment-specific via parameters
- Production: 3653 days (10 years) for compliance
- Dev/QA: 30 days to reduce costs
- Reference: infrastructure/prod-us-east.config.yaml:31

---

### 6. Stack Name Alignment

**Current State (samconfig.toml:5, 27, 42):**
```toml
[default.global.parameters]
stack_name = "weight-processor-prod"

[dev.global.parameters]
stack_name = "weight-processor-dev"

[staging.global.parameters]
stack_name = "weight-processor-staging"
```

**Infrastructure Pattern:**
- Stack names are set in environment config files
- Pattern: `{Service}{Environment}` without hyphens
- Examples: `BackendProdUsEast`, `BackendDevUs`, `BackendQaUs`

**Recommendation:**
- Update to: `WeightProcessorProdUsEast`, `WeightProcessorDevUs`, `WeightProcessorQaUs`
- This aligns with infrastructure naming and improves CloudFormation stack organization
- Makes it clear which environment and region (for future multi-region support)

---

### 7. Environment Parameter Values

**Current State (samconfig.toml:22):**
```toml
parameter_overrides = "Environment=\"prod-us\" ..."
```

**Infrastructure Pattern:**
- Full environment names: `BackendProdUsEast`, `BackendDevUs`, `BackendQaUs`
- Includes region information: `ProdUsEast` vs just `prod-us`

**Recommendation:**
- Consider if region should be part of environment identifier for future multi-region support
- Current `prod-us`, `dev-us`, `qa-us` is acceptable if single-region is guaranteed
- Document decision in DEPLOYMENT_USAGE.md

---

## Priority Recommendations

### High Priority (Security & Compliance)

1. **Add Vanta Compliance Tags** (sam-template-prod.yaml:101-106)
   - Add: `VantaOwner`, `VantaContainsUserData`, `VantaContainsEPHI`, `VantaDescription`, `VantaNonProd`
   - Apply to all resources: Lambda function, DynamoDB table, Log groups

2. **Update Monitoring Topic Reference** (sam-template-prod.yaml, samconfig.toml)
   - Use `Fn::ImportValue` to reference `{BackendStackName}-LoggingStack-*-MonitoringAlarmNotificationTopic`
   - Remove hardcoded ARN placeholders
   - Add mapping for BackendStackName like line 20-26 of sam-template-prod.yaml

### Medium Priority (Operational Excellence)

3. **Align CloudWatch Alarm Configuration** (sam-template-prod.yaml:156-173)
   - Change threshold from 10 to 0
   - Change period from 300 to 60
   - Change `TreatMissingData` from `notBreaching` to `missing`

4. **Environment-Specific Log Retention** (sam-template-prod.yaml:153)
   - Add `LogRetentionDays` parameter
   - Set to 3653 for prod, 30 for dev/qa
   - Update samconfig.toml to pass this parameter

### Low Priority (Consistency)

5. **Rename Staging to QA** (samconfig.toml:40-52)
   - Update `[staging]` section to `[qa]` or keep both for backwards compatibility
   - Update stack name to `weight-processor-qa` or `WeightProcessorQaUs`

6. **Consider Stack Name Pattern** (samconfig.toml)
   - Evaluate changing to `WeightProcessor{Environment}` pattern
   - Document decision if keeping current pattern

---

## Implementation Examples

### Example 1: Update MonitoringAlarmNotificationTopic Reference

**In sam-template-prod.yaml:**

```yaml
# Add to Parameters section (after line 18)
Parameters:
  # ... existing parameters ...

  LogRetentionDays:
    Type: Number
    Default: 30
    AllowedValues:
      - 30
      - 3653
    Description: CloudWatch Logs retention in days

# Update Resources section
Resources:
  # Update alarm (line 162)
  ProcessorErrorAlarm:
    Type: AWS::CloudWatch::Alarm
    Properties:
      AlarmName: !Sub weight-processor-errors-${Environment}
      AlarmDescription: Alert when processor function errors exceed threshold
      AlarmActions:
        - Fn::ImportValue: !Sub
            - "${BackendStackName}-LoggingStack-MonitoringAlarmNotificationTopic"
            - BackendStackName: !FindInMap [EnvironmentConfig, !Ref Environment, BackendStackName]
      # ... rest of properties ...

  # Update log group (line 153)
  ProcessorLogGroup:
    Type: AWS::Logs::LogGroup
    Properties:
      LogGroupName: !Sub /aws/lambda/weight-processor-${Environment}
      RetentionInDays: !Ref LogRetentionDays
```

**In samconfig.toml:**

```toml
# Update prod (line 22)
parameter_overrides = "Environment=\"prod-us\" LogRetentionDays=3653"

# Update dev (line 37)
parameter_overrides = "Environment=\"dev-us\" LogRetentionDays=30"

# Update staging/qa (line 52)
parameter_overrides = "Environment=\"qa-us\" LogRetentionDays=30"
```

### Example 2: Add Vanta Compliance Tags

**In sam-template-prod.yaml (line 5):**

```yaml
Parameters:
  # ... existing parameters ...

  VantaOwner:
    Type: String
    Default: "bernhard.schandl@9am.health"
    Description: Vanta compliance - owner email

  VantaNonProd:
    Type: String
    AllowedValues:
      - "true"
      - "false"
    Description: Vanta compliance - non-production flag
```

**Update Function Tags (line 101):**

```yaml
Tags:
  Environment: !Ref Environment
  Service: weight-processor
  CostCenter: health-services
  DataClassification: PII
  ManagedBy: SAM
  VantaOwner: !Ref VantaOwner
  VantaContainsUserData: "true"
  VantaContainsEPHI: "true"
  VantaDescription: "Weight measurement processing with Kalman filtering"
  VantaNonProd: !Ref VantaNonProd
  nineamhealth:backend:environment: !Ref Environment
```

**Update Table Tags (line 143):**

```yaml
Tags:
  - Key: Environment
    Value: !Ref Environment
  - Key: Service
    Value: weight-processor
  - Key: VantaOwner
    Value: !Ref VantaOwner
  - Key: VantaContainsUserData
    Value: "true"
  - Key: VantaContainsEPHI
    Value: "true"
  - Key: VantaDescription
    Value: "State storage for weight processor Kalman filters"
  - Key: VantaNonProd
    Value: !Ref VantaNonProd
```

**In samconfig.toml:**

```toml
# Prod (line 22)
parameter_overrides = "Environment=\"prod-us\" LogRetentionDays=3653 VantaNonProd=\"false\""

# Dev (line 37)
parameter_overrides = "Environment=\"dev-us\" LogRetentionDays=30 VantaNonProd=\"true\""

# Staging (line 52)
parameter_overrides = "Environment=\"qa-us\" LogRetentionDays=30 VantaNonProd=\"true\""
```

---

## Reference Files

- Infrastructure patterns: `../infrastructure/`
  - Monitoring: `logging/monitoring.template.yaml`
  - Lambda example: `careassistant/function.template.yaml`
  - Common config: `common.config.yaml`
  - Environment configs: `{env}.config.yaml`
  - Utilities: `utils.include.yaml`

- Current weight-processor files:
  - Template: `sam-template-prod.yaml`
  - Config: `samconfig.toml`
  - Documentation: `DEPLOYMENT_USAGE.md`

---

## Next Steps

1. Review recommendations with team
2. Prioritize which changes to implement
3. Update `sam-template-prod.yaml` with selected improvements
4. Update `samconfig.toml` with new parameters
5. Test changes in dev environment first
6. Update `DEPLOYMENT_USAGE.md` with any new parameters or conventions
7. Deploy to qa, then prod

---

## Questions for Discussion

1. Should we align stack naming to `WeightProcessor{Environment}` pattern?
2. Should staging environment be renamed to `qa-us` for consistency?
3. Do we need region in environment identifier for future multi-region support?
4. Should we create a shared utilities template for common CloudWatch alarm patterns?
5. What should be the VantaOwner for weight-processor?
