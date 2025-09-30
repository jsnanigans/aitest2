# AWS & SAM Setup Security Review
**Review Date:** 2025-09-30
**Reviewer:** Expert Council Security Analysis
**Project:** Weight Processor Service

---

## Executive Summary

**Overall Status:** ⚠️ **NOT SAFE TO PUBLISH** - Critical security issues identified

The Weight Processor Service AWS/SAM configuration contains several **critical security vulnerabilities** that must be addressed before production deployment. While the codebase demonstrates good practices in some areas (proper error handling, CloudWatch monitoring, etc.), there are significant security gaps that could lead to unauthorized access, data breaches, and compliance violations.

**Risk Level:** 🔴 **HIGH**

---

### 4. **INSUFFICIENT IAM PERMISSIONS VALIDATION**
**File:** `weight_values/sam-template-local.yaml:124-125`
**Severity:** 🟡 **MEDIUM-HIGH**

```yaml
Policies:
  - DynamoDBCrudPolicy:
      TableName: weight-processor-state-local
```

**Issue:** While `DynamoDBCrudPolicy` is used, it grants **full CRUD access** (Create, Read, Update, Delete) to the entire table.

**Risk:**
- Lambda can delete entire user state records (beyond soft deletes)
- No segregation between read-only and write operations
- Violates principle of least privilege
- If Lambda is compromised, attacker has full database access

**Recommendation:**
```yaml
Policies:
  - DynamoDBReadPolicy:
      TableName: !Ref StateTable
  - Statement:
      - Effect: Allow
        Action:
          - dynamodb:PutItem
          - dynamodb:UpdateItem
          - dynamodb:Query  # For snapshots
        Resource:
          - !GetAtt StateTable.Arn
          - !Sub "${StateTable.Arn}/index/*"
  # Separate admin function for DeleteItem if needed
```

---

### 5. **MISSING ENCRYPTION CONFIGURATION**
**File:** `weight_values/sam-template-prod.yaml:71-98`
**Severity:** 🔴 **HIGH**

**Issue:** DynamoDB table does not specify encryption configuration.

```yaml
StateTable:
  Type: AWS::DynamoDB::Table
  Properties:
    # ❌ No SSESpecification defined
    BillingMode: PAY_PER_REQUEST
    # ... rest of config
```

**Risk:**
- Data may not be encrypted at rest (depends on AWS default settings)
- May violate compliance requirements (HIPAA, GDPR, SOC 2)
- Health data (weight measurements) could be exposed if physical storage is compromised

**Fix:**
```yaml
StateTable:
  Type: AWS::DynamoDB::Table
  Properties:
    SSESpecification:
      SSEEnabled: true
      SSEType: KMS  # Use AWS KMS for encryption
      KMSMasterKeyId: !Ref DatabaseEncryptionKey  # Create separate KMS key
```

---

## 🟡 Security Concerns (Medium Priority)

### 6. **NO VPC CONFIGURATION FOR LAMBDA**
**File:** `weight_values/sam-template-prod.yaml:62-68`
**Severity:** 🟡 **MEDIUM**

```yaml
# Optional: VPC Configuration for private Lambda
# VpcConfig:
#   SecurityGroupIds: ...
#   SubnetIds: ...
```

**Issue:** Lambda function is deployed in the default AWS network, not in a VPC.

**Risk:**
- Function has internet access by default
- Cannot restrict network access to DynamoDB via VPC endpoints
- More difficult to audit network traffic
- Potential for data exfiltration if compromised

**Recommendation:**
- Deploy Lambda in private VPC subnets
- Use VPC endpoints for DynamoDB (avoids internet routing)
- Use security groups to restrict outbound traffic
- Note: VPC deployment increases cold start time (trade-off)

---

### 7. **INSUFFICIENT LOGGING CONFIGURATION**
**File:** `weight_values/lambda_handler.py:28-29`
**Severity:** 🟡 **MEDIUM**

```python
logger = logging.getLogger()
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))
```

**Issue:**
- No structured logging (JSON format)
- No log sampling to prevent DoS via logging
- Potentially sensitive data logged at DEBUG level
- No log retention policy enforcement in code

**Risk:**
- Difficult to query and analyze logs
- Could log sensitive user data (weight, timestamps)
- No rate limiting on log volume (cost risk)

**Recommendation:**
```python
import json
import logging
from aws_lambda_powertools import Logger

logger = Logger(service="weight-processor")

# Structured logging with automatic PII redaction
logger.info("Processing measurement", extra={
    "user_id": mask_pii(user_id),
    "measurement_count": len(measurements)
})
```

---

## 📋 Best Practices Issues

### 11. **MISSING CLOUDWATCH ALARMS FOR SECURITY EVENTS**
**File:** `weight_values/sam-template-prod.yaml:107-157`

**Good:** Error and throttle alarms exist
**Missing:**
- Alarm for unusual access patterns (e.g., 1000s of requests for single user)
- Alarm for authentication failures (when auth is added)
- Alarm for DynamoDB throttling
- Alarm for Lambda duration spikes (could indicate attack)

**Recommendation:** Add security-focused alarms:
```yaml
UnauthorizedAccessAlarm:
  Type: AWS::CloudWatch::Alarm
  Properties:
    MetricName: 4XXError
    Threshold: 10
    # Trigger on repeated unauthorized access attempts
```

---

### 12. **NO DEAD LETTER QUEUE (DLQ)**
**File:** `weight_values/sam-template-prod.yaml:26-69`

**Issue:** Lambda has no DLQ configured for failed invocations.

**Risk:**
- Failed requests are silently dropped
- No way to replay failed measurements
- Data loss if processing fails

**Recommendation:**
```yaml
WeightProcessorFunction:
  Type: AWS::Serverless::Function
  Properties:
    DeadLetterQueue:
      Type: SQS
      TargetArn: !GetAtt ProcessorDLQ.Arn
```

---

### 13. **INSUFFICIENT DYNAMODB BACKUP CONFIGURATION**
**File:** `weight_values/sam-template-prod.yaml:86-92`

**Good:** Point-in-time recovery is enabled ✅
**Missing:**
- No automated backup schedule
- No cross-region backup replication
- No backup retention policy defined

**Recommendation:**
```yaml
BackupPlan:
  Type: AWS::Backup::BackupPlan
  Properties:
    BackupPlanRule:
      - RuleName: DailyBackups
        TargetBackupVault: !Ref BackupVault
        ScheduleExpression: "cron(0 5 ? * * *)"
        Lifecycle:
          DeleteAfterDays: 30
```

---

### 14. **NO XRAY TRACING ENABLED**
**File:** `weight_values/sam-template-prod.yaml`

**Issue:** AWS X-Ray tracing is not enabled.

**Impact:**
- Difficult to debug performance issues
- Cannot trace requests across services
- Harder to identify security incidents

**Recommendation:**
```yaml
Globals:
  Function:
    Tracing: Active  # Enable X-Ray
```

---

### 15. **MISSING RESOURCE TAGS**
**File:** `weight_values/sam-template-prod.yaml`

**Issue:** Inconsistent tagging. DynamoDB table has tags, but Lambda does not.

**Impact:**
- Difficult cost allocation
- Harder to track resources
- Compliance issues (many frameworks require tagging)

**Recommendation:**
```yaml
Globals:
  Function:
    Tags:
      Environment: !Ref Environment
      Service: weight-processor
      CostCenter: health-services
      DataClassification: PII
```

---

## ✅ Positive Findings

The following aspects demonstrate good security practices:

1. **IAM Roles Over Credentials** ✅
   - Production template uses IAM roles (via policies) instead of access keys
   - `weight_values/sam-template-prod.yaml:54-56`

2. **CloudWatch Logging Enabled** ✅
   - Log group with 30-day retention
   - `weight_values/sam-template-prod.yaml:100-104`

3. **CloudWatch Alarms Configured** ✅
   - Error, throttle, and DynamoDB throttle alarms
   - `weight_values/sam-template-prod.yaml:107-157`

4. **Lambda Insights Enabled** ✅
   - Performance monitoring with Lambda Insights layer
   - `weight_values/sam-template-prod.yaml:57-59`

5. **Point-in-Time Recovery** ✅
   - DynamoDB PITR enabled for disaster recovery
   - `weight_values/sam-template-prod.yaml:91-92`

6. **TTL Configuration** ✅
   - Automatic data expiration for snapshots
   - `weight_values/sam-template-prod.yaml:88-90`

7. **Concurrency Limits** ✅
   - Lambda concurrency capped at 100 to prevent runaway costs
   - `weight_values/sam-template-prod.yaml:38`

8. **Request Validation** ✅
   - Pydantic models for input validation
   - `weight_values/lambda_handler.py:175-183`

9. **Proper Error Handling** ✅
   - Comprehensive error handling with specific error codes
   - `weight_values/lambda_handler.py:516-546`

10. **Secure DynamoDB Key Schema** ✅
    - Composite key prevents direct enumeration
    - `weight_values/sam-template-prod.yaml:82-85`

---

## 🔧 Configuration Issues

### 16. **INCONSISTENT ENVIRONMENT NAMING**
**Files:** Multiple SAM templates

**Issue:**
- Local: `weight-processor-state-local`
- Prod: `weight-processor-state-${Environment}`
- Layer: `weight-processor-deps-${Environment}`

**Risk:** Accidental cross-environment resource access

**Fix:** Standardize naming: `weight-processor-${Environment}-${Resource}`

---

### 17. **MISSING SAMCONFIG.TOML**
**File:** Not found

**Issue:** No `samconfig.toml` file for deployment configuration.

**Impact:**
- Must specify all parameters on every deployment
- Easy to deploy to wrong environment
- Inconsistent deployments across team

**Recommendation:** Create `samconfig.toml`:
```toml
version = 0.1
[default.deploy.parameters]
stack_name = "weight-processor-dev"
resolve_s3 = true
region = "us-east-1"
confirm_changeset = true
capabilities = "CAPABILITY_IAM"
```

---

## 📊 Security Metrics Summary

| Category | Status | Count |
|----------|--------|-------|
| 🔴 Critical Issues | **MUST FIX** | 5 |
| 🟡 High Priority | Should Fix | 5 |
| 🟠 Medium Priority | Recommended | 5 |
| ℹ️ Best Practices | Consider | 3 |
| ✅ Positive Findings | Good | 10 |

---

## 🎯 Prioritized Remediation Plan

### Phase 1: Critical Fixes (Before ANY Production Deployment)
**Timeline:** Immediate (1-2 days)

0. **Enable DynamoDB encryption**
   - Add `SSESpecification` to prod template
   - Create KMS key for encryption

0. **Review and restrict IAM permissions**
   - Replace `DynamoDBCrudPolicy` with specific actions
   - Implement read/write separation

---

### Phase 2: High Priority Fixes (Before Public Release)
**Timeline:** 1 week

0. **Deploy Lambda in VPC**
   - Create private subnets
   - Add VPC endpoints for DynamoDB
   - Configure security groups

0. **Implement structured logging**
   - Use AWS Lambda Powertools
   - Add PII redaction
   - Implement log sampling

---

### Phase 3: Best Practices (Before Scale)
**Timeline:** 2-4 weeks

1. Add security-focused CloudWatch alarms
2. Configure Dead Letter Queue
3. Implement automated backup schedule
4. Enable X-Ray tracing
5. Standardize resource tagging
6. Create `samconfig.toml`
7. Document security architecture

---

## 🛡️ Additional Recommendations

### Security Testing
1. **Run security scans before deployment:**
   ```bash
   # Static analysis
   pip install bandit
   bandit -r weight_values/src/

   # Dependency vulnerabilities
   pip install safety
   safety check -r requirements.txt

   # SAM template validation
   cfn-lint weight_values/sam-template-prod.yaml
   ```

2. **Implement automated security testing:**
   - Add `checkov` or `cfn-nag` to CI/CD pipeline
   - Run OAST (SAST/DAST) tools before production

### Monitoring & Incident Response
1. **Create runbook for security incidents**
2. **Set up AWS GuardDuty for threat detection**
3. **Configure AWS Config for compliance monitoring**
4. **Implement AWS Security Hub for centralized security view**

### Documentation
1. **Document threat model**
2. **Create security architecture diagram**
3. **Document data classification (PII handling)**
4. **Create incident response plan**

---

## 💡 Expert Council Review

-- COUNCIL REVIEW --
**Task:** Assess AWS/SAM setup for production readiness and security

**Approach:** Comprehensive review of SAM templates, Lambda configuration, IAM policies, and deployment practices

**Council's Key Concerns:**

• **Nancy Leveson (Safety):** "The hardcoded credentials in the SAM template are a ticking time bomb. If this template is accidentally used in production or shared, you've just exposed AWS access. What's the worst case? Complete account compromise."

• **Matt Blaze (Security):** "CORS set to `*` on a health data API? That's inviting CSRF attacks. Also, no encryption explicitly configured for DynamoDB means you're gambling on defaults. And where's your API authentication? Anyone can invoke these endpoints."

• **Barbara Liskov (Correctness):** "The IAM policy grants full CRUD access when the Lambda only needs read and write. This violates the fundamental invariant of least privilege. If the Lambda is compromised, the attacker has full database access."

• **The SRE on Call (Reliability):** "No DLQ? You're dropping failed requests on the floor. No X-Ray tracing? Good luck debugging production issues. And that CORS `*` will come back to haunt you at 3 AM."

• **Martin Kleppmann (Distributed Systems):** "Your DynamoDB table has no explicit encryption, no cross-region replication, and limited backup strategy. What happens when AWS has a regional outage? Or when you need to meet HIPAA compliance?"

**Recommendation:** **BLOCK PRODUCTION DEPLOYMENT**

This setup has critical security vulnerabilities that pose immediate risk:
1. Credential exposure via hardcoded values
2. No API authentication (open to public access)
3. Unrestricted CORS enabling CSRF
4. Insufficient encryption guarantees
5. Overly permissive IAM policies

**Alternative:** Implement Phase 1 critical fixes before any deployment. This is not about perfection—it's about preventing easily exploitable vulnerabilities that could lead to data breaches, compliance violations, and significant business impact.

**Awaiting User Decision:** Please address the 5 critical issues before proceeding.

-- END COUNCIL --

---

## 📚 References

- [AWS Lambda Security Best Practices](https://docs.aws.amazon.com/lambda/latest/dg/lambda-security.html)
- [AWS SAM Security](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/serverless-policy-templates.html)
- [OWASP API Security Top 10](https://owasp.org/www-project-api-security/)
- [AWS Well-Architected Framework - Security Pillar](https://docs.aws.amazon.com/wellarchitected/latest/security-pillar/welcome.html)
- [CIS AWS Foundations Benchmark](https://www.cisecurity.org/benchmark/amazon_web_services)

---

**Review Completed:** 2025-09-30
**Next Review:** After Phase 1 fixes are implemented
