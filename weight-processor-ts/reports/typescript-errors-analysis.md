# Investigation: TypeScript Type Errors in Weight Processor

## Bottom Line

**Root Cause**: Mismatch between interface definitions using camelCase and implementations using snake_case throughout codebase
**Fix Location**: Multiple files - systematic naming convention alignment needed
**Confidence**: High

## What's Happening

The TypeScript weight processor has 256 type errors falling into 6 major categories. The primary issues are naming convention mismatches between type definitions (camelCase) and implementation code (snake_case), plus incorrect type assertions.

## Why It Happens

**Primary Cause**: Python-to-TypeScript port retained Python naming conventions in implementation code
**Trigger**: `src/models.ts:152-180` - ProcessResult interface uses camelCase
**Decision Point**: Implementation files using snake_case property access throughout

## Error Categories

### 1. Index Signature Access (87 errors - 34%)
- **Pattern**: `Property 'X' comes from an index signature, must be accessed with ['X']`
- **Root Cause**: Config and metadata objects use index signatures `[key: string]: any`
- **Example**: `config.adaptation_days` should be `config['adaptation_days']`
- **Files**: processor.ts, type_conversion.ts, outlier_detection.ts, replay_buffer.ts

### 2. Naming Convention Mismatches (45 errors - 18%)
- **Pattern**: Property name mismatches between interfaces and usage
- **Root Cause**: Interfaces define camelCase, code uses snake_case
- **Examples**:
  - `quality_score` vs `qualityScore`
  - `kalman_estimate` vs `kalmanEstimate`
  - `initial_variance_multiplier` vs `initialVarianceMultiplier`
- **Files**: processor.ts, reset_manager.ts, services/*

### 3. Missing Properties (31 errors - 12%)
- **Pattern**: Properties don't exist on type or using wrong names
- **Root Cause**: ProcessResult missing properties that code expects
- **Examples**:
  - `stage`, `preprocessing`, `noise_multiplier` not in ProcessResult
  - `adaptive_noise` not in Config
  - `lastAcceptedWeight` not in ProcessorState

### 4. Possibly Undefined (30 errors - 12%)
- **Pattern**: Object/property possibly undefined
- **Root Cause**: Optional properties accessed without null checks
- **Files**: database.ts, processor.ts, weight_processor_service.ts

### 5. Type Incompatibilities (28 errors - 11%)
- **Pattern**: Type mismatches in assignments/returns
- **Key Issues**:
  - `Partial<ProcessResult>` assigned to `ProcessResult`
  - `Vector2` (tuple) cast to `number[][]`
  - linearRegression returns `[number, number]` but destructured as object
- **Files**: processor.ts, database.ts

### 6. Unused Variables (35 errors - 14%)
- **Pattern**: Variables declared but never read
- **Low Priority**: Cleanup needed but not blocking

## Evidence

- **Key File**: `src/models.ts:152-180` - ProcessResult interface defines camelCase
- **Search Used**: `rg "quality_score|qualityScore" --type ts` - Shows mixed usage
- **Type Definition**: linearRegression returns tuple `[number, number]` not object

## Next Steps

1. **Standardize on camelCase** - Update all snake_case property access to camelCase
2. **Fix linearRegression usage** - Destructure as array: `const [slope, intercept] = linearRegression(x, y)`
3. **Add missing properties** - Either add to ProcessResult interface or create extended types
4. **Add null checks** - Guard optional property access with proper checks
5. **Remove index signatures** - Create proper typed interfaces for Config objects

## Risks

- **Runtime failures** from undefined property access
- **Data loss** from incorrect type assertions
- **Maintainability issues** from inconsistent naming conventions
