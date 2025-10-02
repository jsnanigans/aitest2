# TypeScript Weight Processor Port - Recommendations

## Executive Summary

Based on the analysis in `discussion.md` and research in `research.md`, this document provides concrete recommendations for implementing the TypeScript weight processor port. The recommended approach prioritizes **correctness and validation** while establishing a foundation for **long-term maintainability**.

## Recommended Approach

### 1. Module Organization: **Hybrid (Mirror → Refactor)**

**Decision**: Start by mirroring Python structure exactly, then refactor after validation.

**Rationale**:
- **Risk Mitigation**: Mirroring Python structure minimizes translation errors during initial port
- **Validation First**: Easier to validate outputs match Python when structure is identical
- **Iterative Improvement**: Can refactor with confidence once comprehensive tests pass
- **Pragmatic**: Delivers working solution fast, improves over time

**Implementation Strategy**:

**Phase 1 (Weeks 1-4): Mirror Python Structure**
```
weight-processor-ts/
├── src/
│   ├── core/
│   │   ├── processing/
│   │   │   ├── processor.ts
│   │   │   ├── kalman.ts
│   │   │   ├── kalman_filter.ts
│   │   │   ├── unified_quality_scorer.ts
│   │   │   └── ... (mirror all 14 files)
│   │   ├── database/
│   │   │   ├── base.ts
│   │   │   ├── database.ts
│   │   │   └── db_wrapper.ts
│   │   └── replay/
│   │       └── ... (mirror all 6 files)
│   ├── config/
│   ├── services/
│   ├── models.ts
│   ├── constants.ts
│   ├── utils.ts
│   └── index.ts
├── local_main.ts
└── config.toml
```

**Phase 2 (Week 5): Validation**
- Generate Python outputs for test datasets
- Run TypeScript implementation
- Compare outputs (must match within 0.1%)
- Fix any discrepancies

**Phase 3 (Week 6+, Optional): Refactor to TypeScript Idioms**
- Reorganize into cleaner module boundaries
- Maintain all tests (they must still pass)
- Improve developer experience
- This phase can be deferred if timeline is tight

**Why Not Start with TypeScript Structure?**
- Higher risk of missing functionality during reorganization
- Harder to validate correctness (less obvious mapping to Python)
- More upfront design time before getting to working code
- Can always refactor later with test coverage protecting us

### 2. Programming Paradigm: **Functional with Namespaces**

**Decision**: Use pure functions organized in namespaces/modules, avoid classes with mutable state.

**Rationale**:
- **Testability**: Pure functions are easiest to test (input → output, no side effects)
- **Type Safety**: No hidden state, all dependencies explicit in parameters
- **Correctness**: Easier to reason about and verify against Python
- **Performance**: Can optimize pure functions more aggressively
- **Tree-shaking**: Better dead code elimination for library consumers

**Implementation Pattern**:

```typescript
// kalman_filter.ts
export namespace KalmanFilter {
    /**
     * Perform Kalman filter prediction step.
     * Pure function - no side effects.
     */
    export function predict(
        params: KalmanParams,
        stateMean: readonly number[],
        stateCovariance: readonly number[][]
    ): PredictResult {
        const F = buildTransitionMatrix(params);
        const Q = params.transitionCovariance;

        const predictedMean = Matrix.multiply(F, stateMean);
        const predictedCov = Matrix.add(
            Matrix.multiply(Matrix.multiply(F, stateCovariance), Matrix.transpose(F)),
            Q
        );

        return {
            predictedStateMean: predictedMean,
            predictedStateCovariance: predictedCov
        };
    }

    /**
     * Perform Kalman filter update step.
     * Pure function - no side effects.
     */
    export function update(
        params: KalmanParams,
        predictedMean: readonly number[],
        predictedCov: readonly number[][],
        observation: readonly number[]
    ): UpdateResult {
        const H = params.observationMatrices;
        const R = params.observationCovariance;

        const innovation = Matrix.subtract(observation, Matrix.multiply(H, predictedMean));
        const innovationCov = Matrix.add(
            Matrix.multiply(Matrix.multiply(H, predictedCov), Matrix.transpose(H)),
            R
        );
        const kalmanGain = Matrix.multiply(
            Matrix.multiply(predictedCov, Matrix.transpose(H)),
            Matrix.invert(innovationCov)
        );

        const filteredMean = Matrix.add(predictedMean, Matrix.multiply(kalmanGain, innovation));

        // Joseph form for numerical stability
        const IKH = Matrix.subtract(
            Matrix.eye(params.nStates),
            Matrix.multiply(kalmanGain, H)
        );
        const filteredCov = Matrix.add(
            Matrix.multiply(Matrix.multiply(IKH, predictedCov), Matrix.transpose(IKH)),
            Matrix.multiply(Matrix.multiply(kalmanGain, R), Matrix.transpose(kalmanGain))
        );

        return {
            filteredStateMean: filteredMean,
            filteredStateCovariance: filteredCov,
            innovation,
            innovationCovariance: innovationCov
        };
    }

    export function filterUpdate(
        params: KalmanParams,
        currentMean: readonly number[],
        currentCov: readonly number[][],
        observation: readonly number[]
    ): FilterUpdateResult {
        const predicted = predict(params, currentMean, currentCov);
        const updated = update(params, predicted.predictedStateMean, predicted.predictedStateCovariance, observation);
        return {
            ...predicted,
            ...updated
        };
    }
}

// Usage
const result = KalmanFilter.predict(params, stateMean, stateCov);
const updated = KalmanFilter.update(params, result.predictedStateMean, result.predictedStateCovariance, obs);
```

**Key Principles**:
1. **Pure Functions**: No mutation of inputs, no side effects
2. **Readonly Parameters**: Use `readonly` to enforce immutability at type level
3. **Explicit Dependencies**: All inputs as parameters (no hidden globals)
4. **Composability**: Small, focused functions that combine cleanly
5. **Namespaces for Organization**: Group related functions logically

**Exception: State Management**
State storage (ProcessorStateDB) can use a class since it's explicitly stateful:

```typescript
export class ProcessorStateDB {
    private states: Map<string, ProcessorState> = new Map();
    private snapshots: Map<string, Snapshot[]> = new Map();

    // Methods that explicitly manage state
    getState(userId: string): ProcessorState | null {
        const state = this.states.get(userId);
        return state ? deepCopy(state) : null;  // Return copy, not reference
    }

    saveState(userId: string, state: ProcessorState): void {
        this.states.set(userId, deepCopy(state));  // Store copy, not reference
    }
}
```

### 3. Matrix Operations: **Minimal + Extension Path**

**Decision**: Implement only what's needed (2x2 matrices), but structure for easy extension.

**Rationale**:
- **Focus**: Don't over-engineer - current needs are specific and limited
- **Performance**: Specialized 2x2 operations faster than generic NxM
- **Simplicity**: Less code to test and maintain
- **Future-Proof**: Clear path to add generic operations if needed
- **Pragmatic**: Ship working code, extend later if requirements change

**Implementation**:

```typescript
// src/core/math/matrix.ts

/**
 * Matrix operations optimized for weight processor use case.
 * Currently focused on 2x2 matrices used by Kalman filter.
 */
export namespace Matrix {
    /**
     * Multiply two 2x2 matrices.
     * Optimized for the specific 2x2 case.
     */
    export function multiply2x2(a: readonly number[][], b: readonly number[][]): number[][] {
        if (a.length !== 2 || a[0].length !== 2 || b.length !== 2 || b[0].length !== 2) {
            throw new Error('multiply2x2 requires 2x2 matrices');
        }

        return [
            [
                a[0][0] * b[0][0] + a[0][1] * b[1][0],
                a[0][0] * b[0][1] + a[0][1] * b[1][1]
            ],
            [
                a[1][0] * b[0][0] + a[1][1] * b[1][0],
                a[1][0] * b[0][1] + a[1][1] * b[1][1]
            ]
        ];
    }

    /**
     * Invert a 2x2 matrix using analytical formula.
     * Much faster than generic LU decomposition.
     */
    export function invert2x2(m: readonly number[][]): number[][] {
        const [[a, b], [c, d]] = m;
        const det = a * d - b * c;

        if (Math.abs(det) < 1e-10) {
            throw new Error('Matrix is singular (determinant near zero)');
        }

        return [
            [d / det, -b / det],
            [-c / det, a / det]
        ];
    }

    /**
     * Multiply matrix by vector (2x2 * 2x1 → 2x1).
     */
    export function multiplyVector2x2(m: readonly number[][], v: readonly number[]): number[] {
        if (m.length !== 2 || v.length !== 2) {
            throw new Error('multiplyVector2x2 requires 2x2 matrix and 2x1 vector');
        }

        return [
            m[0][0] * v[0] + m[0][1] * v[1],
            m[1][0] * v[0] + m[1][1] * v[1]
        ];
    }

    /**
     * Transpose a 2x2 matrix.
     */
    export function transpose2x2(m: readonly number[][]): number[][] {
        return [
            [m[0][0], m[1][0]],
            [m[0][1], m[1][1]]
        ];
    }

    /**
     * Add two 2x2 matrices.
     */
    export function add2x2(a: readonly number[][], b: readonly number[][]): number[][] {
        return [
            [a[0][0] + b[0][0], a[0][1] + b[0][1]],
            [a[1][0] + b[1][0], a[1][1] + b[1][1]]
        ];
    }

    /**
     * Subtract two 2x2 matrices.
     */
    export function subtract2x2(a: readonly number[][], b: readonly number[][]): number[][] {
        return [
            [a[0][0] - b[0][0], a[0][1] - b[0][1]],
            [a[1][0] - b[1][0], a[1][1] - b[1][1]]
        ];
    }

    /**
     * Create 2x2 identity matrix.
     */
    export function eye2(): number[][] {
        return [[1, 0], [0, 1]];
    }

    /**
     * Scalar multiplication of 2x2 matrix.
     */
    export function scalarMultiply2x2(scalar: number, m: readonly number[][]): number[][] {
        return [
            [scalar * m[0][0], scalar * m[0][1]],
            [scalar * m[1][0], scalar * m[1][1]]
        ];
    }

    // TODO: Generic NxM operations (implement when needed)
    // export namespace Generic {
    //     export function multiply(a: number[][], b: number[][]): number[][] {
    //         throw new Error('Generic matrix multiplication not yet implemented');
    //     }
    // }
}
```

**Why Not Generic Matrix Library?**
- Current needs: Only 2x2 matrices (Kalman state is [weight, velocity])
- Specialized code is 2-3x faster
- ~200 lines vs 1000+ for generic
- Can add generic later if needed (clear TODO markers)
- YAGNI principle: You Aren't Gonna Need It (yet)

### 4. Testing Strategy: **Validation + Unit Tests Hybrid**

**Decision**: Python output validation tests + comprehensive unit tests.

**Rationale**:
- **Correctness**: Validation tests prove we match Python exactly
- **Regression Protection**: Unit tests catch bugs during refactoring
- **Confidence**: Dual approach gives highest confidence
- **Granularity**: Unit tests pinpoint issues faster than end-to-end
- **Documentation**: Tests serve as executable specification

**Implementation**:

**Test Structure**:
```
tests/
├── validation/           # Python output comparison
│   ├── fixtures/
│   │   ├── dataset-small.csv
│   │   ├── dataset-small-python-output.json
│   │   ├── dataset-medium.csv
│   │   ├── dataset-medium-python-output.json
│   │   └── ...
│   ├── python-comparison.test.ts
│   └── numerical-accuracy.test.ts
├── unit/                 # Isolated unit tests
│   ├── core/
│   │   ├── kalman-filter.test.ts
│   │   ├── quality-scorer.test.ts
│   │   ├── reset-manager.test.ts
│   │   └── ...
│   ├── math/
│   │   ├── matrix.test.ts
│   │   └── statistics.test.ts
│   └── ...
├── integration/          # Multi-module integration
│   ├── full-pipeline.test.ts
│   ├── replay-system.test.ts
│   └── ...
└── helpers/
    ├── test-data.ts
    ├── assertions.ts
    └── comparison.ts
```

**Validation Test Example**:
```typescript
// tests/validation/python-comparison.test.ts
import { describe, test, expect } from 'bun:test';
import { WeightProcessorService } from '../../src/services/weight_processor_service';
import { ProcessorStateDB } from '../../src/core/database/database';
import { ConfigManager } from '../../src/config/config_manager';
import { loadCsvData } from '../../src/local_main';
import { readFileSync } from 'fs';

describe('Python Output Validation', () => {
    test('dataset-small: full pipeline matches Python within 0.1%', () => {
        // Load test data
        const {userMeasurements} = loadCsvData('tests/validation/fixtures/dataset-small.csv', {
            maxUsers: 0,
            maxRows: 0,
            minReadings: 0
        });

        // Load expected Python output
        const pythonOutput = JSON.parse(
            readFileSync('tests/validation/fixtures/dataset-small-python-output.json', 'utf-8')
        );

        // Process with TypeScript
        const config = ConfigManager.loadConfig();
        const stateStore = new ProcessorStateDB();
        const service = new WeightProcessorService(stateStore, config);

        const tsResults = new Map();
        for (const [userId, measurements] of userMeasurements) {
            const result = service.processBatch(userId, measurements);
            tsResults.set(userId, result);
        }

        // Compare results
        for (const [userId, tsResult] of tsResults) {
            const pyResult = pythonOutput[userId];

            // High-level metrics
            expect(tsResult.measurementsProcessed).toBe(pyResult.measurements_processed);
            expect(tsResult.measurementsAccepted).toBe(pyResult.measurements_accepted);
            expect(tsResult.measurementsRejected).toBe(pyResult.measurements_rejected);

            // Per-measurement comparison
            for (let i = 0; i < tsResult.results.length; i++) {
                const tsRes = tsResult.results[i];
                const pyRes = pyResult.results[i];

                // Binary decisions must match exactly
                expect(tsRes.accepted).toBe(pyRes.accepted);

                if (tsRes.accepted) {
                    // Numerical values within 0.1% tolerance
                    expectClose(tsRes.kalmanEstimate, pyRes.kalman_estimate, 0.001);
                    expectClose(tsRes.qualityScore, pyRes.quality_score, 0.001);
                    expectClose(tsRes.kalmanUncertainty, pyRes.kalman_uncertainty, 0.001);
                } else {
                    // Rejection reasons should match (or both rejected)
                    expect(tsRes.rejectionReason).toBeDefined();
                    expect(pyRes.rejection_reason).toBeDefined();
                }
            }
        }
    });

    // More validation tests for different scenarios
    test('dataset-medium: handles resets correctly', () => { /* ... */ });
    test('dataset-replay: replay mechanism matches Python', () => { /* ... */ });
});

function expectClose(actual: number, expected: number, tolerance: number): void {
    const diff = Math.abs(actual - expected);
    const relativeError = diff / Math.abs(expected);
    expect(relativeError).toBeLessThan(tolerance);
}
```

**Unit Test Example**:
```typescript
// tests/unit/core/kalman-filter.test.ts
import { describe, test, expect } from 'bun:test';
import { KalmanFilter } from '../../../src/core/processing/kalman_filter';

describe('KalmanFilter', () => {
    describe('predict', () => {
        test('predicts next state correctly for typical case', () => {
            const params = {
                transitionMatrices: [[1, 1], [0, 1]],  // F
                observationMatrices: [[1, 0]],         // H
                transitionCovariance: [[0.018, 0], [0, 0.00012]],  // Q
                observationCovariance: [[5.0]],        // R
                nStates: 2,
                nObservations: 1
            };

            const stateMean = [75.0, 0.1];  // 75kg, +0.1kg/day trend
            const stateCov = [[1.0, 0], [0, 0.01]];

            const result = KalmanFilter.predict(params, stateMean, stateCov);

            // Expected: weight = 75 + 1*0.1 = 75.1, trend unchanged
            expect(result.predictedStateMean[0]).toBeCloseTo(75.1);
            expect(result.predictedStateMean[1]).toBeCloseTo(0.1);

            // Covariance should increase (process noise added)
            expect(result.predictedStateCovariance[0][0]).toBeGreaterThan(stateCov[0][0]);
        });

        test('maintains covariance symmetry', () => {
            const params = createDefaultParams();
            const result = KalmanFilter.predict(params, [70, 0], [[1, 0.1], [0.1, 0.01]]);

            // Covariance must be symmetric
            expect(result.predictedStateCovariance[0][1])
                .toBeCloseTo(result.predictedStateCovariance[1][0]);
        });

        test('covariance stays positive definite', () => {
            // ... test numerical stability
        });
    });

    describe('update', () => {
        test('updates state based on observation', () => {
            const params = createDefaultParams();
            const predictedMean = [75.1, 0.1];
            const predictedCov = [[1.018, 0], [0, 0.01012]];
            const observation = [75.5];  // Measured 75.5kg

            const result = KalmanFilter.update(params, predictedMean, predictedCov, observation);

            // State should move toward observation
            expect(result.filteredStateMean[0]).toBeGreaterThan(75.1);
            expect(result.filteredStateMean[0]).toBeLessThan(75.5);

            // Uncertainty should decrease
            expect(result.filteredStateCovariance[0][0]).toBeLessThan(predictedCov[0][0]);
        });

        test('Joseph form prevents numerical instability', () => {
            // Test that covariance stays positive definite
            // after many iterations
        });
    });
});
```

**Integration Test Example**:
```typescript
// tests/integration/full-pipeline.test.ts
describe('Full Processing Pipeline', () => {
    test('processes user from scratch through multiple measurements', () => {
        const service = new WeightProcessorService();
        const userId = 'test-user-001';

        // First measurement - should trigger initial reset
        const result1 = service.processSingle(userId, {
            uuid: 'm1',
            weight: 80.0,
            unit: 'kg',
            effectiveDateTime: new Date('2024-01-01T08:00:00Z'),
            source: 'patient-device'
        });

        expect(result1.accepted).toBe(true);
        expect(result1.resetTriggered).toBe(true);
        expect(result1.resetType).toBe('initial');

        // Second measurement - normal processing
        const result2 = service.processSingle(userId, {
            uuid: 'm2',
            weight: 80.2,
            unit: 'kg',
            effectiveDateTime: new Date('2024-01-02T08:00:00Z'),
            source: 'patient-device'
        });

        expect(result2.accepted).toBe(true);
        expect(result2.resetTriggered).toBe(false);

        // Large gap - should trigger hard reset
        const result3 = service.processSingle(userId, {
            uuid: 'm3',
            weight: 78.0,
            unit: 'kg',
            effectiveDateTime: new Date('2024-02-15T08:00:00Z'),  // 44 days later
            source: 'patient-device'
        });

        expect(result3.accepted).toBe(true);
        expect(result3.resetTriggered).toBe(true);
        expect(result3.resetType).toBe('hard');
    });
});
```

**Test Coverage Goals**:
- Unit tests: >90% line coverage
- Integration tests: All major workflows
- Validation tests: Multiple datasets covering edge cases
- Total coverage: >80% (excluding generated code)

### 5. Build Strategy: **Dual Entry Points (Library + CLI)**

**Decision**: Separate library and CLI builds with proper TypeScript compilation.

**Rationale**:
- **Professional**: Standard npm package structure
- **Reusability**: Library can be imported by other projects
- **Type Safety**: Generate .d.ts files for TypeScript consumers
- **CLI Usability**: Standalone executable for CSV processing
- **Flexibility**: Consumers can use either interface

**Implementation**:

**package.json**:
```json
{
  "name": "@9amhealth/weight-processor",
  "version": "1.0.0",
  "description": "Weight measurement processing with Kalman filtering and quality scoring",
  "type": "module",
  "main": "./dist/index.js",
  "types": "./dist/index.d.ts",
  "bin": {
    "weight-processor": "./dist/cli.js"
  },
  "exports": {
    ".": {
      "import": "./dist/index.js",
      "types": "./dist/index.d.ts"
    },
    "./cli": {
      "import": "./dist/cli.js"
    }
  },
  "files": [
    "dist",
    "config.toml",
    "README.md",
    "LICENSE"
  ],
  "scripts": {
    "build": "npm run build:lib && npm run build:cli",
    "build:lib": "tsc --project tsconfig.lib.json",
    "build:cli": "bun build local_main.ts --outfile dist/cli.js --target node",
    "typecheck": "tsc --noEmit",
    "test": "bun test",
    "test:unit": "bun test tests/unit",
    "test:integration": "bun test tests/integration",
    "test:validation": "bun test tests/validation",
    "test:coverage": "bun test --coverage",
    "dev": "bun run local_main.ts",
    "lint": "eslint src/**/*.ts",
    "format": "prettier --write src/**/*.ts"
  },
  "dependencies": {
    "@iarna/toml": "^2.2.5",
    "csv-parse": "^5.5.0",
    "csv-stringify": "^6.4.0"
  },
  "devDependencies": {
    "@types/bun": "latest",
    "typescript": "^5.3.0",
    "eslint": "^8.50.0",
    "@typescript-eslint/eslint-plugin": "^6.7.0",
    "@typescript-eslint/parser": "^6.7.0",
    "prettier": "^3.0.3"
  },
  "engines": {
    "node": ">=18.0.0",
    "bun": ">=1.0.0"
  },
  "keywords": [
    "weight",
    "kalman-filter",
    "quality-scoring",
    "measurement-processing",
    "health-data"
  ]
}
```

**tsconfig.json** (base):
```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "ESNext",
    "lib": ["ES2022"],
    "moduleResolution": "bundler",
    "strict": true,
    "noImplicitAny": true,
    "strictNullChecks": true,
    "strictFunctionTypes": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noImplicitReturns": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "resolveJsonModule": true,
    "declaration": true,
    "declarationMap": true,
    "sourceMap": true,
    "outDir": "./dist",
    "rootDir": "./src",
    "types": ["bun-types"]
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist", "tests"]
}
```

**tsconfig.lib.json** (library build):
```json
{
  "extends": "./tsconfig.json",
  "compilerOptions": {
    "declaration": true,
    "declarationMap": true,
    "outDir": "./dist"
  },
  "include": ["src/**/*"],
  "exclude": ["src/**/*.test.ts", "local_main.ts"]
}
```

**Library Usage Example**:
```typescript
// Another TypeScript project using the library
import {
    WeightProcessorService,
    ProcessorStateDB,
    ConfigManager,
    type Measurement,
    type ProcessResult
} from '@9amhealth/weight-processor';

const config = ConfigManager.loadConfig();
const stateStore = new ProcessorStateDB();
const service = new WeightProcessorService(stateStore, config);

const measurement: Measurement = {
    uuid: 'measurement-123',
    weight: 75.5,
    unit: 'kg',
    effectiveDateTime: new Date(),
    source: 'patient-device'
};

const result: ProcessResult = service.processSingle('user-456', measurement);

if (result.accepted) {
    console.log(`Kalman estimate: ${result.kalmanEstimate}kg`);
    console.log(`Quality score: ${result.qualityScore}`);
}
```

**CLI Usage Example**:
```bash
# Install globally
npm install -g @9amhealth/weight-processor

# Use CLI
weight-processor \
  --csv-file data/weights.csv \
  --max-users 100 \
  --min-readings 20 \
  --output-dir output \
  --filtered-csv output/filtered.csv

# Or with bun directly
bun run local_main.ts --csv-file data/weights.csv ...
```

## Implementation Timeline

### Week 1-2: Foundation
- ✅ Set up project structure (mirror Python)
- ✅ Implement matrix operations (2x2 specialized)
- ✅ Implement statistical functions
- ✅ Port constants and configuration management
- ✅ Set up build system (tsconfig, package.json)
- ✅ Set up test infrastructure

### Week 3: Core Processing
- ✅ Port Kalman filter (kalman_filter.ts)
- ✅ Port Kalman manager (kalman.ts)
- ✅ Port reset manager
- ✅ Unit tests for each module
- ✅ Validate numerical accuracy against Python

### Week 4: Quality & State
- ✅ Port unified quality scorer
- ✅ Port database layer (in-memory)
- ✅ Port processor orchestrator
- ✅ Integration tests for pipeline
- ✅ Validation tests vs Python

### Week 5: Replay & Services
- ✅ Port replay system (buffer, manager, outlier detection)
- ✅ Port service layer
- ✅ Implement local_main.ts (CLI)
- ✅ CSV reading/writing
- ✅ End-to-end tests

### Week 6: Validation & Polish
- ✅ Generate Python outputs for test datasets
- ✅ Run comprehensive validation tests
- ✅ Fix any discrepancies
- ✅ Performance testing and optimization
- ✅ Documentation (README, API docs)
- ✅ Package for publication

### Week 7+ (Optional): Refactor
- ⏭️ Reorganize to TypeScript idioms (if desired)
- ⏭️ Maintain all test coverage
- ⏭️ Improve developer experience

## Success Criteria

### Functionality
- ✅ All Python modules ported to TypeScript
- ✅ `local_main.ts` produces identical results to `local_main.py`
- ✅ All processing features working (Kalman, quality, reset, replay)

### Quality
- ✅ Numerical accuracy within 0.1% of Python
- ✅ Test coverage > 80%
- ✅ Zero TypeScript errors in strict mode
- ✅ All validation tests passing

### Performance
- ✅ Processes 1000 measurements in < 5 seconds
- ✅ Processes 10,000 measurements in < 30 seconds
- ✅ Matches or exceeds Python performance

### Packaging
- ✅ Publishable npm package
- ✅ Working CLI executable
- ✅ Complete type definitions (.d.ts)
- ✅ Documentation and examples

## Risk Mitigation

### Risk: Numerical Precision Differences
**Mitigation**:
- Use same floating-point precision (64-bit)
- Implement Joseph form for covariance updates
- Comprehensive validation tests
- Log discrepancies for investigation

### Risk: Missing Python Functionality
**Mitigation**:
- Systematic file-by-file port
- Checklist of all modules
- Code review comparing Python to TypeScript
- Integration tests for all workflows

### Risk: Performance Issues
**Mitigation**:
- Profile both Python and TypeScript
- Optimize hot paths (matrix ops, quality scoring)
- Leverage Bun's speed
- Consider typed arrays if needed

### Risk: Maintenance Burden
**Mitigation**:
- Comprehensive test suite
- Clear documentation
- TypeScript catches many bugs at compile time
- Pure functions easier to maintain

## Next Steps

1. **Review and Approve** this recommendation
2. **Create Detailed Plan** with task breakdown (plan.md)
3. **Set Up Project** (repo, dependencies, tooling)
4. **Begin Implementation** following phased approach
5. **Weekly Check-ins** to track progress and adjust

---

## Appendix: Alternative Recommendations (Not Chosen)

### Why Not Classes with Mutable State?
- **Testability**: Harder to test (need to manage instance state)
- **Type Safety**: Hidden state makes dependencies implicit
- **Bugs**: Mutable state is a common source of bugs
- **Functional > OOP**: For data processing pipelines, functional is clearer

### Why Not Generic Matrix Library?
- **YAGNI**: Don't need it yet (only 2x2 matrices)
- **Performance**: Specialized code is faster
- **Complexity**: 1000+ lines vs 200 lines
- **Timeline**: Would delay completion

### Why Not TypeScript-First Structure?
- **Risk**: Higher chance of missing functionality
- **Validation**: Harder to verify correctness
- **Timeline**: More upfront design time
- **Pragmatism**: Can refactor later with test protection

The hybrid approach (mirror → refactor) gives us the best of both worlds: low risk initially, high quality eventually.
