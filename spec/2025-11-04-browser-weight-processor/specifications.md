# TypeScript Weight Processor - Complete Port Specifications

## Overview
Port the complete Python weight processing pipeline to TypeScript as a reusable library. The TypeScript implementation will mirror the Python codebase structure, run with Bun, and provide a CLI interface via `local_main.ts` that replicates `local_main.py` functionality. The library will be structured for npm publishing and reuse in other projects.

## Goals
1. **Complete Feature Parity**: Port all Python modules and functionality to TypeScript
2. **Library Structure**: Organize as publishable npm package with clean public API
3. **CLI Interface**: Provide `local_main.ts` for CSV processing (matching `local_main.py`)
4. **Bun Runtime**: Leverage Bun for fast execution and TypeScript support
5. **Type Safety**: Full TypeScript with strict mode, comprehensive types
6. **Reusability**: Design for easy import and use in other TypeScript/JavaScript projects

## Scope

### In Scope - Complete Codebase Port

#### Core Processing (`src/core/processing/`)
- ✅ `processor.ts` - Main processing orchestrator
- ✅ `kalman.ts` - Adaptive Kalman filter with state management
- ✅ `kalman_filter.ts` - Core Kalman filter implementation
- ✅ `unified_quality_scorer.ts` - 5-component quality scoring system
- ✅ `reset_manager.ts` - Reset type detection and parameter management
- ✅ `kalman_state_validator.ts` - State validation logic
- ✅ `outlier_detection.ts` - Outlier detection algorithms
- ✅ `validation.ts` - Input validation
- ✅ `type_conversion.ts` - Type conversion utilities
- ✅ `circuit_breaker.ts` - Circuit breaker pattern
- ✅ `buffer_factory.ts` - Buffer factory pattern
- ✅ `state_validator.ts` - State validation
- ✅ `persistence_validator.ts` - Persistence validation
- ✅ `reset_transaction.ts` - Reset transaction management

#### State Storage Layer (`src/core/database/`)
**NOTE: Pure in-memory storage only - NO database integration**
- ✅ `base.ts` - Abstract StateStore interface
- ✅ `database.ts` - In-memory state storage using JavaScript Maps/objects
- ✅ `db_wrapper.ts` - State storage utilities (no actual database calls)

#### Replay System (`src/core/replay/`)
- ✅ `replay_manager.ts` - State recovery and replay orchestration
- ✅ `replay_buffer.ts` - Measurement buffering
- ✅ `replay_processor.ts` - Replay processing logic
- ✅ `temporal_consistency_analyzer.ts` - Temporal consistency analysis
- ✅ `enhanced_replay_analyzer.ts` - Enhanced replay analysis
- ✅ `sliding_window_processor.ts` - Sliding window processing

#### Configuration & Constants (`src/`)
- ✅ `constants.ts` - All constants (physiological limits, source profiles, etc.)
- ✅ `config/config_manager.ts` - Configuration loading from TOML
- ✅ `utils.ts` - Utility functions
- ✅ `exceptions.ts` - Custom exception types

#### Models (`src/`)
- ✅ `models.ts` - Data models (Measurement, ProcessResult, etc.)

#### Services (`src/services/`)
- ✅ `weight_processor_service.ts` - Service layer for batch processing

#### CLI (`local_main.ts`)
- ✅ CSV file reading and parsing
- ✅ User filtering (max-users, max-rows, min-readings, specific user IDs)
- ✅ Batch processing with progress tracking
- ✅ Acceptance tracking
- ✅ Filtered CSV output (accepted measurements only)
- ✅ Summary statistics and reporting
- ✅ Command-line argument parsing
- ✅ Data quality statistics

### Out of Scope (for initial port)
- ❌ AWS Lambda handlers (focus on library, not deployment)
- ❌ DynamoDB backend (use in-memory only)
- ❌ Visualization (no Plotly equivalent needed)
- ❌ HTML dashboard generation
- ❌ SAM templates and AWS infrastructure
- ❌ API endpoints (can be added later if needed)

## Architecture

### Directory Structure
```
weight-processor-ts/
├── src/
│   ├── core/
│   │   ├── processing/
│   │   │   ├── processor.ts
│   │   │   ├── kalman.ts
│   │   │   ├── kalman_filter.ts
│   │   │   ├── unified_quality_scorer.ts
│   │   │   ├── reset_manager.ts
│   │   │   ├── kalman_state_validator.ts
│   │   │   ├── outlier_detection.ts
│   │   │   ├── validation.ts
│   │   │   ├── type_conversion.ts
│   │   │   ├── circuit_breaker.ts
│   │   │   ├── buffer_factory.ts
│   │   │   ├── state_validator.ts
│   │   │   ├── persistence_validator.ts
│   │   │   ├── reset_transaction.ts
│   │   │   └── index.ts
│   │   ├── database/
│   │   │   ├── base.ts
│   │   │   ├── database.ts
│   │   │   ├── db_wrapper.ts
│   │   │   └── index.ts
│   │   ├── replay/
│   │   │   ├── replay_manager.ts
│   │   │   ├── replay_buffer.ts
│   │   │   ├── replay_processor.ts
│   │   │   ├── temporal_consistency_analyzer.ts
│   │   │   ├── enhanced_replay_analyzer.ts
│   │   │   ├── sliding_window_processor.ts
│   │   │   └── index.ts
│   │   ├── constants.ts
│   │   ├── utils.ts
│   │   └── exceptions.ts
│   ├── config/
│   │   ├── config_manager.ts
│   │   └── index.ts
│   ├── services/
│   │   ├── weight_processor_service.ts
│   │   └── index.ts
│   ├── models.ts
│   └── index.ts (main library export)
├── local_main.ts (CLI script)
├── config.toml (configuration file)
├── package.json
├── tsconfig.json
├── README.md
└── tests/ (unit and integration tests)
```

### Library Public API (`src/index.ts`)
```typescript
// Main exports for library consumers
export { WeightProcessorService } from './services/weight_processor_service';
export { ProcessorStateDB } from './core/database/database';
export { ConfigManager } from './config/config_manager';
export { process_measurement } from './core/processing/processor';

// Types/Interfaces
export type {
  Measurement,
  ProcessResult,
  ProcessResponseData,
  ProcessorState,
  KalmanParams,
  QualityScore,
  // ... all public types
} from './models';

// Constants
export {
  SUPPORTED_WEIGHT_UNITS,
  PHYSIOLOGICAL_LIMITS,
  SOURCE_PROFILES,
  KALMAN_DEFAULTS
} from './core/constants';
```

## Requirements

### Functional Requirements

#### FR1: Complete Python Port
- Port all Python modules maintaining equivalent functionality
- Preserve all algorithms, logic, and behaviors
- Match numerical outputs within floating-point precision tolerance
- Maintain configuration compatibility (same config.toml format)

#### FR2: CSV Processing (local_main.ts)
- **Input**:
  - Read CSV files with weight measurements
  - Support columns: `id`/`measurement_id`, `user_id`, `value_quantity`/`weight`, `unit`, `effective_date_time`/`effectiveDateTime`, `source_type`
  - Handle missing or NULL values gracefully
  - Parse various timestamp formats (ISO 8601, space-separated)
- **Filtering**:
  - `--max-users N`: Limit number of users processed
  - `--max-rows N`: Limit CSV rows read
  - `--min-readings N`: Filter users with fewer than N measurements
  - `--user-ids "id1,id2,id3"`: Process only specific users
- **Processing**:
  - Group measurements by user_id
  - Sort chronologically per user
  - Process with full Kalman filtering and quality scoring
  - Track acceptance/rejection per measurement
- **Output**:
  - Filtered CSV with only accepted measurements
  - JSON results file with processing statistics
  - Console output with progress and summary

#### FR3: In-Memory State Management
- Implement `ProcessorStateDB` class for in-memory storage
- Track state per user_id:
  - Kalman parameters and state vectors
  - Last measurement metadata
  - Reset history
  - Measurement buffer for replay
- No persistence between runs (in-memory only)

#### FR4: Configuration Management
- Load configuration from `config.toml` file
- Parse TOML format (use library like `@iarna/toml`)
- Support all config sections:
  - `[kalman]` - Kalman filter parameters
  - `[kalman.reset.initial/hard/soft]` - Reset configurations
  - `[quality_scoring]` - Quality scorer settings
  - `[sources.*]` - Source profiles
  - `[replay]` - Replay settings
  - `[processing]` - Processing settings

#### FR5: Library Usage
Enable programmatic usage:
```typescript
import { WeightProcessorService, ProcessorStateDB, ConfigManager } from 'weight-processor-ts';

const config = ConfigManager.loadConfig();
const stateStore = new ProcessorStateDB();
const service = new WeightProcessorService(stateStore, config);

const results = service.processBatch(userId, measurements);
```

### Non-Functional Requirements

#### NFR1: Performance
- Match or exceed Python performance
- Process 1000 measurements in < 5 seconds
- Process 10,000 measurements in < 30 seconds
- Leverage Bun's fast runtime

#### NFR2: Accuracy
- Numerical results must match Python within 0.1% for:
  - Kalman estimates
  - Quality scores
  - Acceptance decisions
- Validate against Python output using same input data

#### NFR3: Code Quality
- 100% TypeScript with strict mode
- No `any` types except for truly dynamic data
- Comprehensive JSDoc comments
- Follow functional programming patterns where appropriate
- Clear module boundaries with single responsibility

#### NFR4: Type Safety
```typescript
// tsconfig.json
{
  "compilerOptions": {
    "strict": true,
    "noImplicitAny": true,
    "strictNullChecks": true,
    "strictFunctionTypes": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noImplicitReturns": true
  }
}
```

#### NFR5: Testing
- Unit tests for all modules
- Integration tests for full pipeline
- Test coverage > 80%
- Validation tests comparing Python vs TypeScript outputs

#### NFR6: Package Quality
- Clean package.json with proper exports
- README with usage examples
- TypeScript declaration files (.d.ts)
- Semantic versioning
- License file (match Python project)

## Data Models

### Core Types (TypeScript)
```typescript
// models.ts

export interface Measurement {
  uuid: string; // measurement_id
  weight: number; // weight_value in kg
  unit: string;
  effectiveDateTime: Date;
  source: string;
  metadata?: Record<string, any>;
}

export interface ProcessorState {
  userId: string;
  kalmanParams: KalmanParams | null;
  lastState: number[] | null; // [weight, velocity]
  lastCovariance: number[][] | null; // 2x2 matrix
  lastTimestamp: Date | null;
  lastRawWeight: number | null;
  lastSource: string | null;
  lastAcceptedTimestamp: Date | null;
  measurementsSinceReset: number;
  resetType: ResetType | null;
  resetParameters: ResetParameters | null;
  resetTimestamp: Date | null;
  resetEvents: ResetEvent[];
  measurementHistory: MeasurementHistoryEntry[];
}

export interface KalmanParams {
  initialStateMean: number[];
  initialStateCovariance: number[][];
  transitionCovariance: number[][];
  observationCovariance: number[][];
}

export interface ProcessResult {
  measurementId: string;
  accepted: boolean;
  qualityScore: number;
  kalmanEstimate: number;
  kalmanUncertainty: number;
  rejectionReason?: string;
  processingStage: string;
  resetTriggered: boolean;
  qualityComponents: QualityComponents;
  innovation?: number;
  normalizedInnovation?: number;
  confidence?: number;
  trend?: number;
  trendWeekly?: number;
}

export interface ProcessResponseData {
  userId: string;
  measurementsProcessed: number;
  measurementsAccepted: number;
  measurementsRejected: number;
  results: ProcessResult[];
  stateUpdate: StateInfo | null;
}

export interface QualityScore {
  overall: number;
  components: QualityComponents;
  threshold: number;
  accepted: boolean;
  rejectionReason?: string;
  metadata: QualityMetadata;
}

export interface QualityComponents {
  kalmanFit: number;
  temporalConsistency: number;
  anomalyDetection: number;
  sourceReliability: number;
  trendAlignment: number;
}

export enum ResetType {
  INITIAL = 'initial',
  HARD = 'hard',
  SOFT = 'soft'
}

export interface ResetEvent {
  timestamp: Date;
  type: ResetType;
  source: string;
  weight: number;
  lastWeight?: number;
  gapDays?: number;
  reason: string;
  parameters: ResetParameters;
}

// ... additional types matching Python models
```

## Implementation Details

### Matrix Operations
Implement custom linear algebra utilities:
```typescript
// src/core/utils.ts or src/core/math/matrix.ts

export class Matrix {
  static multiply(a: number[][], b: number[][]): number[][];
  static add(a: number[][], b: number[][]): number[][];
  static transpose(a: number[][]): number[][];
  static inverse2x2(a: number[][]): number[][];
  static scalarMultiply(scalar: number, matrix: number[][]): number[][];
}
```

### Statistical Functions
```typescript
// src/core/utils.ts or src/core/math/stats.ts

export class Stats {
  static mean(values: number[]): number;
  static median(values: number[]): number;
  static variance(values: number[]): number;
  static std(values: number[]): number;
  static chi2Cdf(x: number, df: number): number; // Approximation
  static linearRegression(x: number[], y: number[]): { slope: number; intercept: number };
}
```

### CSV Processing
```typescript
// Use Bun's built-in APIs or lightweight CSV parser
import { parse } from 'csv-parse/sync'; // or similar

async function loadCsvData(csvPath: string, options: LoadOptions): Promise<{
  userMeasurements: Map<string, Measurement[]>;
  originalRows: CsvRow[];
}>;
```

### TOML Configuration
```typescript
import TOML from '@iarna/toml';

export class ConfigManager {
  static loadConfig(source: 'file' | 'env' = 'file', configPath?: string): Config {
    // Load and parse config.toml
    // Merge with environment variables if needed
  }
}
```

## CLI Interface (local_main.ts)

### Command-Line Arguments
```bash
bun run local_main.ts \
  --csv-file data/weights.csv \
  --max-users 100 \
  --max-rows 50000 \
  --min-readings 20 \
  --user-ids "user1,user2,user3" \
  --output-dir output_local \
  --filtered-csv output_local/filtered.csv \
  --config config.toml
```

### Features
- Progress indicators for large datasets
- Data quality statistics reporting
- Summary output matching Python version
- JSON results export
- Error handling and validation

## Testing Strategy

### Unit Tests
- All core modules (Kalman, Quality, Reset, Replay)
- Matrix operations
- Statistical functions
- Configuration parsing
- CSV parsing

### Integration Tests
- Full processing pipeline
- Multi-user processing
- Reset scenarios
- Replay scenarios

### Validation Tests
- Compare TypeScript output to Python output
- Use same CSV input files
- Verify numerical equivalence (within tolerance)
- Test edge cases

### Test Framework
- Bun's built-in test runner or Vitest
- Coverage reporting
- Snapshot testing for complex outputs

## Package Configuration

### package.json
```json
{
  "name": "@9amhealth/weight-processor",
  "version": "1.0.0",
  "description": "Weight measurement processing with Kalman filtering and quality scoring",
  "type": "module",
  "main": "./dist/index.js",
  "types": "./dist/index.d.ts",
  "exports": {
    ".": {
      "import": "./dist/index.js",
      "types": "./dist/index.d.ts"
    }
  },
  "bin": {
    "weight-processor": "./dist/local_main.js"
  },
  "scripts": {
    "build": "bun build src/index.ts --outdir dist --target node",
    "dev": "bun run local_main.ts",
    "test": "bun test",
    "test:coverage": "bun test --coverage",
    "typecheck": "tsc --noEmit",
    "lint": "eslint src/**/*.ts"
  },
  "dependencies": {
    "@iarna/toml": "^2.2.5",
    "csv-parse": "^5.5.0"
  },
  "devDependencies": {
    "@types/bun": "latest",
    "typescript": "^5.3.0"
  }
}
```

### tsconfig.json
```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "ESNext",
    "lib": ["ES2022"],
    "moduleResolution": "bundler",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "resolveJsonModule": true,
    "declaration": true,
    "declarationMap": true,
    "outDir": "./dist",
    "rootDir": "./src",
    "types": ["bun-types"]
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist", "tests"]
}
```

## Migration Strategy

### Phase 1: Foundation
1. Set up project structure and build configuration
2. Implement matrix operations and statistical utilities
3. Port constants and configuration management
4. Define all TypeScript types/interfaces

### Phase 2: Core Processing
1. Port Kalman filter implementation
2. Port quality scorer
3. Port reset manager
4. Port validator utilities

### Phase 3: State & Replay
1. Port database/state management (in-memory)
2. Port replay system
3. Port outlier detection

### Phase 4: Service Layer
1. Port processor.ts
2. Port weight_processor_service.ts
3. Implement error handling

### Phase 5: CLI & Integration
1. Implement local_main.ts
2. CSV reading/writing
3. Progress tracking and reporting
4. End-to-end testing

### Phase 6: Validation & Polish
1. Validate outputs against Python
2. Performance optimization
3. Documentation
4. Package preparation

## Success Criteria

1. ✅ Complete port of all Python modules to TypeScript
2. ✅ `local_main.ts` produces identical results to `local_main.py`
3. ✅ Numerical accuracy within 0.1% of Python
4. ✅ Performance matches or exceeds Python
5. ✅ Test coverage > 80%
6. ✅ Successful processing of real-world CSV datasets
7. ✅ Clean, type-safe, well-documented code
8. ✅ Publishable npm package structure
9. ✅ CLI works with Bun runtime

## Dependencies

### Runtime
- Bun (runtime environment)
- @iarna/toml (TOML parsing)
- csv-parse (CSV parsing)

### Development
- TypeScript 5.x
- Bun's built-in test runner
- ESLint (optional, for linting)

## Constraints

1. Must run with Bun (not Node.js initially)
2. No external math libraries (implement from scratch)
3. In-memory storage only (no database backend for now)
4. Match Python implementation exactly (no algorithmic changes)
5. No visualization features

## Open Questions

None - scope is clear: complete 1:1 port of Python codebase to TypeScript.
