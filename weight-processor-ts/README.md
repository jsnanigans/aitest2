# Weight Processor TypeScript

A complete TypeScript port of the Python weight processing pipeline. This library provides Kalman filtering, quality scoring, and statistical analysis for weight measurements.

## Features

- **Adaptive Kalman Filtering**: Track weight trends with intelligent state management
- **Quality Scoring**: Multi-component quality assessment system
- **Replay System**: State recovery and reprocessing capabilities
- **Reset Management**: Handle significant weight changes intelligently
- **CLI Interface**: Process CSV files with filtering and statistics
- **Type-Safe**: Full TypeScript with strict mode

## Installation

```bash
bun install @9amhealth/weight-processor
```

## Usage

### As a Library

```typescript
import { WeightProcessorService, ProcessorStateDB } from '@9amhealth/weight-processor';

const stateStore = new ProcessorStateDB();
const service = new WeightProcessorService(stateStore);

const result = await service.processSingle({
  device_id: 'device123',
  user_id: 'user456',
  weight_kg: 75.5,
  timestamp: '2025-11-05T10:00:00Z',
  source: 'QUESTIONNAIRE_ONBOARDING'
});

console.log(result);
```

### CLI Usage

```bash
# Process CSV file
bun run local_main.ts --csv-file data.csv --max-users 100

# Filter specific users
bun run local_main.ts --csv-file data.csv --user-ids user1,user2,user3

# Limit processing
bun run local_main.ts --csv-file data.csv --max-rows 1000 --min-readings 5
```

## Development

```bash
# Install dependencies
bun install

# Run tests
bun test

# Build library
bun run build

# Lint
bun run lint

# Format
bun run format
```

## Project Status

This is a complete port of the Python weight processing pipeline with feature parity.

## License

MIT
