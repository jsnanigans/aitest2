# Weight Processor TypeScript Library

Core infrastructure-agnostic weight processing library with Kalman filtering and quality scoring.

This is a 1:1 port of the Python `python_lib` implementation, maintaining exact algorithmic parity.

## Features

- **Adaptive Kalman Filtering**: Sophisticated weight tracking with trend detection
- **Multi-Component Quality Scoring**: Physiological plausibility, temporal consistency, statistical validation
- **Storage Abstraction**: Pluggable storage backends (in-memory included)
- **Reset Management**: Handles significant weight changes and state recovery
- **Circuit Breaker**: Failure protection for stability
- **Type-Safe**: Full TypeScript type definitions

## Installation

```bash
bun install
```

## Usage

```typescript
import { WeightProcessor } from '@weight-processor/lib';
import { InMemoryStore } from '@weight-processor/lib/core/database/memory_store';

const store = new InMemoryStore();
const processor = new WeightProcessor(store);

const result = await processor.process({
  userId: 'user123',
  weight: 70.5,
  timestamp: new Date(),
  source: 'patient-device',
  unit: 'kg'
});
```

## Testing

```bash
# Run all tests
bun test

# Run with watch mode
bun test --watch

# Type checking
bun run typecheck
```

## Directory Structure

```
typescript_lib/
├── src/
│   └── weight-processor-lib/
│       └── core/
│           ├── processing/      # Core processing logic
│           ├── database/         # Storage abstraction
│           ├── constants.ts      # Configuration constants
│           ├── exceptions.ts     # Custom exceptions
│           └── utils.ts          # Shared utilities
├── tests/                        # Unit tests
└── package.json
```

## Architecture

This library is infrastructure-agnostic and can be used in:
- Node.js/Bun backend services
- Browser applications
- Edge functions (Cloudflare Workers, Vercel Edge, etc.)
- CLI tools
- Batch processing jobs

## License

MIT
