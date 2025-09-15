# Plan: Phase 0 - Architectural Preparation

## Summary
Prepare the codebase architecture for API service transformation by introducing abstraction layers, configuration management, and interface contracts without breaking existing functionality.

## Context
- Current state: Tightly coupled to CLI/CSV processing
- Target state: Service-ready architecture with pluggable components
- Constraint: Maintain backward compatibility during transition

## Requirements
- Create abstraction layer for state persistence
- Separate configuration from implementation
- Define clear interface contracts
- Enable dependency injection pattern
- Maintain existing CLI functionality

## Alternatives

### Option A: Interface-Based Abstraction
**Approach**: Create abstract base classes for all pluggable components
- Pros: Clear contracts, type safety, easy testing
- Cons: More boilerplate, Python ABC overhead
- Risks: Over-abstraction

### Option B: Protocol-Based Design
**Approach**: Use Python protocols for duck typing
- Pros: Flexible, Pythonic, less boilerplate
- Cons: Less explicit contracts, runtime errors
- Risks: Interface drift

### Option C: Configuration-Driven
**Approach**: Use configuration to select implementations
- Pros: Simple, no code changes for switching
- Cons: Runtime resolution, harder to test
- Risks: Configuration complexity

## Recommendation
**Option A: Interface-Based Abstraction** - Provides clearest path for multiple implementations

## High-Level Design

### New Architecture Layers
```
API Layer (future)
    ↓
Service Layer (new)
    ↓
Processing Core (existing)
    ↓
Abstraction Layer (new)
    ↓
Implementation Layer (refactored)
```

### Key Abstractions
1. **StateStore** - Abstract interface for state persistence
2. **ConfigProvider** - Configuration management interface
3. **MetricsCollector** - Metrics/logging abstraction
4. **ProcessingResult** - Standardized result object

## Implementation Plan (No Code)

### Step 1: Create Abstract Base Classes
**File**: `src/interfaces.py` (new)
- Define `AbstractStateStore` with methods:
  - `get_state(user_id) -> Optional[Dict]`
  - `save_state(user_id, state) -> None`
  - `delete_state(user_id) -> bool`
  - `create_initial_state() -> Dict`
  - `batch_get_states(user_ids) -> Dict`
  - `batch_save_states(states) -> None`
- Define `AbstractConfigProvider`:
  - `get_config(key) -> Any`
  - `get_kalman_config() -> Dict`
  - `get_quality_config() -> Dict`
  - `reload() -> None`
- Define `AbstractMetricsCollector`:
  - `record_measurement(metric_name, value, tags)`
  - `increment_counter(counter_name, tags)`
  - `record_timing(operation_name, duration, tags)`

### Step 2: Refactor ProcessorStateDB
**File**: `src/database.py`
- Make `ProcessorStateDB` inherit from `AbstractStateStore`
- Keep existing implementation as `InMemoryStateStore`
- Add factory method `create_state_store(type='memory', **kwargs)`
- Ensure all methods match abstract interface

### Step 3: Create Configuration Manager
**File**: `src/config_manager.py` (new)
- Implement `TOMLConfigProvider(AbstractConfigProvider)`
- Implement `EnvironmentConfigProvider(AbstractConfigProvider)`
- Implement `CompositeConfigProvider` for layered config
- Add config validation and schema checking
- Support hot-reloading for development

### Step 4: Standardize Result Objects
**File**: `src/models.py` (new)
- Create `ProcessingResult` dataclass:
  - Core fields: accepted, quality_score, timestamp
  - Metadata: kalman_state, rejection_reason, components
  - Serialization methods: to_dict(), to_json()
- Create `ValidationResult` dataclass
- Create `QualityScoreResult` dataclass
- Ensure all results are JSON-serializable

### Step 5: Introduce Dependency Injection
**File**: `src/container.py` (new)
- Create `ServiceContainer` class:
  - Register and resolve dependencies
  - Support singleton and transient lifetimes
  - Enable testing with mock implementations
- Update `processor.py` to accept injected dependencies:
  - Accept `state_store` parameter (default to current)
  - Accept `config_provider` parameter
  - Accept `metrics_collector` parameter

### Step 6: Create Service Layer
**File**: `src/service.py` (new)
- Create `WeightValidationService` class:
  - Wraps `process_measurement()`
  - Handles dependency injection
  - Provides high-level API methods
  - Manages transactions and error handling
- Methods:
  - `validate_single(user_id, weight, timestamp, source)`
  - `validate_batch(measurements)`
  - `get_user_state(user_id)`
  - `reset_user_state(user_id)`

### Step 7: Add Metrics and Logging
**File**: `src/metrics.py` (new)
- Implement `ConsoleMetricsCollector` (for CLI)
- Implement `CloudWatchMetricsCollector` (for Lambda)
- Implement `NoOpMetricsCollector` (for testing)
- Add structured logging with correlation IDs

### Step 8: Update Entry Points
**File**: `main.py`
- Refactor to use `ServiceContainer`
- Initialize appropriate implementations
- Maintain backward compatibility
- Add `--backend` flag for state store selection

## Validation & Testing

### Test Strategy
- Unit tests for each abstraction
- Integration tests with different implementations
- Regression tests for existing functionality
- Performance benchmarks

### Verification Checklist
- [ ] Existing CLI still works identically
- [ ] All tests pass without modification
- [ ] New abstractions have 100% test coverage
- [ ] No performance degradation
- [ ] Configuration loading works correctly

## Risks & Mitigations

### Risk 1: Breaking Changes
- **Impact**: Existing scripts/workflows fail
- **Mitigation**: Extensive regression testing
- **Monitoring**: Run parallel validation

### Risk 2: Performance Overhead
- **Impact**: Slower processing
- **Mitigation**: Profile and optimize hot paths
- **Monitoring**: Benchmark key operations

### Risk 3: Complexity Increase
- **Impact**: Harder to maintain
- **Mitigation**: Clear documentation and examples
- **Monitoring**: Code review feedback

## Acceptance Criteria
- [ ] All abstractions defined and documented
- [ ] Existing functionality unchanged
- [ ] New service layer operational
- [ ] Dependency injection working
- [ ] Configuration management flexible
- [ ] Metrics collection abstracted

## Out of Scope
- Actual API implementation
- DynamoDB implementation
- Authentication/authorization
- Network layer
- Deployment configuration

## Dependencies
- No external dependencies for this phase
- Must complete before Phase 1 (API Implementation)

## Migration Path
1. Create new files without touching existing
2. Add abstraction layers incrementally
3. Refactor existing code to use abstractions
4. Verify each step with tests
5. Document changes for team

## Review Notes
- Keep changes minimal and focused
- Prioritize interface stability
- Enable future extensions
- Maintain simplicity where possible