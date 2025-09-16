# Plan: Remove Singleton Anti-Pattern from Replay Buffer

## Decision
**Approach**: Replace singleton with dependency injection using factory pattern + context manager
**Why**: Eliminates global state, improves testability, enables multiple instances for testing
**Risk Level**: Medium (affects active component but has limited usage)

## Implementation Steps

### Phase 1: Create Factory Infrastructure
1. **Create `src/replay/buffer_factory.py`** - Factory class with instance management
   - BufferFactory class with create_buffer(), get_buffer() methods
   - Support for named instances (default, test1, test2, etc.)
   - Configuration management per instance

2. **Add context manager to `src/processing/replay_buffer.py:25`** - Enable with-statement usage
   - Add __enter__ and __exit__ methods to ReplayBuffer
   - Ensure proper cleanup on context exit
   - Thread-safe resource management

### Phase 2: Migration Path (Backward Compatible)
3. **Update `src/processing/replay_buffer.py:387`** - Deprecate singleton getter
   - Keep get_replay_buffer() but add deprecation warning
   - Delegate to factory.get_buffer('default')
   - Log migration message for monitoring

4. **Create migration helper `src/replay/migration_helper.py`** - Assist code migration
   - inject_buffer() decorator for automatic injection
   - buffer_provider() context manager for scoped instances
   - Migration documentation generator

### Phase 3: Update Core Usage
5. **Modify `main.py:166-170`** - Use dependency injection
   - Import BufferFactory instead of get_replay_buffer
   - Create buffer instance at startup
   - Pass buffer to functions that need it

6. **Update processor integration** - Pass buffer as parameter
   - Modify process_measurement() signature if needed
   - Pass buffer through processing pipeline
   - Update any direct buffer access points

### Phase 4: Testing Infrastructure
7. **Create `tests/test_replay_buffer_isolation.py`** - Test isolation verification
   - Test multiple independent buffer instances
   - Verify no state sharing between instances
   - Test cleanup and resource management

8. **Add fixtures `tests/conftest.py`** - Pytest fixtures for buffers
   - @pytest.fixture for isolated_buffer
   - @pytest.fixture for buffer_factory
   - Automatic cleanup after each test

## Files to Change
- `src/processing/replay_buffer.py:25` - Add context manager support
- `src/processing/replay_buffer.py:387-392` - Deprecate singleton, use factory
- `main.py:166-170` - Replace singleton usage with factory
- `src/replay/buffer_factory.py` - NEW: Factory implementation
- `src/replay/migration_helper.py` - NEW: Migration utilities
- `tests/test_replay_buffer_isolation.py` - NEW: Isolation tests
- `tests/conftest.py` - Add buffer fixtures

## Acceptance Criteria
- [ ] Multiple ReplayBuffer instances can exist simultaneously
- [ ] Each instance maintains independent state (no data leakage)
- [ ] Tests can create isolated buffer instances
- [ ] Backward compatibility maintained during transition
- [ ] Deprecation warnings guide migration
- [ ] Zero production disruption during rollout

## Risks & Mitigations
**Main Risk**: Breaking existing code that relies on singleton behavior
**Mitigation**: Three-phase rollout with backward compatibility layer

**Secondary Risk**: Memory overhead from multiple instances
**Mitigation**: Factory limits instance count, implements cleanup policy

**Testing Risk**: Existing tests may assume singleton
**Mitigation**: Migration helper provides compatibility mode for tests

## Out of Scope
- Refactoring entire replay subsystem architecture
- Changing buffer persistence mechanism
- Modifying buffer data structures or algorithms
- Performance optimizations beyond removing singleton

## Migration Timeline
1. **Week 1**: Deploy factory with backward compatibility
2. **Week 2**: Monitor deprecation warnings, update internal usage
3. **Week 3**: Update tests to use new pattern
4. **Week 4**: Remove deprecated singleton (optional, can keep longer)

## Code Examples

### Factory Pattern
```python
# src/replay/buffer_factory.py
class BufferFactory:
    def __init__(self):
        self._instances = {}

    def create_buffer(self, name='default', config=None):
        if name in self._instances:
            raise ValueError(f"Buffer '{name}' already exists")
        self._instances[name] = ReplayBuffer(config)
        return self._instances[name]

    def get_buffer(self, name='default'):
        return self._instances.get(name)
```

### Context Manager Usage
```python
# New usage pattern
with buffer_factory.create_buffer('test1') as buffer:
    buffer.add_measurement(user_id, measurement)
    # Buffer automatically cleaned up
```

### Dependency Injection
```python
# main.py updated pattern
def process_with_replay(measurement, buffer: ReplayBuffer):
    """Process with explicit buffer dependency"""
    buffer.add_measurement(measurement['user_id'], measurement)
```

## Testing Strategy
- Unit tests: Verify factory creates independent instances
- Integration tests: Ensure buffer works in processing pipeline
- Isolation tests: Confirm no state leakage between instances
- Performance tests: Measure memory impact of multiple instances
- Migration tests: Verify backward compatibility layer works