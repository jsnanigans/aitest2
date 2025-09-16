# Plan: Multi-threading Visualization Generation

## Decision
**Approach**: Use ThreadPoolExecutor to parallelize visualization generation loop only
**Why**: Visualization generation is CPU-bound, independent per user, and currently sequential bottleneck
**Risk Level**: Low

## Implementation Steps

### 1. Add Thread Pool Configuration
**File**: `config.toml`
```toml
[visualization.threading]
enabled = true
max_workers = 4  # Default, will be capped by CPU count
batch_size = 10  # Progress reporting batch
timeout_seconds = 30  # Per-visualization timeout
```

### 2. Import Required Modules
**File**: `main.py:425` (after visualization enabled check)
```python
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from threading import Lock
import multiprocessing
```

### 3. Extract Visualization Function
**File**: `main.py:445` (before the loop)
Create standalone function for single user visualization:
```python
def generate_single_visualization(args):
    """Thread-safe single user visualization"""
    user_id, results, viz_dir, config, idx, total = args
    try:
        from src.viz.visualization import create_weight_timeline
        dashboard_path = create_weight_timeline(
            results, user_id, str(viz_dir), config=config
        )
        return (idx, user_id, dashboard_path, None)
    except Exception as e:
        return (idx, user_id, None, str(e))
```

### 4. Replace Sequential Loop with Thread Pool
**File**: `main.py:448-473`
Replace existing loop with:
```python
# Thread pool configuration
max_workers = min(
    config.get("visualization", {}).get("threading", {}).get("max_workers", 4),
    multiprocessing.cpu_count(),
    len(user_results)  # No more threads than users
)

# Progress tracking
progress_lock = Lock()
successful = 0
failed = 0

# Prepare work items
work_items = [
    (user_id, results, viz_dir, config, idx, total_users)
    for idx, (user_id, results) in enumerate(user_results.items(), 1)
]

# Execute with thread pool
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    # Submit all tasks
    futures = {
        executor.submit(generate_single_visualization, item): item[0]
        for item in work_items
    }

    # Process completions
    for future in as_completed(futures):
        idx, user_id, dashboard_path, error = future.result(timeout=30)

        with progress_lock:
            if dashboard_path:
                successful += 1
            else:
                failed += 1

            # Progress reporting
            if total_users > 10 and (successful + failed) % 10 == 0:
                print(f"  Progress: {successful+failed}/{total_users}")
            elif total_users <= 10:
                status = "✓" if dashboard_path else f"✗ ({error[:30]})"
                print(f"  [{idx}/{total_users}] User {user_id[:8]}... {status}")
```

### 5. Add Memory Management
**File**: `main.py:446` (before thread pool)
```python
# Memory guard for large batches
import psutil
available_memory_gb = psutil.virtual_memory().available / (1024**3)
estimated_per_viz_mb = 50  # Conservative estimate
max_concurrent = min(
    max_workers,
    int(available_memory_gb * 1024 / estimated_per_viz_mb / 2)  # Use half available
)
max_workers = max(1, max_concurrent)  # At least 1 thread
```

### 6. Enhanced Error Handling
**File**: `main.py` (in thread pool section)
- Wrap future.result() in try/except for TimeoutError
- Log thread pool exceptions separately
- Graceful degradation on memory pressure

## Files to Change
- `main.py:425-473` - Replace sequential loop with ThreadPoolExecutor
- `config.toml` - Add threading configuration section
- `requirements.txt` - Add `psutil>=5.9.0` for memory monitoring

## Acceptance Criteria
- [ ] Visualizations generate in parallel with configurable thread count
- [ ] Progress reporting works correctly with threading
- [ ] Individual visualization failures don't affect others
- [ ] Memory usage stays within system limits
- [ ] Performance improves by factor of min(thread_count, user_count)
- [ ] Timeout handling prevents hung threads
- [ ] Configuration allows disabling threading (enabled=false)

## Performance Expectations
- **10 users, 4 threads**: ~2.5x speedup
- **100 users, 4 threads**: ~3.8x speedup
- **100 users, 8 threads**: ~6-7x speedup (CPU-bound)
- **Memory overhead**: ~50MB per concurrent visualization

## Testing Strategy

### Unit Tests
```python
# tests/test_viz_threading.py
def test_parallel_visualization():
    # Mock create_weight_timeline
    # Submit 20 users with 4 threads
    # Verify all complete
    # Check thread safety of progress counter

def test_visualization_timeout():
    # Mock slow visualization
    # Verify timeout triggers
    # Check other visualizations continue

def test_memory_limit():
    # Mock low memory condition
    # Verify thread pool reduces size
```

### Integration Tests
1. Process small CSV (5 users) with threading
2. Process large CSV (100+ users) with threading
3. Compare output with sequential version (must be identical)
4. Inject failure in one visualization, verify others complete

### Performance Tests
```bash
# Baseline (threading disabled)
time uv run python main.py data/large_test.csv --config test_no_thread.toml

# With threading (4 workers)
time uv run python main.py data/large_test.csv --config test_thread_4.toml

# With threading (8 workers)
time uv run python main.py data/large_test.csv --config test_thread_8.toml
```

## Risks & Mitigations
**Main Risk**: Memory exhaustion with many concurrent visualizations
**Mitigation**: Dynamic thread pool sizing based on available memory, per-visualization timeout

**Secondary Risk**: Thread safety in visualization libraries (matplotlib/plotly)
**Mitigation**: Each thread creates independent figure objects, no shared state

## Out of Scope
- Parallelizing CSV processing or Kalman filtering
- Process-based parallelization (overkill for I/O-bound viz)
- Distributed processing across machines
- Caching of visualization components
- Async/await implementation (not needed for CPU-bound work)