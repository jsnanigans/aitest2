# WORKFLOW.md - Engineering Problem-Solving Workflow

## A Systematic Approach to Feature Implementation and Bug Fixing

This document captures proven workflows for implementing features, fixing bugs, and improving systems based on real-world problem-solving sessions.

## 1. Problem Discovery & Analysis

### Listen to User Observations
Users often spot patterns that automated tests miss:
- "This user is interesting: [ID], the baseline failed because..."
- "I noticed that when X happens, Y doesn't work"
- Pay attention to specific examples - they often reveal systemic issues

### Investigate Specific Cases First
```python
# Always start with the concrete example
# Find the specific user/case that's failing
python -c "
import csv
from datetime import datetime

# Analyze the specific case
with open('data.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['id'] == 'SPECIFIC_CASE_ID':
            # Detailed analysis
"
```

### Pattern Recognition
1. **Single case → Pattern → System fix**
   - Start with one failing case
   - Find similar cases
   - Identify the common pattern
   - Design a systemic solution

2. **Data-driven discovery**
   ```bash
   # Find all similar cases
   grep -c "similar_pattern" data.csv
   
   # Analyze distribution
   cat data.csv | cut -d',' -f3 | sort | uniq -c | sort -rn
   ```

## 2. Solution Design Philosophy

### Incremental Enhancement
Rather than rebuilding, enhance what exists:

```python
# GOOD: Add fallback logic to existing function
def establish_baseline(self, readings):
    result = self.primary_approach(readings)
    
    # Enhancement: Add fallback
    if not result.success and len(readings) >= min_required:
        result = self.fallback_approach(readings)
    
    return result

# AVOID: Complete rewrite unless absolutely necessary
```

### State Machine Thinking
Complex behaviors often need state machines:

```python
# States for complex processes
class ProcessState:
    NORMAL = 'normal'
    COLLECTING = 'collecting'  
    PENDING = 'pending'        # Retry state
    FAILED = 'failed'

# State transitions with clear conditions
if state == ProcessState.COLLECTING:
    if has_enough_data():
        state = ProcessState.NORMAL
    elif exceeded_timeout():
        state = ProcessState.FAILED
    else:
        state = ProcessState.PENDING  # Keep trying
```

## 3. Implementation Patterns

### The Retry Pattern
When operations can fail due to insufficient data:

```python
class RetryableOperation:
    def __init__(self):
        self.attempts = 0
        self.max_attempts = 10
        self.pending_data = []
    
    def process(self, new_data):
        self.pending_data.append(new_data)
        
        if self.should_attempt():
            result = self.try_operation()
            if result.success:
                return result
            elif self.attempts >= self.max_attempts:
                return self.fallback_strategy()
            else:
                self.attempts += 1
                # Keep collecting, will retry next time
```

### The Fallback Pattern  
Multiple strategies for robustness:

```python
def robust_operation(data):
    # Strategy 1: Ideal approach
    result = primary_strategy(data)
    if result.success:
        return result
    
    # Strategy 2: Relaxed constraints
    logger.info("Primary failed, trying fallback")
    result = fallback_strategy(data)
    if result.success:
        return result
    
    # Strategy 3: Best effort
    logger.warning("All strategies failed, using defaults")
    return default_result()
```

## 4. Debugging Workflow

### Progressive Isolation
1. **Start broad**: Check if feature works at all
2. **Narrow down**: Find specific failing cases
3. **Isolate**: Test with just that case
4. **Expand**: Verify fix works broadly

```bash
# 1. Broad check
python main.py 2>&1 | grep "feature"

# 2. Find failures
grep "ERROR\|FAIL\|Warning" output.log | head -20

# 3. Isolate specific case
echo 'test_case = "SPECIFIC_ID"' > test_config.toml
python main.py --config test_config.toml

# 4. Verify broadly
python main.py --all-cases
```

### Log Analysis Patterns
```bash
# Find patterns in failures
grep "failed" app.log | cut -d: -f2 | sort | uniq -c

# Track specific entity through pipeline
grep "USER_123" app.log | grep -E "start|process|complete|error"

# Time-based analysis
grep "2024-01-15" app.log | grep "ERROR" | wc -l

# Before/after comparison
diff <(grep "baseline" old.log) <(grep "baseline" new.log)
```

## 5. Testing Strategy

### Edge Case Discovery
Look for:
- **Boundary conditions**: First/last items, empty sets, single items
- **Sparse data**: Users with 1 reading, long gaps, irregular patterns  
- **Extreme values**: Min/max values, outliers, invalid data
- **Timing issues**: Race conditions, timeouts, ordering problems

```python
# Systematic edge case testing
edge_cases = {
    'single_item': [item],
    'empty': [],
    'max_size': [item] * MAX_SIZE,
    'gaps': [item1, None, None, item2],
    'duplicates': [item, item, item],
    'boundaries': [MIN_VAL, MAX_VAL]
}

for case_name, test_data in edge_cases.items():
    result = process(test_data)
    assert result.valid, f"Failed on {case_name}"
```

### Progressive Validation
```python
# Level 1: Smoke test (does it run?)
assert system.start() == True

# Level 2: Unit test (does component work?)
assert component.process(simple_input) == expected_output  

# Level 3: Integration (do components work together?)
result = pipeline.process_end_to_end(test_data)
assert result.success

# Level 4: Performance (is it still fast?)
start = time.time()
process_large_dataset()
assert time.time() - start < MAX_TIME

# Level 5: Real data (does it work in production?)
results = process_production_data_sample()
assert results.accuracy > 0.95
```

## 6. Performance Preservation

### Measure Before and After
```python
# Always benchmark before changes
baseline_performance = measure_performance()

# Make changes
implement_feature()

# Verify performance maintained
new_performance = measure_performance()
assert new_performance >= baseline_performance * 0.95  # Allow 5% degradation max
```

### Optimization Patterns
```python
# Cache expensive operations
@lru_cache(maxsize=128)
def expensive_calculation(input):
    return complex_computation(input)

# Batch operations
def process_batch(items):
    # Process together instead of one-by-one
    return batch_operation(items)

# Early termination
def search_with_cutoff(data, target):
    for i, item in enumerate(data):
        if found(item, target):
            return item
        if i > CUTOFF:  # Stop searching after threshold
            return None
```

## 7. State Management in Streaming Systems

### Minimal State Principle
```python
class StreamProcessor:
    def __init__(self):
        # Only track essential state
        self.current_id = None
        self.last_timestamp = None
        self.accumulator = []  # Limited buffer
        
    def process(self, record):
        # Process and immediately output
        result = transform(record)
        
        # Update minimal state
        self.last_timestamp = record.timestamp
        
        # Don't store unless absolutely necessary
        return result
```

### State Transitions
```python
def handle_state_transition(self, event):
    old_state = self.state
    
    # Clear state machine
    if self.state == 'COLLECTING':
        if event == 'ENOUGH_DATA':
            self.state = 'PROCESSING'
        elif event == 'TIMEOUT':
            self.state = 'RETRY'
    elif self.state == 'RETRY':
        if event == 'NEW_DATA':
            self.state = 'COLLECTING'
        elif event == 'MAX_RETRIES':
            self.state = 'FAILED'
    
    if old_state != self.state:
        logger.info(f"State transition: {old_state} → {self.state}")
```

## 8. Real-World Problem Solving Examples

### Example 1: Sparse Data Handling
**Problem**: System fails when user has only 1 reading in initial window

**Discovery Process**:
1. User reports specific ID failing
2. Investigate that user's data pattern
3. Find they have 1 reading, then gap, then many readings
4. Realize this is common pattern

**Solution**:
```python
# Add retry mechanism
if baseline_failed and reason == 'insufficient_data':
    enter_pending_state()
    # Will retry with each new reading
    
# Add fallback strategy  
if window_approach_failed:
    try_first_n_readings_approach()
```

### Example 2: Gap Detection
**Problem**: System doesn't handle data gaps well

**Discovery Process**:
1. Notice Kalman predictions drift after gaps
2. Find users with 30+ day gaps
3. Realize need to reset after long gaps

**Solution**:
```python
def detect_gap(current_date, last_date):
    if not last_date:
        return True  # First reading
    gap_days = (current_date - last_date).days
    return gap_days >= GAP_THRESHOLD

if detect_gap(date, self.last_date):
    trigger_rebaseline()
    reset_kalman_filter()
```

## 9. Communication & Documentation

### Progress Tracking
Use TodoWrite extensively:
```python
todos = [
    {"task": "Understand the problem", "status": "done"},
    {"task": "Find edge cases", "status": "in_progress"},
    {"task": "Implement solution", "status": "pending"},
    {"task": "Test with edge cases", "status": "pending"},
    {"task": "Verify performance", "status": "pending"}
]
```

### Success Metrics
Always define what success looks like:
```python
success_metrics = {
    'functionality': 'All edge cases handled',
    'performance': 'Maintains 2-3 users/second',
    'reliability': '100% of users get baselines',
    'compatibility': 'No breaking changes'
}
```

### Documentation Updates
After implementation:
1. Update technical docs with new behavior
2. Add configuration parameters with comments
3. Document edge cases and how they're handled
4. Include examples of before/after

## 10. Continuous Improvement Mindset

### Pattern Collection
Keep a log of:
- Problems encountered
- Solutions that worked
- Patterns that repeat
- Performance improvements

### Refactoring Opportunities
While fixing bugs, note:
- Code that could be simplified
- Repeated patterns that could be extracted
- State management that could be cleaner
- Performance bottlenecks for later

### Knowledge Transfer
Document your debugging process:
```markdown
## Problem: [Description]
## Found by: [How it was discovered]
## Root cause: [Why it happened]
## Solution: [What fixed it]
## Prevention: [How to avoid in future]
## Similar issues: [Related problems]
```

## Key Principles

1. **Start with specific, expand to general** - Debug one case, fix for all
2. **Enhance don't replace** - Add capabilities to existing code
3. **Fail gracefully** - Always have fallback strategies
4. **Measure everything** - Performance, accuracy, success rates
5. **State machines for complex flows** - Make states explicit
6. **Progressive testing** - Start simple, add complexity
7. **Document the why** - Not just what changed, but why

## Lessons from This Session

### User Observations Lead to Major Improvements
In this session, three user observations led to significant architectural improvements:

1. **"This user is interesting: 04FC553EA92041A9A85A91BE5C3AB212"**
   - Led to: Baseline retry logic with BASELINE_PENDING state
   - Result: 100% baseline establishment (up from 50%)

2. **"Why is baseline not used when Kalman begins?"**
   - Led to: Immediate Kalman processing from first reading
   - Result: 100% Kalman coverage (vs skipping first 7 days)

3. **"That baseline line makes me wonder what it actually is"**
   - Led to: Baseline visualization overhaul
   - Result: Clear baseline markers at establishment points

### Architecture Simplification Pattern
Sometimes the best improvement is removing complexity:
- **Before**: Wait for baseline → Initialize Kalman → Skip early data → Reprocess
- **After**: Start Kalman immediately → Process everything → Enhance with baseline

This reduced code complexity while improving coverage from 50% to 100%.

### Visual Feedback Importance
Dashboard confusion revealed architectural issues:
- Misleading "Signup" label → Revealed assumption about questionnaire source
- Horizontal baseline line → Showed misunderstanding of baseline purpose
- "Kalman Start" marker → Indicated unnecessary coupling

Fixing visualization often requires fixing underlying architecture.

## Conclusion

Successful problem-solving in complex systems requires:
- **Listen to users** - Specific observations reveal systemic issues
- **Question assumptions** - "Why do we wait?" led to major improvements
- **Systematic investigation** - From specific case to general pattern
- **Incremental solutions** - Enhance rather than rebuild
- **Robust strategies** - Multiple approaches with fallbacks
- **Performance awareness** - Measure and maintain speed
- **Clear state management** - Explicit states and transitions
- **Comprehensive testing** - Edge cases and progressive validation
- **Visual clarity** - Good visualization reveals architectural truth

This workflow turns ad-hoc debugging into systematic improvement, ensuring each problem solved makes the system more robust for the future.