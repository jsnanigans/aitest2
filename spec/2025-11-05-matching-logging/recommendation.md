# Recommendation: Inline Logging with Helper Functions

**Date:** 2025-11-05
**Feature:** Add detailed matching logging to Python and TypeScript weight processors

---

## Selected Solution: Option 1 - Inline Logging with Helper Functions

### Decision Summary

After evaluating three options, we recommend **Option 1: Inline Logging with Helper Functions** based on:

1. **Simplicity** - Straightforward implementation, easy to understand
2. **User preference** - Aligns with user's approval of inline approach
3. **Sufficient functionality** - Meets all requirements without over-engineering
4. **Maintainability** - Easy to modify and debug
5. **Low risk** - Minimal abstraction, fewer potential issues

**Configuration:**
- Enable via environment variable: `VERBOSE_LOGGING=true`
- Output to stdout
- Floating-point tolerance: 1e-6 for comparisons

---

## Implementation Approach

### Core Components

1. **Formatting Helper Functions**
   - `_format_num(value)` - Format numbers to 6 decimal places
   - `_format_vec(vector)` - Format state vectors consistently
   - `_log(message)` - Conditional logging based on environment variable

2. **Inline Logging Calls**
   - Add logging at each major processing step
   - Use consistent step markers: "Step 1:", "Step 2:", etc.
   - Include visual separators between measurements

3. **Configuration**
   - Read `VERBOSE_LOGGING` environment variable
   - Default: disabled (no logging overhead)
   - Enable for testing/debugging

### Python Implementation

```python
# At top of processor.py
import os
import logging

logger = logging.getLogger(__name__)
VERBOSE_LOGGING = os.getenv("VERBOSE_LOGGING", "false").lower() == "true"

def _log(message: str):
    """Log processing step if verbose logging enabled."""
    if VERBOSE_LOGGING:
        logger.info(f"[PY] {message}")

def _format_num(value: float | None) -> str:
    """Format number to 6 decimal places."""
    if value is None:
        return "None"
    return f"{float(value):.6f}"

def _format_vec(vec) -> str:
    """Format state vector/array."""
    if vec is None:
        return "None"
    # Handle numpy arrays and lists
    if hasattr(vec, 'flatten'):
        flat = vec.flatten()
    elif isinstance(vec, list):
        flat = vec if not isinstance(vec[0], list) else [item for sublist in vec for item in sublist]
    else:
        flat = vec
    return f"[{', '.join(_format_num(float(v)) for v in flat)}]"

# In process_measurement function
def process_measurement(...):
    _log("=" * 80)
    _log(f"Processing measurement for user {user_id[:12]}...")
    _log(f"  Weight: {_format_num(weight)} {unit}")
    _log(f"  Timestamp: {timestamp.isoformat()}")
    _log(f"  Source: {source}")

    _log("Step 1: Data cleaning and preprocessing")
    # ... processing code ...
    _log(f"  Cleaned weight: {_format_num(cleaned_weight)}")

    # Continue for each step...
```

### TypeScript Implementation

```typescript
// At top of processor.ts
const VERBOSE_LOGGING = process.env.VERBOSE_LOGGING === "true";

function _log(message: string): void {
    if (VERBOSE_LOGGING) {
        console.log(`[TS] ${message}`);
    }
}

function _formatNum(value: number | null | undefined): string {
    if (value === null || value === undefined) {
        return "null";
    }
    return value.toFixed(6);
}

function _formatVec(vec: number[][] | number[] | null | undefined): string {
    if (!vec) return "null";
    const flat = Array.isArray(vec[0])
        ? (vec as number[][]).flat()
        : vec as number[];
    return `[${flat.map(v => v.toFixed(6)).join(', ')}]`;
}

// In processMeasurement function
export async function processMeasurement(...) {
    _log("=".repeat(80));
    _log(`Processing measurement for user ${userId.substring(0, 12)}...`);
    _log(`  Weight: ${_formatNum(weight)} ${unit}`);
    _log(`  Timestamp: ${timestamp.toISOString()}`);
    _log(`  Source: ${source}`);

    _log("Step 1: Data cleaning and preprocessing");
    // ... processing code ...
    _log(`  Cleaned weight: ${_formatNum(cleaned_weight)}`);

    // Continue for each step...
}
```

---

## Key Design Decisions

### 1. Environment Variable Over CLI Flag

**Decision:** Use environment variable `VERBOSE_LOGGING`

**Rationale:**
- Works for both implementations consistently
- No need to modify CLI argument parsing in both scripts
- Easy to enable/disable for testing: `VERBOSE_LOGGING=true uv run python local_main.py ...`
- Can be set in shell session for multiple runs

**Usage:**
```bash
# Python
VERBOSE_LOGGING=true uv run python local_main.py --csv-file test_user.csv ...

# TypeScript
VERBOSE_LOGGING=true bun run local_main.ts --csv-file test_user.csv ...
```

### 2. Stdout for Log Output

**Decision:** Use stdout for all log output

**Rationale:**
- Easy to capture: `command > logs.txt 2>&1`
- Can be piped to other tools: `command | grep "Step 1"`
- Consistent with existing console.log usage in TypeScript
- Python logger.info goes to stdout by default

### 3. Floating-Point Tolerance of 1e-6

**Decision:** Accept differences < 1e-6 in numeric comparisons

**Rationale:**
- Six decimal places precision in logs
- Accounts for floating-point arithmetic differences
- Stricter than typical (1e-4) but reasonable for weight measurements
- Can be tightened or relaxed based on findings

### 4. Prefix for Implementation Identification

**Decision:** Use `[PY]` and `[TS]` prefixes

**Rationale:**
- Clear identification of which implementation produced each log
- Makes side-by-side comparison easier
- Enables filtering: `grep "\[PY\]"` or `grep "\[TS\]"`
- Short and unobtrusive

---

## Logging Coverage

### Processing Steps to Log

Each measurement will have logs for:

1. **Input header**
   - User ID, weight, unit, timestamp, source

2. **Step 1: Preprocessing**
   - Cleaned weight value
   - Rejection reason if failed

3. **Step 2: State management**
   - State exists or created
   - Previous weight, timestamp, Kalman status

4. **Step 3: Reset detection**
   - Reset check performed
   - Reset type and reason if triggered

5. **Step 4: Kalman initialization** (if needed)
   - Adaptive config parameters
   - Observation covariance
   - Initial state vector

6. **Step 5: Quality scoring**
   - Kalman prediction
   - Innovation covariance
   - Overall quality score
   - Rejection reason if failed

7. **Step 6: Kalman update** (if not already done)
   - Observation covariance
   - Updated state vector
   - Trend limiting applied

8. **Final result**
   - Acceptance status
   - Kalman estimate and uncertainty
   - Quality score
   - Stage reached

---

## Testing Strategy

### Phase 1: Implementation
1. Add helper functions to both implementations
2. Add inline logging calls at each step
3. Verify code compiles/runs without syntax errors

### Phase 2: Initial Testing
1. Enable logging via environment variable
2. Run Python with test_user.csv
3. Run TypeScript with test_user.csv
4. Check that logs are produced

### Phase 3: Comparison
1. Capture logs to files:
   ```bash
   VERBOSE_LOGGING=true uv run python local_main.py \
     --csv-file test_user.csv \
     --user-ids ADC64C0B-CB46-41F9-BDA0-CC11A35942D7 \
     --min-readings 0 \
     --filtered-csv filtered_py.csv \
     > logs_py.txt 2>&1

   VERBOSE_LOGGING=true bun run local_main.ts \
     --csv-file test_user.csv \
     --user-ids ADC64C0B-CB46-41F9-BDA0-CC11A35942D7 \
     --min-readings 0 \
     --filtered-csv filtered_ts.csv \
     > logs_ts.txt 2>&1
   ```

2. Compare logs manually or with diff:
   ```bash
   # Side-by-side comparison
   diff -y logs_py.txt logs_ts.txt | less

   # Show only differences
   diff -u logs_py.txt logs_ts.txt
   ```

3. Compare CSV outputs:
   ```bash
   diff filtered_py.csv filtered_ts.csv
   ```

### Phase 4: Analysis
1. Identify any divergence points in logs
2. Verify numeric values match within tolerance (1e-6)
3. Document any differences found
4. Verify CSV outputs are identical

---

## Success Criteria

Implementation is complete when:

- ✅ Both implementations have matching logging at all steps
- ✅ Logs can be enabled via `VERBOSE_LOGGING` environment variable
- ✅ Logs output to stdout with `[PY]` and `[TS]` prefixes
- ✅ Numeric values formatted to 6 decimal places
- ✅ Both implementations run successfully with test_user.csv
- ✅ Output CSVs are identical
- ✅ Logs contain same information at same steps

---

## Risks and Mitigations

### Risk: Logging affects processing logic
**Mitigation:** Isolate logging in pure functions, no side effects

### Risk: Numeric formatting differs between implementations
**Mitigation:** Use same format string/method (`.6f` / `.toFixed(6)`)

### Risk: State vector formatting inconsistent
**Mitigation:** Flatten all arrays/matrices before formatting

### Risk: Timestamp formats differ
**Mitigation:** Use ISO format (`.isoformat()` / `.toISOString()`)

---

## Next Steps

Proceed to implementation plan with:
1. Detailed task breakdown for Python implementation
2. Detailed task breakdown for TypeScript implementation
3. Testing and comparison procedures
4. Estimated time and complexity for each task
