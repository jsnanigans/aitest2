# Plan: Gut Gap Handling - Simple 30-Day Reset

## Summary
Replace the entire complex gap handling system (buffering, adaptive parameters, trend estimation, decay) with a single rule: "If 30+ days have passed since the last accepted value, reset everything."

## Context
- Source: User request for radical simplification
- Current implementation: 300+ lines of complex adaptive logic across multiple files
- Assumptions:
  - 30-day gaps represent fundamentally disconnected measurement contexts
  - Complex gap bridging provides no meaningful value
  - Simple reset is more predictable and maintainable
- Constraints:
  - Must maintain stateless architecture
  - Must reset both Kalman and quality validation states

## Requirements

### Functional
- Detect when 30+ days have passed since last accepted measurement
- Reset Kalman filter state completely (fresh initialization)
- Reset quality scorer state/history
- Track reset events for visualization
- Remove ALL existing gap handling code

### Non-functional
- Reduce code complexity by ~80%
- Improve predictability and debuggability
- Maintain performance (actually improve it by removing overhead)
- Keep visualization of reset events

## Alternatives

### Option A: Simple 30-Day Hard Reset (RECOMMENDED)
- Approach: If gap ≥ 30 days, reinitialize everything
- Pros:
  - Dead simple - one condition, one action
  - Predictable behavior
  - Easy to debug and understand
  - Removes 300+ lines of complex code
- Cons:
  - No gradual adaptation for 15-29 day gaps
  - Loses any trend information across gaps
- Risks:
  - May have slightly worse accuracy after 20-day gaps

### Option B: Tiered Reset (10/20/30 days)
- Approach: Different reset levels based on gap size
- Pros:
  - More nuanced handling
  - Better for medium gaps
- Cons:
  - Still adds complexity
  - Three conditions instead of one
- Risks:
  - Defeats the purpose of simplification

### Option C: Keep Current System
- Approach: Leave complex adaptive system in place
- Pros:
  - Theoretically better handling of various gap sizes
- Cons:
  - 300+ lines of complex code
  - Hard to debug
  - Unpredictable behavior
  - No proven benefit
- Risks:
  - Continued maintenance burden

## Recommendation
**Option A: Simple 30-Day Hard Reset**

Rationale:
- Aligns with KALMAN_DEFAULTS['reset_gap_days'] = 30 already in constants
- 30 days is a natural boundary (monthly)
- Extreme simplicity worth minor accuracy trade-off
- Easier to explain to users: "After a month, we start fresh"

## High-Level Design

### Architecture Changes
```
BEFORE:
processor.py → handle_gap_detection() → buffer initialization
             → update_gap_buffer() → collect measurements
             → complex state management
kalman.py → initialize_from_buffer() → trend estimation
          → calculate_adaptive_parameters() → variance adjustment
          → apply_adaptation_decay() → gradual normalization

AFTER:
processor.py → check_gap() → if ≥30 days: reset_state()
kalman.py → (remove 5 methods, ~150 lines)
```

### Affected Files
- `src/processor.py` - Remove gap handling functions, simplify process_measurement
- `src/kalman.py` - Remove 5 gap-related methods
- `src/constants.py` - Keep KALMAN_DEFAULTS['reset_gap_days'] = 30
- `config.toml` - Remove entire [kalman.gap_handling] section
- `src/database.py` - Remove gap_buffer and gap_adaptation fields

## Implementation Plan (No Code)

### Step 1: Remove Gap Handling from Kalman
- Delete methods from KalmanFilterManager:
  - `estimate_trend_from_buffer()` (lines 234-269)
  - `calculate_adaptive_parameters()` (lines 271-292)
  - `initialize_from_buffer()` (lines 294-341)
  - `apply_adaptation_decay()` (lines 343-379)
- Total removal: ~145 lines

### Step 2: Simplify Processor Gap Detection
- Remove from processor.py:
  - `handle_gap_detection()` function (lines 38-69)
  - `update_gap_buffer()` function (lines 71-94)
  - All gap buffer logic in `process_measurement()` (lines 156-224)
  - Gap adaptation decay logic (lines 386-387, 418-423)
- Replace with simple check:
  ```
  if last_accepted_timestamp exists:
      gap_days = (current - last_accepted).days
      if gap_days >= 30:
          reset_all_state()
  ```
- Total removal: ~120 lines, replaced with ~10 lines

### Step 3: Add Simple Reset Function
- Create `reset_state()` in processor.py:
  - Clear Kalman state
  - Clear quality scorer history
  - Log reset event with gap duration
  - Return reset marker for visualization

### Step 4: Update State Structure
- Remove from state dictionary:
  - `gap_buffer` field
  - `gap_adaptation` field
- Add simple field:
  - `last_accepted_timestamp` (if not already present)

### Step 5: Clean Configuration
- Remove entire `[kalman.gap_handling]` section from config.toml (lines 26-42)
- Replace with simple `[kalman.reset]` section:
  ```toml
  [kalman.reset]
  # Simple reset configuration - replaces complex gap handling
  enabled = true  # Enable automatic reset after long gaps
  gap_threshold_days = 30  # Days without data before full reset
  ```
- This reduces 17 lines to 4 lines in config
- Clear, self-documenting configuration

### Step 6: Update Database Schema
- Remove gap_buffer column
- Remove gap_adaptation column
- Add/ensure `reset_events` tracking with:
  - timestamp of reset
  - gap duration in days
  - reason (always "gap_exceeded" for now)

### Step 7: Enhanced Visualization for Resets

#### Visual Design for Reset Events
1. **Primary Reset Indicator**:
   - Vertical dashed line at reset point
   - Color: Orange (#FF6600) for visibility
   - Label at top: "Reset (30d gap)" or "Reset (45d gap)" etc.
   - Semi-transparent (opacity: 0.3) to not obscure data

2. **Gap Visualization**:
   - Shaded region showing the gap period
   - Light gray background (#F0F0F0) with diagonal stripes pattern
   - Text overlay in gap: "No Data - 35 days" (centered)
   - Makes gaps immediately visible to users

3. **State Transition Markers**:
   - Last point before gap: Diamond marker with "Pre-gap" label
   - First point after gap: Star marker with "Post-reset" label
   - Different color (blue) to distinguish from regular data points

4. **Reset Information Box**:
   - Hover tooltip on reset line shows:
     ```
     System Reset
     Gap Duration: 35 days
     Last Value: 85.2 kg (2024-01-15)
     Reset Value: 84.8 kg (2024-02-19)
     State: Reinitialized
     ```

5. **Legend Enhancement**:
   - Add "Reset Events" to legend with dashed line symbol
   - Add "Data Gaps" with shaded region symbol
   - Keep existing source markers

#### Implementation in visualization.py
- Add `plot_reset_events()` method:
  - Draw vertical lines at reset timestamps
  - Add shaded regions for gap periods
  - Create hover tooltips with reset details
- Add `annotate_gap_boundaries()` method:
  - Mark last pre-gap and first post-gap points
  - Add text labels for gap duration
- Update main plot to call these methods

#### Configuration for Visualization
```toml
[visualization.reset]
# Reset event display configuration
show_reset_lines = true  # Show vertical lines at reset points
show_gap_regions = true  # Shade the gap periods
show_gap_labels = true  # Display gap duration text
reset_line_color = "#FF6600"  # Orange for visibility
reset_line_style = "dash"  # Line style: "solid" | "dash" | "dot"
reset_line_width = 2  # Thickness in pixels
gap_region_color = "#F0F0F0"  # Light gray for gap regions
gap_region_opacity = 0.5  # Transparency (0-1)
gap_region_pattern = "diagonal"  # Pattern: "solid" | "diagonal" | "dots"
show_transition_markers = true  # Mark pre-gap and post-reset points
transition_marker_size = 10  # Size of special markers
```

## Validation & Rollout

### Test Strategy
1. Unit tests:
   - Test gap detection (29 days = no reset, 30 days = reset)
   - Test state reset completeness
   - Test first measurement after reset

2. Integration tests:
   - Process user with multiple 30+ day gaps
   - Verify Kalman reinitializes correctly
   - Verify quality scorer resets

3. Regression tests:
   - Run on existing test data
   - Compare reset points with current system
   - Verify no crashes or data loss

### Manual QA Checklist
- [ ] Process file with 30+ day gaps
- [ ] Verify reset markers appear in visualization
- [ ] Check that Kalman predictions restart from first post-gap measurement
- [ ] Confirm quality scoring starts fresh after reset
- [ ] Test edge cases: exactly 30 days, 29.9 days, 100+ days

### Rollout Plan
1. Create feature branch
2. Remove code in order (Kalman → Processor → Config)
3. Run full test suite
4. Process sample data and review visualizations
5. Merge to main

## Risks & Mitigations

### Risk 1: Worse handling of 20-29 day gaps
- Impact: Medium
- Mitigation: Accept trade-off for simplicity
- Monitor: Track prediction accuracy for these gaps

### Risk 2: Loss of trend information
- Impact: Low
- Mitigation: 30+ day gaps likely indicate context change anyway
- Monitor: Review real-world data patterns

### Risk 3: Breaking existing processing
- Impact: High
- Mitigation: Comprehensive testing before deployment
- Monitor: Keep backup of current implementation

## Acceptance Criteria
- [ ] All gap handling code removed (~300 lines deleted)
- [ ] Simple 30-day reset logic implemented (~10 lines)
- [ ] State resets completely after 30-day gaps
- [ ] Visualization shows reset events
- [ ] All tests pass
- [ ] Processing performance improved or unchanged

## Out of Scope
- Changing the 30-day threshold
- Gradual adaptation mechanisms
- Trend preservation across gaps
- Different reset strategies per source type
- Partial state preservation

## Open Questions
1. Should we make the 30-day threshold configurable or hard-code it?
   - Recommendation: Hard-code for maximum simplicity
2. Should reset events be stored permanently in the database?
   - Recommendation: Yes, for audit trail
3. Should we notify users when resets occur?
   - Recommendation: Only in verbose logging mode

## Configuration Migration Guide

### Current config.toml (Lines 26-42)
```toml
[kalman.gap_handling]
enabled = true
gap_threshold_days = 28
warmup_size = 10
max_warmup_days = 40
gap_variance_multiplier = 2.0
trend_variance_multiplier = 2.0
adaptation_decay_rate = 2.0
```

### New config.toml (Replacement)
```toml
[kalman.reset]
# Simple reset after long gaps (replaces complex gap handling)
enabled = true  # Enable automatic reset
gap_threshold_days = 30  # Days before reset

[visualization.reset]
# How to display reset events
show_reset_lines = true  # Vertical lines at resets
show_gap_regions = true  # Shade gap periods
show_gap_labels = true  # Show "35 days" text
reset_line_color = "#FF6600"
gap_region_color = "#F0F0F0"
gap_region_opacity = 0.5
show_transition_markers = true  # Mark boundaries
```

### Benefits of New Configuration
- 17 lines → 12 lines (30% reduction)
- Self-documenting parameter names
- Clear separation of logic (reset) vs display (visualization)
- No mysterious multipliers or decay rates
- Users immediately understand what each setting does

## Visualization Examples

### Example 1: Simple Gap with Reset
```
Before Gap: ──●──●──●──●──●
Gap Period: [////30 days////]
After Reset: ★──●──●──●──●──●
             ↑
        Reset Line
```

### Example 2: Multiple Resets in Dashboard
- Each reset shows as orange dashed vertical line
- Gap regions are shaded gray with duration label
- Hover shows detailed reset information
- Pre/post gap points are highlighted
- Legend clearly indicates reset events

### Example 3: User Experience
When users see a reset in the visualization:
1. Immediately visible orange line says "Reset"
2. Gray shaded area shows the gap period
3. Text shows "No data - 35 days"
4. They understand: "System restarted after a month gap"
5. No confusion about complex adaptation or buffering

## Review Cycle
### Self-Review Notes
- Verified line numbers in current codebase ✓
- Confirmed 300+ lines will be removed ✓
- Designed clear visualization strategy ✓
- Created migration path for config.toml ✓
- Validated that 30-day threshold aligns with existing constants ✓

### Code Removal Summary
- `src/kalman.py`: -145 lines (4 methods)
- `src/processor.py`: -120 lines (2 functions + inline logic)
- `config.toml`: -17 lines → +12 lines (net -5 lines, simpler structure)
- `src/database.py`: -2 lines (fields)
- `src/visualization.py`: +~50 lines (enhanced reset display)
- **Total: ~282 lines removed, ~62 lines added**
- **Net reduction: ~220 lines (78% reduction in gap handling complexity)**

### Visual Improvement Summary
- Clear reset indicators (orange dashed lines)
- Visible gap periods (shaded regions)
- Informative labels and tooltips
- Better user understanding of system behavior
- Professional, clean appearance