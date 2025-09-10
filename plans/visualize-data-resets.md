# Plan: Visualize Data Resets Due to Gaps

## Summary
Add visual indicators to the dashboard that clearly show when the Kalman filter state was reset due to gaps in data (>30 days). This will help users understand why certain measurements might appear disconnected from the filtered trend line and provide transparency about the processor's behavior.

## Context
- Source: User request to show when data is reset because of gaps in visualization
- Current behavior: Processor resets state after gaps > 30 days (configurable via `reset_gap_days`)
- Current visualization: No indication of when resets occur
- Assumptions:
  - Users need to understand why filtered weights might suddenly jump after gaps
  - Visual clarity is important for debugging and validation
  - Reset information should be non-intrusive but clear

## Requirements

### Functional
- Detect when a state reset occurred between measurements
- Visually indicate reset points on all relevant plots
- Show gap duration that triggered the reset
- Maintain visual clarity without cluttering the plots
- Support both full-range and cropped views

### Non-functional
- Performance: Minimal impact on rendering time
- Clarity: Reset indicators should be immediately understandable
- Consistency: Same visual language across all plots
- Accessibility: Consider colorblind users

## Alternatives

### Option A: Add Reset Flag to Result Dictionary
- **Approach**: Modify processor to include `was_reset: true` flag in results when reset occurs
- **Pros**: 
  - Clean data flow from processor to visualization
  - Easy to detect in visualization code
  - No need to infer resets from gaps
- **Cons**: 
  - Requires processor modification
  - Need to ensure backward compatibility

### Option B: Detect Resets in Visualization by Gap Analysis
- **Approach**: Calculate gaps between timestamps in visualization and infer resets
- **Pros**: 
  - No processor changes needed
  - Works with existing data
- **Cons**: 
  - Duplicates logic from processor
  - Might miss edge cases
  - Less accurate than explicit flag

### Option C: Track Reset Events Separately
- **Approach**: Store reset events in a separate list in processor state
- **Pros**: 
  - Complete reset history available
  - Could show multiple reset reasons in future
- **Cons**: 
  - More complex state management
  - Requires database schema changes

## Recommendation
**Option A** - Add reset flag to result dictionary. This is the cleanest approach that maintains single source of truth for reset logic in the processor while providing clear signals to the visualization layer.

## High-Level Design

### Data Flow
1. Processor detects gap > `reset_gap_days`
2. Processor adds `was_reset: true` and `gap_days: N` to result
3. Visualization reads reset flag from results
4. Visualization renders reset indicators

### Visual Indicators
- **Vertical dashed line** at reset point (light gray, alpha=0.5)
- **Text annotation** showing gap duration (e.g., "31 day gap")
- **Break in filtered line** to show discontinuity
- **Legend entry** explaining reset markers
- **Optional**: Shaded region showing gap period

### Affected Components
- `src/processor.py`: Add reset flag to result dictionary
- `src/visualization.py`: Add reset visualization logic
- All plot types that show time series data

## Implementation Plan (No Code)

### Step 1: Modify Processor Result Structure
- Location: `src/processor.py`, method `_create_result()`
- Add `was_reset` boolean field (default: False)
- Add `gap_days` float field when reset occurs
- Set these fields in the reset branch (lines 118-132)

### Step 2: Update Visualization Data Processing
- Location: `src/visualization.py`, function `create_dashboard()`
- Extract reset information from results
- Build list of reset points with timestamps and gap durations
- Handle both accepted and rejected results with resets

### Step 3: Add Reset Indicators to Main Plot
- Location: `src/visualization.py`, subplot ax1 (lines 111-150)
- Add vertical lines at reset points
- Add text annotations with gap duration
- Consider line breaks in filtered weight plot

### Step 4: Add Reset Indicators to Cropped Views
- Location: `src/visualization.py`, cropped subplots
- Filter reset points to cropped date range
- Apply same visual indicators as main plot
- Ensure consistency across all time-based plots

### Step 5: Update Innovation and Confidence Plots
- Location: `src/visualization.py`, innovation/confidence subplots
- Add vertical lines at reset points
- Help explain why innovation might spike after reset

### Step 6: Add Legend and Documentation
- Add reset indicator to plot legends
- Include explanation in plot titles or subtitles
- Consider adding a text box explaining reset behavior

### Step 7: Handle Edge Cases
- First measurement (no gap to calculate)
- Multiple resets in sequence
- Resets at plot boundaries
- Resets with rejected measurements

## Validation & Rollout

### Test Strategy
1. **Unit tests**: Verify reset flag is set correctly in processor
2. **Integration tests**: Ensure visualization correctly reads and displays reset flags
3. **Visual tests**: Generate plots with known reset points and verify appearance
4. **Edge case tests**: 
   - User with no resets
   - User with multiple resets
   - Reset at first/last measurement
   - Reset with rejected measurement

### Manual QA Checklist
- [ ] Reset indicators appear at correct timestamps
- [ ] Gap duration is accurately displayed
- [ ] Indicators are visible but not overwhelming
- [ ] Legends correctly explain indicators
- [ ] Cropped views show appropriate subset of resets
- [ ] Performance is not degraded

### Rollout Plan
1. Implement processor changes with feature flag
2. Test with known problematic users (e.g., user ending in 106kg)
3. Implement visualization changes
4. Test with full dataset
5. Document new feature in README

## Risks & Mitigations

### Risk 1: Visual Clutter
- **Risk**: Too many reset indicators make plots hard to read
- **Mitigation**: Use subtle styling, consider aggregating nearby resets

### Risk 2: Backward Compatibility
- **Risk**: Existing processed data won't have reset flags
- **Mitigation**: Gracefully handle missing fields, could infer from gaps as fallback

### Risk 3: Performance Impact
- **Risk**: Additional processing slows down visualization
- **Mitigation**: Pre-calculate reset points once, reuse across plots

## Acceptance Criteria
- [ ] Reset indicators appear when gap > reset_gap_days
- [ ] Gap duration is shown in human-readable format
- [ ] Indicators are visible in both full and cropped views
- [ ] Legend explains what reset indicators mean
- [ ] No performance degradation > 10%
- [ ] Works with existing test data

## Out of Scope
- Configurable reset indicator styles
- Interactive tooltips on reset points
- Reset statistics in summary panels
- Different indicators for different reset reasons
- Undo/redo of resets

## Open Questions
1. Should we show the exact timestamp of the last measurement before the gap?
2. Should reset indicators be toggleable in the UI?
3. Should we use different colors/styles for different gap durations?
4. Should we show predicted vs actual weight at reset point?
5. Do we need to indicate resets in the CSV output as well?

## Review Cycle

### Self-Review Notes
- Considered multiple visual approaches to avoid clutter
- Ensured solution works with existing architecture
- Kept changes minimal and focused
- Maintained backward compatibility considerations

### Stakeholder Feedback Points
- Is the visual style of reset indicators appropriate?
- Should gap duration be shown in days or weeks for very long gaps?
- Do we need more detailed reset information in tooltips?
- Should resets be highlighted more or less prominently?

### Council Review

**Butler Lampson** (Simplicity): "Adding a simple flag to the result is the right approach. Don't overcomplicate with separate event tracking."

**Don Norman** (User Experience): "Users need to understand why the line 'jumps' - make sure the gap annotation is clear and positioned where users will look. Consider using visual metaphors like a 'break' in the line."

**Ward Cunningham** (Documentation): "This is self-documenting design at its best - the visualization explains the processor's behavior. Make sure the legend clearly explains what a reset means."

**Kent Beck** (Testing): "Test with real user data that has known gaps. The 106kg user case you mentioned is perfect for validation."