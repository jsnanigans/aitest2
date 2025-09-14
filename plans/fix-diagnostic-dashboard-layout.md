# Plan: Fix Diagnostic Dashboard Layout Issues

## Summary
Fix the overlapping and misaligned elements in the diagnostic dashboard visualization to improve readability and user experience.

## Context
- Source: User screenshot showing layout issues in diagnostic dashboard
- The dashboard at `file:///Users/brendanmullins/Projects/aitest/strem_process_anchor/output/viz_test_no_date/03a0814f-496e-484e-9472-f44f28a5c049_diagnostic.html` shows overlapping charts and misaligned elements
- Current implementation uses Plotly subplots with a 5x2 grid layout

## Assumptions
- The issue is primarily with subplot spacing and sizing
- The dashboard needs to maintain all current visualizations
- The solution should work across different screen sizes
- The fix should apply to all diagnostic dashboards generated

## Requirements

### Functional
- Fix overlapping chart titles and content
- Ensure proper spacing between subplots
- Maintain readability of all chart elements
- Preserve all existing visualizations and data

### Non-functional
- Dashboard should render correctly on standard screen sizes (1920x1080 minimum)
- Performance should not be degraded
- Solution should be maintainable and clear

## Alternatives

### Option A: Adjust Subplot Spacing and Heights
- Approach: Modify vertical_spacing, horizontal_spacing, and row heights in make_subplots
- Pros:
  - Simple implementation
  - Maintains current structure
  - Quick to test and iterate
- Cons:
  - May not solve all layout issues
  - Fixed spacing might not work for all data sizes
- Risks: Minimal

### Option B: Redesign Layout with Fewer Rows
- Approach: Reduce from 5x2 to 4x2 grid, combine or remove less critical charts
- Pros:
  - More space per chart
  - Better visual hierarchy
  - Cleaner appearance
- Cons:
  - Loss of some information
  - Requires deciding what to remove/combine
- Risks: Users may miss removed information

### Option C: Dynamic Layout with Responsive Sizing
- Approach: Calculate subplot sizes based on content and screen size
- Pros:
  - Adapts to different scenarios
  - Optimal use of space
- Cons:
  - More complex implementation
  - Harder to maintain
  - May have performance impact
- Risks: Complexity could introduce bugs

## Recommendation
**Option A** - Adjust Subplot Spacing and Heights

Rationale:
- Preserves all existing functionality
- Quick to implement and test
- Can be enhanced incrementally
- Low risk of breaking existing features

## High-Level Design

### Architecture Changes
- Modify `DiagnosticDashboard._create_diagnostic_figure()` method in `src/visualization.py`
- Adjust subplot configuration parameters
- Update individual plot methods to handle spacing better

### Key Parameters to Adjust
1. Increase `vertical_spacing` from 0.08 to 0.12-0.15
2. Increase `horizontal_spacing` from 0.12 to 0.15-0.18  
3. Increase overall figure `height` from 1400 to 1600-1800
4. Add row height ratios to give more space to complex charts
5. Adjust font sizes for better readability

### Affected Components
- `src/visualization.py`:
  - `DiagnosticDashboard._create_diagnostic_figure()`
  - Individual plot methods may need margin adjustments

## Implementation Plan (No Code)

### Step 1: Analyze Current Layout Issues
- Identify specific overlapping elements
- Measure current spacing values
- Document which charts need more space

### Step 2: Adjust Main Figure Configuration
- Update `make_subplots()` parameters:
  - Set `vertical_spacing=0.15`
  - Set `horizontal_spacing=0.18`
  - Add `row_heights` parameter to allocate space proportionally
  - Increase figure height to 1600-1800 pixels

### Step 3: Optimize Individual Charts
- Adjust title positioning for each subplot
- Reduce marker sizes where appropriate
- Optimize legend positioning
- Ensure axis labels don't overlap

### Step 4: Handle Edge Cases
- Test with minimal data (empty charts)
- Test with maximum data (crowded charts)
- Verify table rendering in bottom-right position

### Step 5: Fine-tune Typography
- Reduce subplot title font sizes if needed
- Adjust axis label sizes
- Ensure consistent font sizing across charts

## Validation & Rollout

### Test Strategy
1. Generate diagnostic dashboards for multiple users
2. Verify no overlapping elements
3. Check readability at different zoom levels
4. Test with various data densities

### Manual QA Checklist
- [ ] All subplot titles are fully visible
- [ ] No chart content overlaps with titles
- [ ] Legends don't obscure data
- [ ] Table in bottom-right is properly formatted
- [ ] Overall dashboard is readable without horizontal scrolling
- [ ] Charts maintain proper aspect ratios

### Rollout Plan
1. Apply changes to `src/visualization.py`
2. Test with existing test data
3. Generate new dashboards for verification
4. No migration needed - changes apply to new generations only

## Risks & Mitigations

### Risk 1: Increased Height May Require Scrolling
- Mitigation: Keep height at reasonable maximum (1800px)
- Monitor: Check on standard screen resolutions

### Risk 2: Spacing Changes May Affect Other Dashboard Types
- Mitigation: Changes isolated to DiagnosticDashboard class
- Monitor: Verify InteractiveDashboard still works correctly

## Acceptance Criteria
- [ ] No overlapping elements in diagnostic dashboard
- [ ] All text is readable without zooming
- [ ] Charts maintain visual hierarchy
- [ ] Dashboard renders correctly for all users in test data
- [ ] Performance is not degraded

## Out of Scope
- Redesigning the overall dashboard structure
- Adding new visualizations
- Changing the data processing logic
- Mobile responsiveness
- Print layout optimization

## Open Questions
1. Should we add a configuration option for dashboard height?
2. Do we need different layouts for different data densities?
3. Should subplot titles be shortened to save space?

## Review Cycle
- Initial implementation tested with sample data
- Adjustments made based on visual inspection
- Final parameters determined through iterative testing