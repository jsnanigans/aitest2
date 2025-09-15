# Plan: Enhanced Visualization with Reset Markers and Source Icons

## Summary
Enhance the weight timeline visualization to display:
1. Reset events after data gaps with visual markers and annotations
2. Source type indicators using distinct marker icons for each data source
3. Improved hover information showing reset context and source details

## Context
- **Source**: User request for better visibility of data resets and source types
- **Current State**: Visualization shows accepted/rejected points but lacks reset event markers and source type differentiation
- **Assumptions**: 
  - Reset events are already tracked in processor.py (was_reset flag)
  - Source types are defined in constants.py with profiles
  - Users need to understand when/why data resets occur
  - Different source types have different reliability levels that should be visually apparent

## Requirements

### Functional
1. Display reset event markers on the timeline when Kalman filter resets occur
2. Show different marker icons/shapes for each source type for both accepted and rejected points
3. Use color to indicate acceptance status (green gradient for accepted, red for rejected) while maintaining shape differentiation
4. Include reset reason and gap duration in hover information
5. Visually distinguish between different reset triggers (gap-based, questionnaire-based)
6. Maintain existing quality score color gradient for accepted points alongside shape markers
7. Show rejection reason categories with both shape (source) and color intensity (severity)

### Non-functional
- Performance: No significant impact on rendering time
- Accessibility: Markers should be distinguishable by shape, not just color
- Compatibility: Work with both basic and enhanced visualization modes

## Alternatives

### Option A: Overlay Reset Markers as Separate Trace
- **Approach**: Add a new Plotly trace specifically for reset events
- **Pros**: 
  - Clean separation of concerns
  - Easy to toggle on/off
  - Can use custom styling (vertical lines, annotations)
- **Cons**: 
  - May clutter the chart with too many traces
  - Requires coordination between multiple data series
- **Risks**: Visual overlap with existing data points

### Option B: Embed Reset Info in Existing Data Points
- **Approach**: Modify existing accepted/rejected traces to include reset markers
- **Pros**: 
  - Simpler implementation
  - Less visual clutter
  - Direct association with affected measurements
- **Cons**: 
  - Limited styling options
  - Harder to make reset events stand out
- **Risks**: Reset events might not be noticeable enough

### Option C: Hybrid Approach with Annotations and Modified Markers
- **Approach**: Use modified markers for source types and Plotly annotations for reset events
- **Pros**: 
  - Best of both worlds
  - Clear visual hierarchy
  - Flexible styling options
- **Cons**: 
  - More complex implementation
  - Need to manage both markers and annotations
- **Risks**: Potential performance impact with many annotations

## Recommendation
**Option C: Hybrid Approach** - This provides the best user experience by clearly distinguishing both source types (via markers) and reset events (via annotations) without overwhelming the visualization.

## High-Level Design

### Architecture Flow
```
Results Data → Extract Reset Events → Generate Annotations
     ↓                                        ↓
Extract Source Types → Map to Marker Symbols → Create Traces
     ↓                                        ↓
Determine Colors (Quality/Rejection) → Apply to Markers
     ↓                                        ↓
Combine All Elements → Render Interactive Timeline
```

### Visual Encoding Strategy
- **Shape**: Indicates data source type (triangle, circle, square, diamond, hexagon)
- **Color**: Indicates acceptance status and quality
  - Accepted: Green gradient (darker = higher quality score)
  - Rejected: Red shades (darker = more severe rejection)
- **Annotations**: Reset events with vertical lines/markers
- **Size**: Can indicate confidence or importance (optional)

### Data Model Changes
- Ensure `was_reset`, `reset_reason`, and `gap_days` are propagated from processor
- Add source type to marker symbol mapping
- Add rejection severity calculation for color intensity
- Create reset event collection for annotations

### Affected Modules
- `src/visualization.py`: Main changes for basic visualization
- `src/viz_enhanced.py`: Enhanced visualization updates
- `src/processor.py`: Ensure reset metadata is properly returned
- `src/constants.py`: Define marker symbols and rejection severity mapping

## Implementation Plan (No Code)

### Step 1: Define Source Type Marker Mapping and Color Schemes
- Location: `src/constants.py`
- Add SOURCE_MARKER_SYMBOLS dictionary mapping source types to Plotly symbols
- Symbol mapping:
  - `care-team-upload`: triangle-up (professional/authoritative)
  - `patient-upload`: circle (standard/default)
  - `internal-questionnaire`: square (form/structured)
  - `initial-questionnaire`: square (form/structured)
  - `patient-device`: diamond (automated/digital)
  - `https://connectivehealth.io`: hexagon (external API)
  - `https://api.iglucose.com`: hexagon (external API)
  - Default/unknown: circle
- Add REJECTION_SEVERITY_COLORS for rejected points:
  - Critical: '#8B0000' (dark red) - for impossible values
  - High: '#CC0000' (medium-dark red) - for extreme deviations
  - Medium: '#FF4444' (medium red) - for suspicious values
  - Low: '#FF9999' (light red) - for minor issues
- Color scheme:
  - Accepted: Green gradient based on quality score (existing)
  - Rejected: Red shades based on rejection severity (using get_rejection_severity())
- Ensure symbols are accessibility-friendly (distinguishable shapes)

### Step 2: Track Reset Events in Processing
- Location: `src/processor.py`
- Verify reset metadata is included in results:
  - `was_reset`: boolean flag
  - `reset_reason`: string description
  - `gap_days`: numeric gap duration
  - `reset_timestamp`: when reset occurred
- Ensure metadata persists through entire pipeline

### Step 3: Update Basic Visualization
- Location: `src/visualization.py`
- Modify `create_weight_timeline()`:
  - Extract reset events from results
  - Map source types to marker symbols for BOTH accepted and rejected points
  - Update accepted trace:
    - Use source-based marker symbols
    - Maintain quality score color gradient (green scale)
  - Update rejected trace:
    - Use source-based marker symbols (same as accepted)
    - Use red color with intensity based on rejection severity
    - Consider: light red (minor), medium red (moderate), dark red (severe)
  - Add reset annotations using `fig.add_annotation()`
- Update hover text generation to include reset context and source info

### Step 4: Update Enhanced Visualization
- Location: `src/viz_enhanced.py`
- Apply same changes to `create_enhanced_weight_timeline()`
- For rejected points in enhanced view:
  - Use source-based marker shapes
  - Apply rejection severity colors
  - Consider adding rejection category to subplot
- Consider adding reset events to quality subplot
- Ensure consistency with basic visualization
- In the quality components subplot, could show rejection categories distribution

### Step 5: Create Visual Legend
- Add legend entries for:
  - Different source type markers (shapes)
  - Acceptance status (color: green = accepted, red = rejected)
  - Quality score gradient (for accepted points)
  - Rejection severity levels (red intensity)
  - Reset event indicators
- Legend structure:
  - Source Types section (showing shapes)
  - Status section (showing color meanings)
  - Reset Events section (showing annotation style)
- Position legend to avoid overlap with data
- Include brief descriptions in legend

### Step 6: Add Configuration Options
- Location: `config.toml` structure
- Add visualization options:
  - `show_reset_markers`: boolean
  - `show_source_icons`: boolean
  - `reset_marker_style`: dict with color, size, etc.
- Allow users to customize appearance

## Validation & Rollout

### Test Strategy
1. **Unit Tests**: 
   - Test marker symbol mapping
   - Verify reset event extraction
   - Test annotation generation

2. **Integration Tests**:
   - Process sample data with known gaps
   - Verify reset markers appear correctly
   - Test with multiple source types

3. **Visual Tests**:
   - Generate test visualizations with various scenarios
   - Verify marker shapes are distinguishable
   - Check hover information completeness

### Manual QA Checklist
- [ ] Reset markers appear at correct timestamps
- [ ] Source type markers are visually distinct for both accepted and rejected points
- [ ] Rejected points show correct source shape with red color
- [ ] Accepted points show correct source shape with green quality gradient
- [ ] Rejection severity is reflected in red color intensity
- [ ] Hover text includes reset information and source details
- [ ] Legend accurately describes all markers, shapes, and colors
- [ ] Performance is acceptable with many data points
- [ ] Visualization works in both basic and enhanced modes
- [ ] Color-blind users can still distinguish points by shape

### Rollout Plan
1. Implement changes in development branch
2. Generate test visualizations with existing test data
3. Review with stakeholders for feedback
4. Adjust styling based on feedback
5. Deploy to production

## Risks & Mitigations

### Risk 1: Visual Clutter
- **Risk**: Too many markers/annotations make chart unreadable
- **Mitigation**: Add configuration to toggle features on/off, use subtle styling

### Risk 2: Performance Impact
- **Risk**: Many annotations slow down rendering
- **Mitigation**: Limit number of visible annotations, use efficient rendering

### Risk 3: Browser Compatibility
- **Risk**: Complex Plotly features may not work in all browsers
- **Mitigation**: Test in major browsers, provide fallback options

## Acceptance Criteria
1. ✓ Reset events are clearly visible on timeline with timestamp and reason
2. ✓ Each source type has a unique, distinguishable marker shape for both accepted and rejected points
3. ✓ Rejected points use same source-based shapes but with red color indicating rejection
4. ✓ Accepted points use source-based shapes with green quality score gradient
5. ✓ Rejection severity is visually indicated through red color intensity
6. ✓ Hover information includes reset context, source type, and rejection details when applicable
7. ✓ Legend clearly explains shapes (sources), colors (status), and annotations (resets)
8. ✓ Features can be toggled via configuration
9. ✓ No significant performance degradation
10. ✓ Works in both basic and enhanced visualization modes
11. ✓ Accessibility maintained through shape differentiation, not just color

## Out of Scope
- Real-time visualization updates
- Animation of reset events
- Custom user-defined marker shapes
- Export of reset event data to separate file
- Historical comparison of reset patterns

## Open Questions
1. Should reset markers be shown for all resets or only gap-based resets?
2. What color scheme should be used for reset annotations?
3. Should we show the previous state value before reset?
4. Do we need to indicate confidence level changes after reset?
5. Should source reliability scores be shown in the legend?
6. Should rejected point sizes vary based on how far off they were from expected values?
7. Should we use different red shades or patterns (striped, dotted outline) for different rejection reasons?
8. Do we need a separate trace for each source type or can we handle it with a single trace using arrays?

## Review Cycle
### Self-Review Notes
- Considered user feedback about difficulty identifying data gaps
- Ensured accessibility with shape-based differentiation
- Balanced information density with readability
- Provided configuration options for customization

### Revisions
- Added configuration options after considering different user preferences
- Included accessibility considerations for color-blind users
- Added performance mitigation strategies