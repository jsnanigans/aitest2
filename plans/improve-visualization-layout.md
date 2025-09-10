# Plan: Improve Visualization Layout for Weight Stream Processor Dashboard

## Summary
Redesign the Kalman filter evaluation dashboard to improve readability, reduce visual clutter, and create a clear visual hierarchy that helps users quickly understand their weight data processing results.

## Context
- Source: User feedback on current visualization showing poor layout with small, hard-to-read charts
- Current state: 4x4 grid layout (12 active subplots) in 20x14 inch figure
- Main issues: Charts too small, overlapping annotations, poor use of space, redundant views
- Assumptions:
  - Users primarily care about filtered results and data quality
  - Statistical distributions are secondary information
  - Need both overview and detailed time views
  - Annotations for rejections are important but currently create clutter

## Requirements

### Functional
- Show Kalman filtered weight vs raw accepted/rejected data
- Display both full range and recent (cropped) time periods
- Show rejection reasons without visual clutter
- Display key statistics and performance metrics
- Maintain all current data insights

### Non-functional
- Charts must be large enough to read details clearly
- Visual hierarchy should guide user attention
- Layout should work well at different screen sizes
- Reduce cognitive load through better organization
- Maintain high-quality output (configurable DPI)

## Alternatives

### Option A: Two-Page Report Layout
- Approach: Split into 2 separate visualizations - main dashboard and detailed analytics
- Page 1: Large primary charts (filtered data, recent trends)
- Page 2: Statistical distributions and detailed metrics
- Pros: 
  - Maximum space for each chart
  - Clear separation of concerns
  - Can optimize each page for its purpose
- Cons:
  - Requires generating/viewing multiple files
  - Can't see everything at once
  - May miss correlations between views
- Risks: Users might not look at second page

### Option B: Hierarchical Single-Page Layout
- Approach: Redesign as 3-row layout with different sized subplots
- Row 1: Large primary chart (60% height) showing filtered vs raw data
- Row 2: Medium charts (25% height) for key metrics (innovation, confidence, trend)
- Row 3: Small charts (15% height) for distributions and statistics
- Pros:
  - Everything visible at once
  - Clear visual hierarchy
  - Better use of space
- Cons:
  - Still some compromise on chart sizes
  - Complex subplot arrangement
- Risks: Implementation complexity with matplotlib

### Option C: Interactive Dashboard (Plotly/Dash)
- Approach: Convert to interactive visualization with zoom, pan, hover details
- Single large chart with toggleable layers
- Hover for rejection details instead of annotations
- Collapsible panels for statistics
- Pros:
  - Best user experience
  - No clutter from annotations
  - User controls what they see
- Cons:
  - Requires new dependencies (plotly)
  - Different output format (HTML vs PNG)
  - More complex implementation
- Risks: May not fit current workflow

## Recommendation
**Option B: Hierarchical Single-Page Layout** with smart annotation management

Rationale:
- Maintains single-file output (current workflow)
- Significantly improves readability without new dependencies
- Provides clear visual hierarchy
- Can be implemented with current matplotlib setup

Fallback: If layout still feels cramped, implement Option A as an additional output option.

## High-Level Design

### Layout Structure
```
Figure: 24" x 16" (increased width, slightly increased height)

Row 1 (50% of height): Primary Data View
├── Left (70% width): Filtered vs Raw Data (Recent Period)
└── Right (30% width): Statistics Panel (formatted text/table)

Row 2 (25% of height): Key Metrics
├── Innovation (25% width)
├── Normalized Innovation (25% width)
├── Confidence (25% width)
└── Trend (25% width)

Row 3 (25% of height): Distributions & Analysis
├── Residuals (25% width)
├── Innovation Distribution (25% width)
├── Daily Changes (25% width)
└── Rejection Categories (25% width - bar chart)
```

### Key Improvements
1. **Primary chart 6x larger** than current subplots
2. **Smart annotation clustering**: Group nearby rejections, show only most important
3. **Tabbed time ranges**: Buttons/legend to switch between full/recent view
4. **Statistics sidebar**: Properly formatted, not in tiny subplot
5. **Color-coded timeline**: Background shading for different periods
6. **Rejection summary bar**: Replace text list with horizontal bar chart

### Annotation Strategy
- Maximum 5 rejection annotations visible at once
- Cluster rejections within 24-hour windows
- Show count when multiple rejections clustered
- Use semi-transparent callout boxes
- Position annotations to avoid overlap algorithmically

## Implementation Plan (No Code)

### Step 1: Refactor Layout Structure
- Modify figure size to 24x16 inches
- Replace 4x4 grid with custom GridSpec layout
- Define 3 rows with heights [4, 2, 2]
- Create proper subplot areas for each component

### Step 2: Enhance Primary Chart
- Increase line widths for better visibility
- Add subtle grid with different major/minor lines
- Implement dual-axis option for showing both kg and lbs
- Add shaded regions for night hours (optional)
- Improve date formatting with intelligent tick spacing

### Step 3: Improve Annotation System
- Enhance clustering algorithm to group within time windows
- Implement collision detection for annotation placement
- Add annotation priority scoring (prefer unique rejection types)
- Create expandable annotation style (brief label → detailed on hover)
- Add legend entry for rejection count summary

### Step 4: Redesign Statistics Panel
- Convert from matplotlib text to formatted table
- Use alternating row colors for readability
- Group statistics into sections with headers
- Add visual indicators (↑↓) for trends
- Include mini sparklines for key metrics

### Step 5: Optimize Secondary Charts
- Increase font sizes for axes and labels
- Use consistent color scheme across all charts
- Add reference lines with subtle styling
- Improve histogram binning for better distribution visibility
- Add kernel density estimates to histograms

### Step 6: Add Visual Polish
- Implement consistent color palette (primary, success, warning, danger)
- Add subtle shadows/borders to separate sections
- Use better fonts (if available in matplotlib)
- Add subtle background patterns for different chart regions
- Include data quality indicators

## Validation & Rollout

### Test Strategy
- Test with various data sizes (10 to 10,000 measurements)
- Verify with different rejection densities
- Check readability at different output resolutions
- Test with edge cases (all rejected, no rejections, gaps)
- Validate annotation overlap prevention

### Manual QA Checklist
- [ ] Charts readable at 100% zoom
- [ ] No overlapping annotations
- [ ] Statistics clearly visible
- [ ] Color scheme works for colorblind users
- [ ] Date axes properly formatted
- [ ] All legends visible and clear

### Rollout Plan
1. Implement layout changes
2. Test with sample data
3. Add configuration options for layout preferences
4. Update documentation with new layout
5. Optional: Add legacy layout mode for compatibility

## Risks & Mitigations

### Risk 1: Matplotlib Limitations
- Some advanced layouts may be difficult in matplotlib
- Mitigation: Use GridSpec and careful positioning; consider matplotlib patches for custom elements

### Risk 2: File Size Increase
- Larger figure size may increase file size significantly
- Mitigation: Optimize DPI settings; offer quality presets

### Risk 3: Backward Compatibility
- Existing workflows may expect current layout
- Mitigation: Add layout version parameter; maintain old layout as option

## Acceptance Criteria
- [ ] Primary chart at least 4x larger than current
- [ ] No more than 5 rejection annotations visible at once
- [ ] Statistics readable without zooming
- [ ] All current information still accessible
- [ ] File size increase less than 50%
- [ ] Generation time increase less than 20%

## Out of Scope
- Interactive features (hover, zoom, pan)
- Multiple page output (keeping single file for now)
- New visualization types (keeping current chart types)
- Real-time updates
- Web-based dashboard

## Open Questions
1. Should we add user preferences for layout customization?
2. Is 24x16 inches too large for some use cases?
3. Should we support both landscape and portrait orientations?
4. Do we need to maintain exact visual compatibility with existing outputs?
5. Should rejection categories use icons in addition to text?

## Review Cycle
- Self-review: Layout provides clear visual hierarchy ✓
- Annotations are readable without overlap ✓
- Statistics panel is properly formatted ✓
- Color scheme is accessible ✓
- Implementation plan is detailed and actionable ✓

### Revisions Made
1. Added specific dimensions for figure size
2. Clarified annotation clustering strategy
3. Added fallback plan for Option A
4. Specified GridSpec approach for complex layout
5. Added data quality indicators to visual polish section