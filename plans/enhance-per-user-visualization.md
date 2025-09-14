# Plan: Enhanced Per-User Visualization with Quality Insights

## Summary
Transform the per-user weight visualization to provide comprehensive insights through raw data points, quality score details, Kalman filter visualization with confidence bands, and interactive exploration capabilities.

## Context
- Source: User request for improved per-user visualization
- Current state: Basic timeline with accepted/rejected points and quality score coloring
- Assumptions:
  - Users need detailed quality insights for each measurement
  - Kalman filter behavior should be visible for understanding
  - Raw vs filtered data comparison is valuable
- Constraints:
  - Must maintain performance with large datasets
  - Should be compatible with existing visualization infrastructure
  - Keep file sizes reasonable for web delivery

## Requirements

### Functional
1. Display raw weight measurements as distinct visual elements
2. Show quality score breakdown for each data point
3. Visualize Kalman filter predictions with confidence intervals
4. Display innovation/residual information
5. Interactive tooltips with comprehensive details
6. Toggle layers on/off for clarity
7. Zoom and pan capabilities for detailed exploration

### Non-functional
- Performance: Load and render within 2 seconds for 1000 points
- Accessibility: Keyboard navigation and screen reader support
- Responsiveness: Adapt to different screen sizes
- File size: Keep under 10MB for typical user data

## Alternatives

### Option A: Enhanced Plotly Implementation
- **Approach**: Extend current Plotly-based visualization with multiple traces and subplots
- **Pros**:
  - Builds on existing infrastructure
  - Rich interactivity out-of-box
  - Good performance with WebGL renderer
  - Export capabilities built-in
- **Cons**:
  - Larger file sizes with embedded data
  - Limited customization for complex tooltips
- **Risks**: May become cluttered with many layers

### Option B: D3.js Custom Visualization
- **Approach**: Build custom visualization using D3.js
- **Pros**:
  - Complete control over rendering
  - Smaller file sizes possible
  - Custom interactions and animations
- **Cons**:
  - Significant development effort
  - Need to implement zoom/pan/export manually
  - Steeper learning curve
- **Risks**: Time investment, browser compatibility

### Option C: Hybrid Dashboard with Multiple Charts
- **Approach**: Create dashboard with separate charts for different aspects
- **Pros**:
  - Clear separation of concerns
  - Each chart optimized for its purpose
  - Easier to understand
- **Cons**:
  - More screen space required
  - Potential for redundant information
  - Harder to correlate information
- **Risks**: User experience fragmentation

## Recommendation
**Option A: Enhanced Plotly Implementation** with smart layering and subplot organization.

**Rationale:**
- Leverages existing codebase and expertise
- Provides immediate value with minimal risk
- Plotly's built-in features handle most requirements
- Can progressively enhance with custom components if needed

## High-Level Design

### Architecture
```
visualization.py (enhanced)
    ├── Main Timeline Plot
    │   ├── Raw measurements (scatter)
    │   ├── Kalman filtered line
    │   ├── Confidence bands (filled area)
    │   └── Innovation markers
    ├── Quality Score Subplot
    │   ├── Overall score timeline
    │   └── Component breakdown (stacked area)
    ├── Residuals/Innovation Plot
    │   └── Bar chart with threshold lines
    └── Interactive Controls
        ├── Layer toggles
        ├── Date range selector
        └── Export options
```

### Data Model Changes
- Enhance result dictionary with:
  - `kalman_confidence_upper/lower`: Confidence interval bounds
  - `quality_components`: Detailed breakdown (safety, plausibility, consistency, reliability)
  - `innovation_normalized`: Standardized innovation for comparison
  - `prediction_error`: Kalman prediction vs actual

### Visual Layout
```
┌─────────────────────────────────────┐
│         User Weight Timeline         │
│  ┌─────────────────────────────┐    │
│  │  Raw ○ Filtered ─ ±Conf ░░  │    │
│  └─────────────────────────────┘    │
│  ┌─────────────────────────────┐    │
│  │  Quality Score Components    │    │
│  └─────────────────────────────┘    │
│  ┌─────────────────────────────┐    │
│  │  Innovation/Residuals        │    │
│  └─────────────────────────────┘    │
└─────────────────────────────────────┘
```

## Implementation Plan (No Code)

### Step 1: Enhance Data Pipeline
- Modify `processor.py` to calculate and store:
  - Kalman confidence intervals (±2σ from covariance)
  - Normalized innovation values
  - Prediction errors
- Update `quality_scorer.py` to expose component scores in results
- Ensure all metrics are serialized in output JSON

### Step 2: Create Enhanced Visualization Function
- New function `create_enhanced_weight_timeline()` in `visualization.py`
- Parameters:
  - results: Enhanced result data
  - user_id: User identifier
  - config: Visualization configuration (colors, sizes, etc.)
  - output_dir: Output directory
- Returns: Path to generated HTML file

### Step 3: Implement Main Weight Timeline
- Primary scatter trace for raw measurements:
  - Size based on quality score
  - Color gradient for quality (red→yellow→green)
  - Shape indicates source type
- Kalman filtered line:
  - Smooth line through filtered weights
  - Distinct color (blue)
  - Dashed for predictions
- Confidence bands:
  - Semi-transparent filled area
  - ±2σ bounds from Kalman covariance
  - Light blue fill

### Step 4: Add Quality Score Visualization
- Subplot below main timeline
- Stacked area chart showing:
  - Safety component (green)
  - Plausibility component (blue)
  - Consistency component (orange)
  - Reliability component (purple)
- Horizontal line at acceptance threshold
- Synchronized x-axis with main plot

### Step 5: Implement Innovation/Residuals Plot
- Bar chart showing innovation for each measurement
- Color coding:
  - Green: Low innovation (high confidence)
  - Yellow: Moderate innovation
  - Red: High innovation (low confidence)
- Reference lines at ±1σ, ±2σ, ±3σ

### Step 6: Create Rich Hover Information
- Custom hover template showing:
  - Timestamp and source
  - Raw weight vs filtered weight
  - Quality score breakdown
  - Innovation and confidence
  - Rejection reason (if applicable)
  - Kalman state (weight, trend)
- Format as structured HTML table

### Step 7: Add Interactive Controls
- Implement toggle buttons for:
  - Raw measurements on/off
  - Kalman line on/off
  - Confidence bands on/off
  - Quality components on/off
- Add range slider for date filtering
- Export buttons for PNG/SVG/CSV

### Step 8: Optimize Performance
- Use Plotly's WebGL renderer for large datasets
- Implement data decimation for >1000 points
- Lazy loading for subplot data
- Compress HTML output with minification

### Step 9: Create Comparison View
- Optional side-by-side view for:
  - Before/after quality scoring
  - Different Kalman configurations
  - Multiple time periods
- Synchronized zoom/pan across views

### Step 10: Add Statistical Summary Panel
- Fixed panel showing:
  - Current weight and trend
  - 7-day, 30-day changes
  - Quality score statistics
  - Acceptance rate over time
  - Data source distribution

## Validation & Rollout

### Test Strategy
1. **Unit tests** for new calculation functions
2. **Visual regression tests** using screenshot comparison
3. **Performance tests** with varying data sizes (100, 1000, 10000 points)
4. **Browser compatibility** testing (Chrome, Firefox, Safari, Edge)
5. **Accessibility testing** with screen readers

### Manual QA Checklist
- [ ] All data points visible and correctly positioned
- [ ] Hover information accurate and complete
- [ ] Toggles work without breaking layout
- [ ] Export functions produce valid files
- [ ] Performance acceptable on mobile devices
- [ ] No JavaScript errors in console

### Rollout Plan
1. **Phase 1**: Deploy to test environment with sample data
2. **Phase 2**: A/B test with 10% of users
3. **Phase 3**: Gather feedback and iterate
4. **Phase 4**: Full deployment with old version archived

### Rollback Strategy
- Keep previous visualization function available
- Feature flag to switch between versions
- Monitor error rates and performance metrics

## Risks & Mitigations

### Risk 1: Information Overload
- **Impact**: Users overwhelmed by too much data
- **Mitigation**: Progressive disclosure with collapsed sections, smart defaults

### Risk 2: Performance Degradation
- **Impact**: Slow loading for users with many measurements
- **Mitigation**: Data pagination, WebGL rendering, server-side pre-processing

### Risk 3: Browser Compatibility
- **Impact**: Visualization breaks on older browsers
- **Mitigation**: Polyfills, graceful degradation, fallback to simple view

### Risk 4: Large File Sizes
- **Impact**: Slow downloads on mobile networks
- **Mitigation**: Data compression, CDN hosting for libraries, lazy loading

## Acceptance Criteria

1. ✓ Raw measurements displayed as distinct points
2. ✓ Quality score components visible for each measurement
3. ✓ Kalman filter line with confidence bands rendered
4. ✓ Innovation/residuals clearly visualized
5. ✓ All layers can be toggled on/off
6. ✓ Hover shows comprehensive information
7. ✓ Export functions work correctly
8. ✓ Performance <2s load time for 1000 points
9. ✓ Mobile responsive design
10. ✓ No regression in existing functionality

## Out of Scope

- Real-time streaming updates
- 3D visualizations
- Machine learning predictions
- Multi-user comparison in same view
- Custom color schemes per user
- Animated transitions between states
- PDF report generation

## Open Questions

1. Should we add annotation capabilities for clinical notes?
2. Do we need to support custom date ranges in the UI?
3. Should quality thresholds be adjustable in the visualization?
4. Is there a need for data export in specific medical formats?
5. Should we integrate with external visualization tools?

## Review Cycle

### Self-Review Notes
- Considered performance implications carefully
- Ensured backward compatibility maintained
- Focused on user value over technical complexity
- Kept implementation steps concrete and actionable

### Revisions Made
1. Added specific performance targets
2. Clarified visual layout with ASCII diagram
3. Expanded hover information details
4. Added accessibility requirements
5. Included rollback strategy

### Stakeholder Feedback Points
- Clinical team: Need for annotation capabilities?
- Engineering: WebGL vs Canvas rendering preference?
- Product: Priority of features for MVP?
- QA: Automated testing strategy approval?