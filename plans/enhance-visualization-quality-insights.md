# Plan: Enhanced Visualization with Quality Score Insights

## Summary
Transform the current visualization system to fully leverage the unified quality scoring system, providing deep insights into measurement quality, acceptance decisions, and trends. This will make the scoring transparent, actionable, and help identify data quality patterns.

## Context
- Source: Investigation of visualization gaps after quality scoring implementation
- Assumptions:
  - Quality scoring system is fully implemented and working
  - Quality data exists in processor results but isn't visualized
  - Users need to understand why measurements are accepted/rejected
- Constraints:
  - Must maintain existing dashboard layout structure
  - Cannot break backward compatibility
  - Performance must remain acceptable (<2s render time)

## Requirements

### Functional
- Display quality scores for all measurements
- Show component score breakdowns (safety, plausibility, consistency, reliability)
- Visualize quality trends over time
- Highlight measurements near rejection threshold
- Provide actionable insights for improving data quality
- Show quality score distributions and patterns

### Non-functional
- Maintain current dashboard rendering performance
- Keep visualizations intuitive and not overwhelming
- Ensure color schemes are accessible (colorblind-friendly options)
- Support both light and dark themes (future)

## Alternatives

### Option A: Overlay Quality on Existing Charts
- Approach: Add quality information as overlays/colors on current charts
- Pros: Minimal layout changes, familiar interface, quick to implement
- Cons: May clutter existing charts, limited space for new insights
- Risks: Information overload, reduced clarity

### Option B: Replace Low-Value Charts with Quality Charts
- Approach: Replace daily change distribution and other less-used charts with quality-focused ones
- Pros: Dedicated space for quality insights, cleaner presentation, better focus
- Cons: Loses some existing functionality, requires user adjustment
- Risks: Users may miss removed charts

### Option C: Add Second Dashboard Page for Quality
- Approach: Create separate quality-focused dashboard
- Pros: Comprehensive quality analysis, no loss of existing features, specialized views
- Cons: More complex navigation, duplicate some data, longer implementation
- Risks: Users may not discover second page

## Recommendation
**Option B** - Replace low-value charts with quality-focused visualizations, enhanced with selective overlays from Option A.

**Rationale**: 
- Daily change distribution and some other charts provide limited value
- Quality insights are more actionable than distribution statistics
- Maintains single-page dashboard simplicity
- Can be implemented incrementally

## High-Level Design

### Architecture
```
Dashboard Layout (3x5 grid):
┌─────────────────────────────────────────────┬──────────┐
│ Main Chart (with quality overlay)           │ Stats    │
│ [Quality-colored points + threshold line]   │ Panel    │
├──────────┬──────────┬──────────┬──────────┤ [Quality │
│ Quality  │ Component│ Quality  │ Quality   │  Metrics]│
│ Timeline │ Scores   │ Distrib. │ Insights  │          │
├──────────┴──────────┼──────────┴──────────┤          │
│ Rejection Analysis   │ Source Quality Map   │          │
└─────────────────────┴──────────────────────┴──────────┘
```

### Data Flow
1. Processor outputs quality scores → Results dictionary
2. Visualization extracts quality data from results
3. Quality analyzer computes insights and patterns
4. Charts render with quality-aware styling
5. Interactive tooltips show detailed scores

### Affected Files
- `src/visualization.py` - Main visualization logic
- `src/viz_constants.py` - Add quality color schemes
- `src/viz_quality.py` (new) - Quality-specific visualization helpers
- `config.toml` - Add visualization preferences

## Implementation Plan (No Code)

### Phase 1: Add Quality Data Extraction
**Files**: `src/visualization.py`

1. **Extract Quality Scores from Results**
   - Parse `quality_score` field from each result
   - Parse `quality_components` dictionary
   - Handle missing quality data (backward compatibility)
   - Build time series of quality scores

2. **Calculate Quality Statistics**
   - Mean, median, std of quality scores
   - Component score averages
   - Count near-threshold measurements (0.55-0.65)
   - Identify quality trends (improving/declining)

3. **Categorize Quality Levels**
   - High quality: > 0.8
   - Good quality: 0.7-0.8
   - Acceptable: 0.6-0.7
   - Near threshold: 0.55-0.6
   - Rejected: < 0.6

### Phase 2: Enhance Main Chart
**Files**: `src/visualization.py`, `src/viz_constants.py`

1. **Add Quality-Based Point Coloring**
   - Create color gradient: deep green (1.0) → yellow (0.7) → orange (0.6) → red (0.0)
   - Color accepted points by quality score
   - Keep rejected points with current edge coloring
   - Add alpha channel for score confidence

2. **Add Quality Threshold Line**
   - Horizontal dashed line at quality score 0.6
   - Label: "Quality Threshold"
   - Shade rejection zone below threshold

3. **Enhance Tooltips** (if interactive)
   - Show quality score on hover
   - Display component scores
   - Explain acceptance/rejection decision

### Phase 3: Create Quality Timeline Chart
**Files**: `src/visualization.py`
**Location**: Replace innovation chart (row 2, col 1)

1. **Quality Score Time Series**
   - Line plot of quality scores over time
   - Color-coded background zones (high/good/acceptable/rejected)
   - Highlight measurements near threshold with markers
   - Show rolling average trend line

2. **Add Context Markers**
   - Mark state resets
   - Annotate significant quality drops
   - Show source changes that affect quality

3. **Interactive Elements**
   - Click to see component breakdown
   - Zoom to investigate quality issues

### Phase 4: Create Component Scores Chart
**Files**: `src/visualization.py`
**Location**: Replace normalized innovation (row 2, col 2)

1. **Stacked Area Chart**
   - Four areas: safety, plausibility, consistency, reliability
   - Weighted by component importance
   - Overall score as bold line overlay
   - Time-aligned with main chart

2. **Component Insights**
   - Identify weakest component over time
   - Show which component causes rejections
   - Highlight component improvements/degradations

3. **Visual Design**
   - Use distinct but harmonious colors
   - Add subtle patterns for accessibility
   - Include component legends with weights

### Phase 5: Create Quality Distribution Chart
**Files**: `src/visualization.py`
**Location**: Replace confidence chart (row 2, col 3)

1. **Histogram of Quality Scores**
   - Bins from 0.0 to 1.0 (20 bins)
   - Separate colors for accepted/rejected
   - Vertical line at threshold (0.6)
   - Show count and percentage labels

2. **Statistical Overlays**
   - Mean and median lines
   - Standard deviation shading
   - Percentile markers (25th, 75th)

3. **Insights Panel**
   - "X% of measurements are high quality"
   - "Y measurements barely passed threshold"
   - "Most common quality range: 0.7-0.8"

### Phase 6: Create Quality Insights Chart
**Files**: `src/visualization.py`, `src/viz_quality.py` (new)
**Location**: Replace daily change distribution (row 2, col 4)

1. **Actionable Insights Display**
   - "Improvement Opportunities" section
   - "Quality Alerts" for concerning patterns
   - "Best Practices" from high-quality sources

2. **Pattern Detection**
   - Time-of-day quality patterns
   - Source-specific quality issues
   - Correlation with gaps/resets

3. **Recommendations**
   - "Plausibility scores low after gaps - consider reset logic"
   - "Source X has 20% lower quality than average"
   - "Quality improves with frequent measurements"

### Phase 7: Update Statistics Panel
**Files**: `src/visualization.py`

1. **Add Quality Metrics Section**
   ```
   QUALITY METRICS
   Overall Quality: 0.75 (Good)
   Acceptance Rate: 94.2%
   Near Threshold: 5 measurements
   
   Component Averages:
   Safety:       0.95 ████████░░
   Plausibility: 0.72 ███████░░░
   Consistency:  0.68 ██████░░░░
   Reliability:  0.85 ████████░░
   
   Quality Trend: ↑ Improving
   Best Source: patient-device (0.89)
   Needs Attention: patient-upload (0.61)
   ```

2. **Add Visual Elements**
   - Mini bar charts for component scores
   - Color coding for quality levels
   - Trend arrows for changes

### Phase 8: Enhance Rejection Analysis
**Files**: `src/visualization.py`
**Location**: Bottom row, first two columns

1. **Quality-Based Rejection Grouping**
   - Group by quality score ranges (0-0.2, 0.2-0.4, 0.4-0.6)
   - Show which component failed for each group
   - Display "would accept if" analysis

2. **Rejection Insights**
   - "45% of rejections due to low plausibility"
   - "Safety component prevented 12 dangerous values"
   - "Consistency issues after gaps >7 days"

3. **Visual Improvements**
   - Use quality score gradient for bar colors
   - Add component breakdown within bars
   - Show improvement potential

### Phase 9: Create Source Quality Map
**Files**: `src/visualization.py`
**Location**: Bottom row, columns 3-4

1. **Source Quality Comparison**
   - Heatmap or grouped bar chart
   - Quality score by source
   - Component breakdown per source
   - Acceptance rate overlay

2. **Source Insights**
   - Identify problematic sources
   - Show improvement over time
   - Correlate with measurement frequency

3. **Recommendations**
   - "Consider stricter validation for source X"
   - "Source Y quality improved 15% this month"

### Phase 10: Add Configuration Options
**Files**: `config.toml`, `src/visualization.py`

1. **User Preferences**
   ```toml
   [visualization.quality]
   show_quality_scores = true
   quality_color_scheme = "gradient"  # or "discrete"
   highlight_threshold = true
   threshold_buffer_zone = 0.05  # highlight 0.55-0.65
   show_component_details = true
   quality_insights_enabled = true
   ```

2. **Accessibility Options**
   - Colorblind-friendly palettes
   - High contrast mode
   - Pattern fills for quality zones

## Validation & Rollout

### Test Strategy
1. **Visual Regression Tests**
   - Compare before/after screenshots
   - Verify layout stability
   - Check responsive behavior

2. **Performance Tests**
   - Measure render time with quality features
   - Test with large datasets (10k+ points)
   - Profile memory usage

3. **User Acceptance Tests**
   - Quality scores clearly visible
   - Insights are actionable
   - No information overload

### Manual QA Checklist
- [ ] Quality scores display correctly for all measurements
- [ ] Component breakdowns sum to overall score
- [ ] Color gradients are smooth and distinguishable
- [ ] Statistics panel shows accurate metrics
- [ ] Insights are relevant and actionable
- [ ] Charts align temporally
- [ ] Legends are clear and complete
- [ ] Performance remains acceptable

### Rollout Plan
1. **Week 1**: Implement Phases 1-2 (data extraction, main chart)
2. **Week 2**: Implement Phases 3-5 (quality charts)
3. **Week 3**: Implement Phases 6-7 (insights, statistics)
4. **Week 4**: Implement Phases 8-10 (rejection analysis, configuration)
5. **Week 5**: Testing and refinement
6. **Week 6**: Documentation and deployment

## Risks & Mitigations

### Risk 1: Information Overload
**Mitigation**: Progressive disclosure - basic view by default, detailed on demand

### Risk 2: Performance Degradation
**Mitigation**: Lazy loading, data sampling for large datasets, caching calculations

### Risk 3: Color Accessibility Issues
**Mitigation**: Multiple color schemes, patterns as fallback, WCAG compliance

### Risk 4: Breaking Existing Workflows
**Mitigation**: Feature flags, gradual rollout, preserve original mode option

## Acceptance Criteria
- [ ] Quality scores visible on main chart with color coding
- [ ] Component scores breakdown available
- [ ] Quality timeline shows trends and patterns
- [ ] Distribution chart reveals quality profile
- [ ] Insights provide actionable recommendations
- [ ] Statistics panel includes quality metrics
- [ ] Rejection analysis explains quality failures
- [ ] Source quality comparison identifies issues
- [ ] Configuration allows customization
- [ ] Performance impact < 10% on render time

## Out of Scope
- Real-time quality score updates
- Interactive drill-down dashboards
- Machine learning for quality prediction
- Custom quality scoring formulas per user
- Export of quality reports
- Quality score animations
- Mobile-responsive layouts

## Open Questions
1. Should we show quality scores for historical data before system was implemented?
   - **Recommendation**: No, mark as "N/A" to avoid confusion

2. How many quality insights should we show at once?
   - **Recommendation**: Top 3 most actionable

3. Should quality colors override source colors in main chart?
   - **Recommendation**: Use quality for fill, source for shape/edge

4. Do we need quality score history beyond the visible window?
   - **Recommendation**: Yes, for trend calculation (keep 30 days)

## Review Cycle

### Council Feedback Integration

**Butler Lampson** (Simplicity): "Keep the quality overlay subtle on the main chart. Don't overwhelm with too many metrics at once."
- ✓ Progressive disclosure approach
- ✓ Clean, minimal default view

**Don Norman** (User Experience): "Users need to immediately understand what quality scores mean for their data. Use intuitive colors and clear labels."
- ✓ Traffic light color scheme (green/yellow/red)
- ✓ Plain language insights
- ✓ Contextual help text

**Edward Tufte** (Data Visualization): "Maximize data-ink ratio. Every pixel should convey information."
- ✓ Remove redundant charts
- ✓ Use small multiples for components
- ✓ Integrate quality into existing visualizations

**Barbara Liskov** (Interfaces): "The visualization API should be extensible for future quality metrics."
- ✓ Modular chart components
- ✓ Clean data interfaces
- ✓ Pluggable insight generators

### Self-Review Notes
- Considered removing all existing charts but decided gradual replacement is safer
- Added configuration options after realizing one-size-fits-all won't work
- Included accessibility features as core requirement, not afterthought
- Balanced detail vs. clarity by using progressive disclosure

## Implementation Priority

### Phase Priority Matrix
| Phase | Impact | Effort | Priority | Dependencies |
|-------|--------|--------|----------|--------------|
| 1. Data Extraction | High | Low | **Critical** | None |
| 2. Main Chart Enhancement | High | Medium | **Critical** | Phase 1 |
| 3. Quality Timeline | High | Medium | **High** | Phase 1 |
| 4. Component Scores | Medium | Medium | **Medium** | Phase 1 |
| 5. Distribution Chart | Medium | Low | **High** | Phase 1 |
| 6. Quality Insights | High | High | **Medium** | Phases 1-5 |
| 7. Statistics Panel | High | Low | **High** | Phase 1 |
| 8. Rejection Analysis | Medium | Medium | **Medium** | Phases 1,5 |
| 9. Source Quality Map | Low | Medium | **Low** | Phases 1,5 |
| 10. Configuration | Low | Low | **Low** | All phases |

### Quick Wins (Week 1)
1. Extract quality data (Phase 1)
2. Update statistics panel (Phase 7)
3. Add quality threshold line to main chart

### High Impact (Weeks 2-3)
1. Full main chart enhancement (Phase 2)
2. Quality timeline chart (Phase 3)
3. Distribution chart (Phase 5)

### Completeness (Weeks 4-5)
1. Component scores (Phase 4)
2. Quality insights (Phase 6)
3. Rejection analysis (Phase 8)

### Polish (Week 6)
1. Source quality map (Phase 9)
2. Configuration options (Phase 10)
3. Documentation and testing