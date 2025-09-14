# Plan: Enhanced Dashboard with Kalman Filter Insights

## Summary
Enhance the interactive dashboard to include a comprehensive main weight graph that combines all data points with Kalman filter visualization, uncertainty bands, and detailed filter performance metrics. This will provide users with deep insights into how the Kalman filter is processing their weight measurements.

## Context
- Current State: Interactive Plotly dashboard with quality scoring visualization
- Gap: Missing actual weight values and Kalman filter behavior visualization
- Reference: Need a main graph similar to provided image showing:
  - Actual weight measurements (accepted/rejected)
  - Kalman filtered line
  - Uncertainty/confidence bands
  - Baseline reference
  - Filter performance metrics

## Requirements

### Functional
- **Main Weight Graph**
  - Display all weight measurements (accepted as filled markers, rejected as hollow/different)
  - Show Kalman filtered line (smooth trajectory)
  - Display uncertainty bands (±1σ and ±2σ)
  - Include baseline weight reference line
  - Show extreme deviation thresholds
  - Mark reset points and gaps
  
- **Kalman Filter Insights**
  - Innovation (measurement - prediction) subplot
  - Normalized innovation with statistical significance
  - Measurement confidence evolution
  - Daily change distribution
  - Filter convergence indicators
  
- **Enhanced Interactivity**
  - Synchronized zoom across all Kalman-related charts
  - Hover details showing:
    - Raw weight
    - Filtered weight
    - Prediction
    - Innovation
    - Uncertainty
    - Quality score
  - Click to see detailed filter state at that point

### Non-functional
- Maintain <3 second load time
- Clear visual hierarchy (main graph prominent)
- Consistent color scheme across filter visualizations
- Mobile-responsive layout adjustments

## Alternatives

### Option A: Replace Current Main Chart
- Swap quality-focused chart with weight/Kalman chart
- Pros: Clean, focused on primary data
- Cons: Loses quality overlay prominence
- Risk: Users might miss quality insights

### Option B: Dual-View Toggle
- Toggle between quality view and Kalman view
- Pros: Full detail for both perspectives
- Cons: Extra interaction required, context switching
- Risk: Users might not discover both views

### Option C: Integrated Enhanced View (Recommended)
- Combine weight values, Kalman filter, AND quality in main chart
- Add Kalman-specific subplots below
- Pros: Complete picture, no context switching
- Cons: More complex visualization
- Risk: Information density

## Recommendation
**Option C: Integrated Enhanced View** - Create a comprehensive main chart that shows weight values with both Kalman filter processing and quality scoring, supplemented by detailed filter analytics subplots.

## High-Level Design

### Enhanced Dashboard Layout
```
┌────────────────────────────────────────────────────────────────┐
│                 Weight Stream Analytics Dashboard               │
├────────────────────────────────────────────────────────────────┤
│ [Overview] [Kalman Analysis] [Quality] [Sources] [Export]      │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────────────────────────────┬────────────┐│
│  │  MAIN: Weight Measurements & Kalman Filter   │ Statistics ││
│  │                                               │            ││
│  │  ━━━ Kalman Filtered (with uncertainty)      │ Current:   ││
│  │  ● Accepted  ○ Rejected  ▼ Reset Points      │ Weight: 112││
│  │  ░░░ ±1σ band  ░░░ ±2σ band                  │ Filtered:  ││
│  │  --- Baseline  ··· Extreme thresholds        │   111.8    ││
│  │  Quality: [color gradient on points]          │ Uncert: ±1.2││
│  └──────────────────────────────────────────────┴────────────┘│
│                                                                 │
│  ┌─────────────┬─────────────┬─────────────┬─────────────────┐│
│  │ Innovation  │ Normalized  │ Confidence  │ Daily Changes   ││
│  │ (Meas-Pred) │ Innovation  │ Evolution   │ Distribution    ││
│  └─────────────┴─────────────┴─────────────┴─────────────────┘│
│                                                                 │
│  ┌──────────────────────────┬──────────────────────────────────┐│
│  │ Filter Performance        │ Processing Overview              ││
│  │ • Convergence metrics     │ • Measurements by source         ││
│  │ • Innovation statistics   │ • Rejection categories          ││
│  │ • State covariance        │ • Quality vs Innovation         ││
│  └──────────────────────────┴──────────────────────────────────┘│
└────────────────────────────────────────────────────────────────┘
```

### Kalman Analysis Tab (New)
```
┌────────────────────────────────────────────────────────────────┐
│                    Kalman Filter Deep Dive                      │
├────────────────────────────────────────────────────────────────┤
│  ┌────────────────────────────────────────────────────────────┐│
│  │ State Evolution (Weight & Trend)                            ││
│  │ Upper: Filtered weight with components                      ││
│  │ Lower: Trend/velocity component                             ││
│  └────────────────────────────────────────────────────────────┘│
│                                                                 │
│  ┌─────────────────────┬──────────────────────────────────────┐│
│  │ Covariance Matrix   │ Innovation Autocorrelation            ││
│  │ Evolution           │ (Should be white noise if optimal)    ││
│  └─────────────────────┴──────────────────────────────────────┘│
│                                                                 │
│  ┌────────────────────────────────────────────────────────────┐│
│  │ Filter Diagnostics                                          ││
│  │ • NIS (Normalized Innovation Squared) test                  ││
│  │ • Whiteness test p-values over time                         ││
│  │ • Consistency metrics                                       ││
│  └────────────────────────────────────────────────────────────┘│
└────────────────────────────────────────────────────────────────┘
```

## Implementation Plan

### Phase 1: Data Extraction Enhancement (Day 1)
**Files**: `src/viz_quality.py`, new `src/viz_kalman.py`

1. **Extract Kalman Filter Data**
   - Filtered weight values
   - Prediction values
   - Innovation (measurement - prediction)
   - State covariance/uncertainty
   - Reset points and gap markers
   
2. **Calculate Filter Metrics**
   - Innovation mean and variance
   - Normalized innovation squared (NIS)
   - Measurement confidence scores
   - Daily weight changes
   - Convergence indicators

3. **Prepare Time Series**
   - Align all measurements temporally
   - Interpolate uncertainty bands
   - Calculate baseline (initial or average)
   - Identify extreme thresholds

### Phase 2: Main Weight Graph (Day 2-3)
**Files**: `src/viz_plotly.py`

1. **Enhanced Main Chart**
   ```python
   # Pseudocode structure
   fig.add_trace(go.Scatter(
       name="Kalman Filtered",
       x=timestamps,
       y=filtered_weights,
       mode='lines',
       line=dict(color='blue', width=2)
   ))
   
   # Uncertainty bands
   fig.add_trace(go.Scatter(
       name="±1σ",
       x=timestamps + timestamps[::-1],
       y=upper_1sigma + lower_1sigma[::-1],
       fill='toself',
       fillcolor='rgba(0,100,200,0.2)',
       line=dict(color='rgba(255,255,255,0)')
   ))
   
   # Accepted measurements
   fig.add_trace(go.Scatter(
       name="Accepted",
       x=accepted_times,
       y=accepted_weights,
       mode='markers',
       marker=dict(
           size=8,
           color=quality_scores,
           colorscale='RdYlGn',
           symbol='circle'
       )
   ))
   ```

2. **Visual Elements**
   - Baseline reference line (dashed)
   - Extreme deviation thresholds (dotted)
   - Reset point markers (triangles)
   - Gap indicators (shaded regions)
   - Quality color overlay on points

3. **Interactive Features**
   - Unified hover showing all metrics
   - Click for detailed state view
   - Range slider for time navigation
   - Buttons for uncertainty band toggle

### Phase 3: Kalman Analytics Subplots (Day 4)
**Files**: `src/viz_plotly.py`, `src/viz_kalman.py`

1. **Innovation Chart**
   - Time series of (measurement - prediction)
   - Zero reference line
   - Color by magnitude
   - Running statistics overlay

2. **Normalized Innovation**
   - Innovation / sqrt(innovation_covariance)
   - ±2σ significance bands
   - Highlight statistical outliers
   - Goodness-of-fit indicators

3. **Confidence Evolution**
   - Measurement confidence over time
   - State uncertainty evolution
   - Convergence indicators
   - Reset impact visualization

4. **Daily Change Distribution**
   - Histogram of day-to-day changes
   - Normal distribution overlay
   - Percentile markers
   - Outlier identification

### Phase 4: Filter Performance Panel (Day 5)
**Files**: `src/viz_kalman.py`

1. **Performance Metrics**
   - Mean innovation (should be ~0)
   - Innovation variance ratio
   - Normalized innovation squared (NIS)
   - Autocorrelation metrics
   - Convergence rate

2. **State Diagnostics**
   - Current state estimate
   - State covariance matrix
   - Prediction accuracy
   - Update vs prediction ratio

3. **Visual Indicators**
   - Traffic light for filter health
   - Trend arrows for convergence
   - Warning flags for issues

### Phase 5: Kalman Analysis Tab (Day 6)
**Files**: `src/viz_plotly.py`

1. **State Evolution Plots**
   - Dual plot: weight and trend
   - Component breakdown
   - Uncertainty evolution
   - Phase space view (optional)

2. **Covariance Analysis**
   - Heatmap of covariance matrix
   - Eigenvalue evolution
   - Correlation structure

3. **Statistical Tests**
   - Innovation whiteness test
   - Chi-squared test for NIS
   - Consistency checks
   - Optimality indicators

### Phase 6: Integration & Polish (Day 7)
**Files**: All visualization files

1. **Layout Optimization**
   - Responsive grid adjustments
   - Proper spacing and alignment
   - Consistent color schemes
   - Clear visual hierarchy

2. **Performance Tuning**
   - Optimize data calculations
   - Implement caching
   - Reduce redundant updates
   - WebGL for large datasets

3. **User Experience**
   - Tooltips and help text
   - Legend improvements
   - Export options for Kalman data
   - Documentation

## Data Requirements

### From Processor Results
```python
{
    "timestamp": datetime,
    "weight": float,
    "accepted": bool,
    "kalman_state": {
        "filtered_weight": float,
        "filtered_trend": float,
        "state_covariance": [[float]],
        "prediction": float,
        "innovation": float,
        "innovation_covariance": float,
        "gain": [float],
        "likelihood": float
    },
    "quality_score": float,
    "is_reset": bool,
    "gap_days": int
}
```

## Validation & Testing

### Test Cases
1. **Visual Accuracy**
   - Kalman line passes through/near accepted points
   - Uncertainty bands contain appropriate % of measurements
   - Innovation centers around zero
   - Normalized innovation follows standard normal

2. **Interactivity**
   - All hover information displays correctly
   - Zoom synchronizes across subplots
   - Click actions work as expected
   - Export includes all Kalman data

3. **Performance**
   - Load time <3 seconds for 10k points
   - Smooth interactions (60 fps)
   - Memory usage reasonable

### Acceptance Criteria
- [ ] Main graph shows weight values with Kalman filter
- [ ] Uncertainty bands clearly visible
- [ ] Innovation subplot shows filter performance
- [ ] Confidence evolution tracked
- [ ] Daily changes analyzed
- [ ] Filter metrics calculated and displayed
- [ ] Interactive features functional
- [ ] Mobile responsive
- [ ] Documentation complete

## Risks & Mitigations

### Risk 1: Information Overload
**Mitigation**: Progressive disclosure, clear visual hierarchy, optional detail levels

### Risk 2: Kalman Data Not Available
**Mitigation**: Graceful degradation, calculate from state if needed, clear messages

### Risk 3: Performance Impact
**Mitigation**: Lazy loading, data decimation, caching strategies

### Risk 4: User Confusion
**Mitigation**: Clear labels, comprehensive tooltips, help documentation

## Success Metrics
- Users understand their weight trends better
- Filter issues identified more quickly
- Reduced questions about "jumpy" data
- Increased trust in filtered values

## Implementation Priority
1. **Critical**: Main weight graph with Kalman line
2. **High**: Innovation and confidence subplots
3. **Medium**: Filter performance metrics
4. **Low**: Advanced statistical diagnostics

## Council Review

**Butler Lampson** (Simplicity): "The main graph should be immediately understandable - weight over time with smoothing. Hide complexity in subplots."

**Don Norman** (User Experience): "Users care about their weight trend, not Kalman math. Use plain language: 'smoothed weight', 'confidence', not 'innovation covariance'."

**Edward Tufte** (Data Visualization): "The uncertainty bands are data-ink well spent. They show the filter's confidence without cluttering."

**Leslie Lamport** (Distributed Systems): "Ensure the temporal ordering is crystal clear, especially around resets and gaps."

## Next Steps
1. Review and approve plan
2. Verify Kalman data availability in processor output
3. Begin Phase 1 implementation
4. Iterate based on initial results

## Estimated Timeline
- Day 1: Data extraction
- Day 2-3: Main weight graph
- Day 4: Kalman analytics
- Day 5: Performance panel
- Day 6: Analysis tab
- Day 7: Integration and testing
- **Total: 1 week for full implementation**