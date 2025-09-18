# Investigation: Visualization Recommendations for Raw vs Filtered Weight Data

## Bottom Line

**Root Cause**: Need effective visualizations to show filtering impact on weight measurements  
**Fix Location**: Extend existing viz infrastructure in `src/viz/` and `src/analysis/`  
**Confidence**: High

## What's Happening

The system compares raw weight measurements (with outliers/noise) against Kalman-filtered data. Current analysis in `create-report/run.py` calculates extensive metrics but lacks visualization.

## Why It Happens

**Primary Cause**: Analysis focuses on numerical metrics without visual representation  
**Trigger**: `create-report/run.py:476-783` - Comprehensive analysis without charts  
**Decision Point**: No visualization imports or chart generation in run.py

## Evidence

- **Key File**: `src/viz/visualization.py:12-17` - Already uses Plotly for interactive charts
- **Search Used**: `rg "visualiz|chart|plot" -g "*.py"` - Found existing viz infrastructure
- **Key File**: `src/analysis/analysis_visualizer.py:38-39` - Uses matplotlib/seaborn for static charts
- **Key File**: `create-report/run.py:476-783` - Calculates all metrics but no viz

## Next Steps

### 1. Interactive Comparison Dashboard (Plotly)
Create `create-report/comparison_dashboard.py`:
- **Before/After Timeline**: Dual-axis plot showing raw (scatter) vs filtered (line)
- **Difference Heatmap**: User×Time matrix colored by |raw-filtered| difference
- **Statistical Distribution**: Side-by-side histograms with KDE overlay
- **Outlier Detection View**: Highlight removed points with hover details

### 2. Static Analysis Charts (Matplotlib/Seaborn)
Extend `create-report/run.py` with visualization module:
- **Variance Reduction Chart**: Bar chart comparing std dev before/after
- **Trajectory Alignment Plot**: Scatter plot (raw_change vs filtered_change)
- **Quality Score Impact**: Heatmap of quality score vs difference magnitude
- **Noise Pattern Analysis**: Time-series of daily variance reduction

### 3. Key Visualization Types

**For Variance Reduction (lines 548-566)**:
```python
# Paired bar chart
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].bar(['Raw', 'Filtered'], [raw_std, filtered_std])
ax[1].plot(time, rolling_variance_ratio)  # Show improvement over time
```

**For Outlier Removal (lines 567-611)**:
```python
# Plotly interactive scatter
fig = go.Figure()
fig.add_trace(go.Scatter(x=times, y=raw_weights, mode='markers', 
                         name='Raw', marker=dict(color='red')))
fig.add_trace(go.Scatter(x=times, y=filtered_weights, mode='lines',
                         name='Filtered', line=dict(color='blue')))
# Highlight removed points
removed_mask = ~np.isin(raw_ids, filtered_ids)
fig.add_trace(go.Scatter(x=times[removed_mask], y=raw_weights[removed_mask],
                         mode='markers', marker=dict(size=12, symbol='x'),
                         name='Outliers Removed'))
```

**For Consistency Improvements (lines 612-643)**:
```python
# Violin plot for daily changes
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.violinplot([raw_daily_changes, filtered_daily_changes])
ax2.hist2d(raw_changes, filtered_changes, bins=30, cmap='Blues')
```

**For Trajectory Analysis (lines 644-688)**:
```python
# Sankey diagram for trajectory alignment
import plotly.graph_objects as go
fig = go.Figure(go.Sankey(
    node=dict(label=["Raw Loss", "Raw Gain", "Filtered Loss", "Filtered Gain"]),
    link=dict(source=[0,0,1,1], target=[2,3,2,3], 
              value=trajectory_flows)
))
```

### 4. Implementation Priority

1. **Immediate**: Add matplotlib charts to `run.py` for static reports
2. **Next Sprint**: Create Plotly dashboard for interactive exploration  
3. **Future**: Integrate with existing `src/viz/visualization.py` timeline

### 5. Library Recommendations

**Primary**: Plotly (already in use)
- Best for: Interactive dashboards, hover details, zoom/pan
- Use cases: Timeline comparison, outlier exploration

**Secondary**: Matplotlib + Seaborn (already in use)  
- Best for: Static reports, statistical plots, publication-ready
- Use cases: Distribution analysis, statistical comparisons

**Avoid**: Bokeh, Altair (would add dependencies without clear benefit)

## Risks

- Performance with large datasets (>10k measurements per user)
- Browser memory limits for interactive charts with many users
