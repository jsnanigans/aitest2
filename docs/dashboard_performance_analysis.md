# Dashboard Loading Performance Analysis

## Problem Identified
Individual dashboard graphs take too long to load when switching between users due to:

1. **External CDN Dependency** - Every dashboard loads Plotly.js from CDN (3MB+)
2. **Network Latency** - Each user switch requires network request to cdn.plot.ly
3. **Repeated Downloads** - Browser may not cache effectively across iframes
4. **Large Script Parsing** - Full Plotly library parsed for each dashboard
5. **Complex Visualizations** - Each dashboard has 10+ interactive plots

## Current Implementation Issues

### File Sizes
- Each dashboard HTML: ~26KB base + 3MB Plotly from CDN
- Total transfer per user switch: ~3MB
- With 5 users switching: ~15MB total transfer

### Loading Process (Current)
1. User clicks → iframe src changes
2. Browser loads HTML (26KB)
3. Browser discovers Plotly CDN link
4. Network request to cdn.plot.ly (3MB)
5. Download and parse Plotly.js
6. Execute dashboard JavaScript
7. Render all plots

## Performance Optimization Solutions

### Solution 1: Inline Plotly (Recommended for Small Sets)
**Approach**: Embed Plotly.js directly in each HTML
```python
fig.write_html(
    str(html_file),
    include_plotlyjs='inline',  # Instead of 'cdn'
    ...
)
```
**Pros**: 
- No network requests
- Instant loading after initial cache
- Works offline

**Cons**: 
- Larger HTML files (~3MB each)
- More disk space used

### Solution 2: Directory-Level Plotly (Recommended)
**Approach**: Use a shared Plotly.js file in the output directory
```python
fig.write_html(
    str(html_file),
    include_plotlyjs='directory',  # Shared plotly.min.js
    ...
)
```
**Pros**: 
- Single Plotly.js file for all dashboards
- Browser caches effectively
- Smaller individual HTML files

**Cons**: 
- Requires relative path handling

### Solution 3: Preload in Parent Frame
**Approach**: Load Plotly once in index.html, share with iframes
```javascript
// In index.html
<script src="plotly.min.js"></script>
<script>
window.plotlyLib = Plotly;
</script>

// In dashboards
if (window.parent.plotlyLib) {
    Plotly = window.parent.plotlyLib;
}
```
**Pros**: 
- Single load for all dashboards
- Fastest switching

**Cons**: 
- More complex implementation
- Cross-origin restrictions

### Solution 4: Lazy Loading with Progressive Enhancement
**Approach**: Show static preview first, load interactive on demand
```javascript
// Show static image first
<img src="dashboard_preview.png" />

// Load Plotly on user interaction
button.onclick = () => loadInteractiveDashboard();
```
**Pros**: 
- Instant visual feedback
- Load only what's needed

**Cons**: 
- Requires preview generation
- Two-step interaction

### Solution 5: Use Lighter Visualization Library
**Approach**: Replace Plotly with lighter alternatives
- Chart.js (~60KB)
- D3.js (~280KB)
- ApexCharts (~450KB)

**Pros**: 
- Much smaller file sizes
- Faster parsing

**Cons**: 
- Complete rewrite needed
- Less features

## Recommended Implementation

### Immediate Fix (Quick Win)
Change both `viz_plotly.py` and `viz_plotly_enhanced.py`:

```python
# FROM:
include_plotlyjs='cdn'

# TO:
include_plotlyjs='directory'  # or 'inline' for standalone files
```

### Enhanced Solution
1. Use 'directory' mode for shared Plotly.js
2. Add loading states in index.html
3. Implement iframe preloading for next likely user
4. Add caching headers for local files

## Performance Improvements Expected

### Current Performance
- Initial load: 2-3 seconds
- Subsequent switches: 1-2 seconds
- Network dependent

### After Optimization
- Initial load: 0.5-1 second
- Subsequent switches: <0.2 seconds
- No network dependency

## Additional Optimizations

### 1. Reduce Plot Complexity
- Limit data points shown initially
- Use sampling for large datasets
- Implement zoom-to-detail

### 2. Progressive Loading
- Load main plot first
- Load supplementary plots on scroll
- Defer table rendering

### 3. Dashboard Simplification
- Reduce from 10 plots to 5-6 essential
- Move detailed analytics to separate view
- Use tabs for different views

### 4. Caching Strategy
```javascript
// Preload next/previous user dashboards
function preloadAdjacent(currentIndex) {
    const prev = users[currentIndex - 1];
    const next = users[currentIndex + 1];
    
    if (prev) preloadDashboard(prev);
    if (next) preloadDashboard(next);
}
```

## Implementation Priority

1. **Immediate** - Change to 'directory' or 'inline' mode
2. **Short-term** - Add loading states and progress indicators
3. **Medium-term** - Implement iframe preloading
4. **Long-term** - Consider lighter visualization options