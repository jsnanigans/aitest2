# Investigation: Multi-threading Performance Issue in Visualization Generation

## Bottom Line
**Root Cause**: Python's GIL prevents parallel execution of CPU-bound Plotly HTML generation
**Fix Location**: `main.py:550-586` - ThreadPoolExecutor implementation
**Confidence**: High

## What's Happening
ThreadPoolExecutor with 8 threads shows no performance improvement over single-threaded execution when generating 500 visualizations. All complete successfully but parallelism provides no speedup.

## Why It Happens
**Primary Cause**: Plotly's `write_html()` is CPU-bound (JSON serialization, HTML generation)
**Trigger**: `src/viz/visualization.py:302` - Each call does heavy computation
**Decision Point**: `main.py:551` - ThreadPoolExecutor cannot bypass GIL for CPU work

## Evidence
- **Key File**: `src/viz/visualization.py:132-302` - Creates 17+ traces per visualization
- **Search Used**: `rg "write_html"` - Single CPU-intensive operation per viz
- **Test Result**: Threading gives 1.04x speedup vs expected 4x (GIL blocking)
- **Plotly Profile**: 0.02s per write, all CPU-bound JSON/HTML generation

## Next Steps
1. Replace ThreadPoolExecutor with ProcessPoolExecutor for true parallelism
2. Move matplotlib backend setting out of thread function (line 86-87)
3. Consider async I/O if file writes become bottleneck after CPU fix

## Risks
- Current threading adds overhead without performance benefit
- 500 visualizations take same time as single-threaded but with more complexity
