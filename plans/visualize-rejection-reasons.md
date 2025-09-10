# Plan: Display Rejection Reasons in Visualization Without Clutter

## Summary
Add clear, non-intrusive visualization of rejection reasons to help users understand why certain measurements were rejected by the processor. The solution should provide detailed information on demand while maintaining clean, readable charts.

## Context
- Source: User request to show rejection reasons in visualization without cluttering
- Current state: Rejection reasons are collected but commented out in visualization (lines 388-391)
- Assumptions:
  - Users need to understand why measurements are rejected for debugging and trust
  - Visual clarity is paramount - too much information can overwhelm
  - Different users have different levels of interest in rejection details
- Constraints:
  - Must work within existing matplotlib-based visualization framework
  - Cannot use interactive elements (static PNG output only)
  - Must handle multiple rejection reasons efficiently

## Requirements

### Functional
- Display rejection reasons for all rejected measurements
- Group and summarize rejection reasons to avoid repetition
- Provide both overview and detailed views of rejections
- Maintain visual hierarchy - rejections are secondary to main data
- Support both full-range and cropped time views
- Handle edge cases (many rejections, long reason strings)

### Non-functional
- Performance: No significant impact on rendering time (<100ms added)
- Readability: Text must be legible at dashboard resolution (default 100 DPI)
- Accessibility: Consider colorblind users, use shapes + colors
- Scalability: Handle 0-1000+ rejections gracefully

## Alternatives

### Option A: Rejection Summary Panel + Annotated Points
- **Approach**: 
  - Add dedicated subplot showing rejection reason breakdown (pie/bar chart)
  - Use different marker shapes/colors for different rejection categories on main plot
  - Add hover-like annotations for most recent rejections
- **Pros**:
  - Clear visual separation of concerns
  - Easy to see patterns in rejection reasons
  - Scalable for many rejections
  - No clutter on main plots
- **Cons**:
  - Takes up subplot space
  - Static annotations can't show all details

### Option B: Color-Coded Rejection Markers with Legend
- **Approach**:
  - Use different colors/shapes for each rejection reason type
  - Comprehensive legend mapping colors to reasons
  - Size markers by frequency or recency
- **Pros**:
  - All information visible at once
  - No additional subplot needed
  - Clear visual patterns emerge
- **Cons**:
  - Limited to ~5-7 distinct reasons before confusion
  - Legend can become unwieldy
  - Colorblind accessibility issues

### Option C: Temporal Rejection Timeline
- **Approach**:
  - Add thin timeline bar below main plot
  - Color segments by rejection reason
  - Include text annotations for significant clusters
- **Pros**:
  - Temporal patterns clearly visible
  - Doesn't interfere with main data
  - Compact representation
- **Cons**:
  - Hard to read individual rejections
  - Requires additional vertical space
  - May not align well with irregular timestamps

### Option D: Smart Annotation Strategy
- **Approach**:
  - Annotate only "interesting" rejections (first of type, outliers, clusters)
  - Group nearby rejections into single annotation
  - Use callout boxes with reason summaries
  - Add rejection statistics to existing stats panel
- **Pros**:
  - Minimal visual clutter
  - Highlights important patterns
  - Works with existing layout
  - Progressive disclosure of information
- **Cons**:
  - Some rejections not explicitly shown
  - Requires smart clustering logic
  - May miss some patterns

## Recommendation
**Option D (Smart Annotation Strategy) + Enhanced Stats Panel** - This provides the best balance of information density and visual clarity. It follows the principle of progressive disclosure while ensuring all information is available.

## High-Level Design

### Architecture Overview
```
Rejection Processing Pipeline:
1. Extract rejection data from results
2. Categorize and cluster rejections
3. Identify "interesting" rejections for annotation
4. Generate visual elements
5. Update statistics panel
```

### Rejection Categories
Based on current processor implementation:
1. **Bounds Violations**: Weight outside [30, 400]kg range
2. **Physiological Limits**: Exceeds time-based change limits
   - Hydration/bathroom (<1h)
   - Meals+hydration (<6h)
   - Daily fluctuation (<24h)
   - Sustained change (>24h)
3. **Extreme Deviations**: Statistical outliers from Kalman prediction
4. **Session Variance**: Multiple users detected
5. **Basic Validation**: Other validation failures

### Visual Elements
1. **Main Plot Enhancements**:
   - Rejected points remain as red 'x' markers
   - Smart annotations for significant rejections
   - Subtle connecting lines to annotation boxes

2. **Annotation Rules**:
   - First rejection of each type
   - Clusters of 3+ rejections (show count + reason)
   - Rejections after long gaps
   - Extreme outliers (>100kg deviation)

3. **Statistics Panel Enhancement**:
   - Add "Rejection Analysis" section
   - Show breakdown by category with counts
   - Include percentage of total measurements
   - Highlight most common reason

4. **Visual Hierarchy**:
   - Primary: Filtered weight line
   - Secondary: Accepted raw measurements
   - Tertiary: Rejected measurements
   - Quaternary: Rejection annotations

### Affected Files
- `src/visualization.py`: Main implementation
- No processor changes needed (already provides reasons)

## Implementation Plan (No Code)

### Step 1: Create Rejection Analysis Functions
- Location: `src/visualization.py`, add new helper functions
- `categorize_rejection(reason_text)`: Map text to category
- `cluster_rejections(rejected_results)`: Group nearby rejections
- `identify_interesting_rejections()`: Select which to annotate
- `format_rejection_annotation()`: Create concise labels

### Step 2: Enhance Main Plot (ax1) Annotations
- Location: Lines 136-140 (rejected scatter plot)
- After plotting rejected points, add annotation logic
- Use matplotlib `annotate()` with bbox styling
- Position annotations to avoid overlapping
- Connect to points with subtle arrows

### Step 3: Enhance Cropped Plot Annotations
- Location: Lines 201-204 (cropped rejected scatter)
- Apply same annotation logic as main plot
- Filter to only rejections in cropped range
- Adjust annotation density for zoom level

### Step 4: Add Rejection Analysis to Stats Panel
- Location: Lines 383-391 (currently commented out)
- Uncomment and enhance rejection reasons display
- Group by category with counts
- Add percentage calculations
- Format as structured table

### Step 5: Implement Smart Clustering
- Temporal clustering: Group rejections within 1 day
- Reason clustering: Group identical reasons
- Spatial clustering: Avoid overlapping annotations
- Priority system: Most important rejections first

### Step 6: Add Visual Polish
- Consistent color scheme for rejection categories
- Subtle shadows/borders on annotation boxes
- Proper text wrapping for long reasons
- Alpha transparency for less important annotations

### Step 7: Handle Edge Cases
- No rejections: Skip rejection analysis
- Many rejections (>50): Show only top patterns
- Very long reason strings: Truncate with ellipsis
- Overlapping annotations: Smart repositioning

## Validation & Rollout

### Test Strategy
1. **Unit Tests**:
   - Test categorization function with all reason types
   - Test clustering algorithm with various patterns
   - Test annotation selection logic

2. **Integration Tests**:
   - Generate plots with known rejection patterns
   - Verify annotations appear correctly
   - Check stats panel calculations

3. **Visual Tests**:
   - User with no rejections
   - User with few rejections (<5)
   - User with many rejections (>50)
   - User with diverse rejection reasons
   - User with clustered rejections

### Manual QA Checklist
- [ ] Rejection annotations are readable
- [ ] Annotations don't overlap with data
- [ ] Color coding is consistent
- [ ] Stats panel shows accurate counts
- [ ] Cropped view shows appropriate subset
- [ ] Performance impact < 100ms
- [ ] PNG output quality maintained

### Rollout Plan
1. Implement helper functions
2. Test with user ending in 106kg (known rejections)
3. Add annotations to main plot
4. Extend to cropped plot
5. Enhance statistics panel
6. Test with full dataset
7. Document in README

## Risks & Mitigations

### Risk 1: Visual Clutter
- **Risk**: Too many annotations make plot unreadable
- **Mitigation**: Limit to 5-7 annotations per plot, summarize rest in stats

### Risk 2: Overlapping Annotations
- **Risk**: Annotations overlap each other or data
- **Mitigation**: Implement collision detection and smart positioning

### Risk 3: Performance Impact
- **Risk**: Clustering and annotation slow down rendering
- **Mitigation**: Pre-calculate during data processing, cache results

### Risk 4: Misleading Patterns
- **Risk**: Visual emphasis might create false patterns
- **Mitigation**: Use neutral colors, focus on factual presentation

## Acceptance Criteria
- [ ] All rejection reasons are accessible in visualization
- [ ] Main plots remain clean and readable
- [ ] Rejection patterns are identifiable
- [ ] Statistics accurately summarize rejections
- [ ] Annotations don't overlap significantly
- [ ] Performance impact < 100ms
- [ ] Works with all test users

## Out of Scope
- Interactive tooltips (static PNG only)
- Filtering by rejection reason
- Rejection reason configuration
- Real-time updates
- Detailed rejection explanations
- Rejection recovery suggestions

## Open Questions
1. Should we use icons/symbols for rejection categories?
2. What's the maximum number of annotations to show?
3. Should annotation verbosity be configurable?
4. Should we show rejection trends over time?
5. Do we need different annotation styles for different severities?
6. Should the legend include rejection categories?

## Review Cycle

### Self-Review Notes
- Focused on progressive disclosure principle
- Maintained existing visual hierarchy
- Kept changes localized to visualization.py
- Considered accessibility throughout
- Balanced information density with clarity

### Council Review

**Don Norman** (User Experience): "Progressive disclosure is exactly right here. Show the pattern first, details on demand. The smart annotation approach prevents information overload while ensuring users can understand what's happening. Consider using visual metaphors - perhaps rejection severity could map to annotation opacity?"

**Edward Tufte** (Data Visualization): "The data-ink ratio must be preserved. Every pixel should earn its place. Clustering nearby rejections is essential - show the pattern, not the noise. The statistics panel is the right place for detailed breakdowns."

**Butler Lampson** (Simplicity): "Don't overthink this. Red X's for rejections, a few key annotations, and a summary table. That's all you need. The smart annotation strategy is good but keep the 'smart' part simple - first occurrence and clusters, nothing fancier."

**Ward Cunningham** (Documentation): "The visualization IS documentation. Make sure the rejection reasons are self-explanatory. 'Exceeds 24h limit' is better than 'Physiological constraint violated'. Users shouldn't need a manual to understand what they're seeing."

**Kent Beck** (Testing): "Test with your worst case - a user with hundreds of rejections. If it stays readable there, you've succeeded. Also test with zero rejections to ensure graceful degradation."

### Revision Notes
After council review:
- Simplified annotation rules to just "first" and "clusters"
- Emphasized self-explanatory reason text
- Added consideration for visual metaphors (opacity for severity)
- Confirmed focus on data-ink ratio