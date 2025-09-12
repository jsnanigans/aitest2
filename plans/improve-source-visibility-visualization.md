# Plan: Improve Source Visibility in Visualization

## Summary
Enhance the weight stream processor visualization to better display data source information for both accepted and rejected measurements, treating each unique source as distinct rather than grouping them into categories, while maintaining chart readability.

## Context
- Source: User request to improve source visibility in visualization.py
- Current state: Accepted values show grouped source categories; rejected values don't show source information at all
- Assumptions:
  - Each unique source string in the data should be treated as a distinct source
  - Visual clarity must be maintained despite potentially many unique sources
  - Both accepted and rejected measurements need source identification

## Requirements

### Functional
1. Display each unique source string as its own distinct source (no grouping)
2. Show source information for rejected measurements
3. Provide unique visual identification (icon/marker/color) for each source
4. Maintain chart readability with potentially many sources
5. Clear visual distinction between accepted/rejected status while showing source

### Non-functional
1. Performance: Handle datasets with 10-100 unique sources efficiently
2. Accessibility: Ensure color choices are distinguishable
3. Scalability: Design must work with varying numbers of sources
4. Maintainability: Clean separation of source styling logic

## Alternatives

### Option A: Dual-Encoding System (Shape + Color)
- Approach: Use marker shapes for sources, colors for accept/reject status
- Pros:
  - Clear separation of concerns (source vs status)
  - Works well with colorblind users
  - Scalable to many sources
- Cons:
  - Limited distinct shapes available (~15-20)
  - May need fallback for excess sources
- Risks: Shape similarity confusion with many sources

### Option B: Color-Coded Sources with Status Indicators
- Approach: Unique color per source, use solid/hollow markers for accept/reject
- Pros:
  - Unlimited color variations possible
  - Visually intuitive status indication
  - Preserves current color associations
- Cons:
  - Color similarity issues with many sources
  - Harder for colorblind users
- Risks: Color palette exhaustion

### Option C: Interactive Legend with Dynamic Highlighting
- Approach: Base visualization with interactive legend for source filtering
- Pros:
  - Handles unlimited sources elegantly
  - User-controlled complexity
  - Best for exploration
- Cons:
  - Requires interactive capabilities (not static PNG)
  - More complex implementation
- Risks: Loss of at-a-glance insights

## Recommendation
**Option A: Dual-Encoding System** with intelligent fallbacks

Rationale:
- Best balance of clarity and scalability
- Accessibility-friendly
- Works within current static PNG output constraints
- Can be enhanced later with interactivity

## High-Level Design

### Architecture Changes
```
visualization.py modifications:
├── Source Management
│   ├── get_unique_sources() - Extract all unique source strings
│   ├── create_source_registry() - Map sources to visual properties
│   └── get_source_style_v2() - New style system for unique sources
├── Marker System
│   ├── generate_marker_sequence() - Ordered list of distinct markers
│   ├── assign_source_markers() - Map sources to markers by frequency
│   └── get_fallback_style() - Handle overflow sources
├── Rejection Visualization
│   ├── plot_rejected_with_sources() - New rejection plotting
│   └── create_rejection_legend() - Combined source/status legend
└── Legend Management
    ├── create_hierarchical_legend() - Two-tier legend system
    └── optimize_legend_layout() - Smart positioning
```

### Visual Encoding Strategy
1. **Primary Sources (top 10-15 by frequency)**
   - Unique marker shapes
   - Consistent colors per source
   - Size variation for emphasis

2. **Secondary Sources**
   - Grouped as "Other Sources" with subdivisions
   - Smaller markers
   - Muted colors

3. **Status Indication**
   - Accepted: Filled markers with edge
   - Rejected: X overlay on source marker
   - Partial rejection: Hollow markers

### Data Model Changes
```python
source_registry = {
    'source_string': {
        'id': 'source_001',
        'marker': 'o',
        'color': '#2E7D32',
        'size': 60,
        'label': 'Shortened Label',
        'frequency': 150,
        'priority': 1
    }
}
```

## Implementation Plan (No Code)

### Phase 1: Source Discovery and Registry
1. Analyze existing data flow for source extraction
2. Create source frequency analysis
3. Implement source registry with deduplication
4. Add source string normalization (preserve originals)

### Phase 2: Marker System Development
1. Define marker sequence (15-20 distinct shapes)
2. Create marker assignment algorithm (frequency-based)
3. Implement size scaling based on importance
4. Add fallback grouping for overflow sources

### Phase 3: Accepted Values Enhancement
1. Modify current scatter plot logic
2. Replace grouped sources with unique source plotting
3. Update legend generation for individual sources
4. Add source frequency to labels

### Phase 4: Rejected Values Visualization
1. Create new plotting function for rejected with sources
2. Implement dual-encoding (source marker + rejection indicator)
3. Add rejection reason as secondary information
4. Integrate with existing rejection clustering

### Phase 5: Legend Optimization
1. Implement two-tier legend (sources + statuses)
2. Add source sorting by frequency/priority
3. Create compact legend layout algorithm
4. Add "show more" capability for many sources

### Phase 6: Visual Polish
1. Optimize color palette for distinctiveness
2. Add subtle visual hierarchies
3. Implement smart label truncation
4. Add source tooltips (as text annotations)

## Validation & Rollout

### Test Strategy
1. Unit tests for source registry and marker assignment
2. Visual regression tests with sample datasets
3. Performance tests with 100+ unique sources
4. Accessibility validation (color contrast, shapes)

### Test Cases
- Single source dataset
- 5-10 sources with varying frequencies
- 50+ unique sources
- Mixed accept/reject ratios per source
- Sources with special characters/long names

### Manual QA Checklist
- [ ] All unique sources visible in legend
- [ ] Clear distinction between accepted/rejected
- [ ] No marker shape collisions
- [ ] Legend remains readable
- [ ] Chart not overcrowded
- [ ] Source labels properly truncated

### Rollout Plan
1. Feature flag: `enable_unique_source_viz`
2. A/B test with subset of users
3. Gradual rollout by user cohort
4. Monitor rendering performance
5. Collect user feedback on clarity

## Risks & Mitigations

### Risk 1: Visual Overload
- **Risk**: Too many unique sources cluttering chart
- **Mitigation**: Implement smart clustering for low-frequency sources
- **Monitoring**: Track unique source counts per user

### Risk 2: Performance Degradation
- **Risk**: Slow rendering with many unique markers
- **Mitigation**: Implement marker caching and reuse
- **Monitoring**: Track visualization generation time

### Risk 3: Legend Space Exhaustion
- **Risk**: Legend too large for allocated space
- **Mitigation**: Scrollable or paginated legend sections
- **Monitoring**: Legend size vs panel size ratio

## Acceptance Criteria

1. ✓ Each unique source string has distinct visual representation
2. ✓ Rejected measurements show source information
3. ✓ Maximum 15 primary sources shown individually
4. ✓ Overflow sources grouped intelligently
5. ✓ Legend shows source frequency/count
6. ✓ Visual distinction between accepted/rejected maintained
7. ✓ Chart remains readable with 50+ sources
8. ✓ Source labels truncated appropriately
9. ✓ Color palette is accessible
10. ✓ Performance impact < 10% for typical datasets

## Out of Scope

- Interactive filtering/toggling of sources
- Source metadata beyond the string value
- Historical source evolution tracking
- Source reliability scoring
- Custom source icons/images
- Source grouping by user preference
- Export of source-specific data

## Open Questions

1. Should we preserve any source categorization for backwards compatibility?
2. What's the maximum number of unique sources we should support individually?
3. Should source colors be consistent across different users?
4. How should we handle source strings that differ only slightly (e.g., typos)?
5. Should rejected sources appear in the main legend or a separate section?

## Review Cycle

### Self-Review Notes
- Considered marker shape limitations carefully
- Prioritized accessibility in design choices
- Balanced detail vs. clarity trade-offs
- Ensured backwards compatibility path

### Revisions
- Added fallback grouping for overflow sources
- Included frequency information in legend
- Specified truncation strategy for long names
- Added performance monitoring points