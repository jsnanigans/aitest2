# Plan: Improve Rejection Colors and Marker Visibility

## Summary
Update the visualization color scheme for rejected values to use a red-orange-black gradient, add white outlines to accepted values for better overlap handling, and refine marker sizes for visual consistency.

## Context
- Source: User request for specific visual improvements
- Current state: Rejection categories use various colors that don't follow a cohesive scheme
- Accepted values can overlap and become hard to distinguish
- Different marker types have inconsistent visual weights making some appear more prominent
- Need for a unified red-orange-black color scheme for rejections

## Requirements

### Functional
1. Implement red-orange-black color gradient for all rejection categories
2. Add white outlines to accepted values for better overlap visibility
3. Maintain colored outlines for rejected values using the new scheme
4. Standardize marker sizes across different types for visual consistency
5. Preserve source-specific marker shapes

### Non-functional
1. Improve visual clarity when points overlap
2. Maintain a cohesive color scheme
3. Professional appearance suitable for medical data
4. Performance unchanged
5. Visual weight balance across all marker types

## Current Issues Analysis

Looking at the visualization:
- **Color Scheme**: Current rejection colors don't follow a cohesive gradient
- **Overlapping Points**: Accepted values can overlap and become indistinguishable
- **Marker Size Inconsistency**: Different marker shapes (circles, squares, triangles) appear to have different visual weights
- **Visual Hierarchy**: Need clearer distinction between accepted and rejected values

## Proposed Color Scheme

### Design Principles
1. **Red-Orange-Black Gradient**: Use a cohesive gradient from red through orange to black
2. **Severity-Based Mapping**: Most severe issues use darker/redder colors
3. **Visual Consistency**: All rejection colors follow the same palette
4. **Clear Hierarchy**: Darker = more severe, lighter = less severe

### New Rejection Color Map

```python
def get_rejection_color_map():
    """Red-orange-black gradient for rejection categories."""
    return {
        # Most Severe (Dark Red to Black)
        "BMI Value": "#1A0000",        # Near black - BMI detected
        "Unit Convert": "#330000",     # Very dark red - unit conversion
        "Physio Limit": "#4D0000",     # Dark red - physiological bounds
        
        # High Severity (Dark Red)
        "Extreme Dev": "#660000",      # Dark red - extreme deviation
        "Out of Bounds": "#800000",    # Maroon - statistical bounds
        
        # Medium Severity (Red to Red-Orange)
        "High Variance": "#990000",    # Medium red - variance issues
        "Sustained": "#B30000",        # Bright red - sustained change
        "Limit Exceed": "#CC0000",     # Pure red - limit exceeded
        
        # Lower Severity (Orange-Red to Orange)
        "Daily Flux": "#E63300",       # Red-orange - daily fluctuation
        "Medium Term": "#FF6600",      # Orange - medium-term
        "Short Term": "#FF9933",       # Light orange - short-term
        
        # Unknown
        "Other": "#333333"             # Dark grey - uncategorized
    }
```

## Critical Addition: Consistent Source Markers Across All Users

### Problem
Currently, source markers (shapes and colors) are assigned based on frequency within each user's data. This means:
- The same source (e.g., "patient-device-scale-ABC123") gets different markers for different users
- Users cannot compare visualizations meaningfully
- Source identification becomes inconsistent across the system

### Solution: Global Source Registry

#### 1. Create Persistent Source Mapping
```python
def get_global_source_registry():
    """Get consistent source->marker mapping across all users."""
    # Define canonical source order (most reliable/common first)
    canonical_sources = [
        'patient-device',           # Most reliable
        'internal-questionnaire',   # Common manual entry
        'connectivehealth',         # API sources
        'api.iglucose.com',        
        'patient-upload',          # Manual uploads
        # ... extend as needed
    ]
    
    # Assign markers and colors consistently
    markers = generate_marker_sequence()
    colors = generate_color_palette(len(canonical_sources))
    
    registry = {}
    for i, source_pattern in enumerate(canonical_sources):
        registry[source_pattern] = {
            'marker': markers[i % len(markers)],
            'color': colors[i],
            'priority': i
        }
    
    return registry
```

#### 2. Source Matching Strategy
```python
def match_source_to_canonical(source: str, global_registry: dict):
    """Match a source string to its canonical representation."""
    source_lower = source.lower()
    
    # Try exact match first
    if source in global_registry:
        return global_registry[source]
    
    # Try pattern matching
    for pattern, style in global_registry.items():
        if pattern in source_lower:
            return style
    
    # Default for unknown sources
    return {
        'marker': '.',
        'color': '#9E9E9E',
        'priority': 999
    }
```

#### 3. Implementation Locations
- **create_source_registry()**: Modify to use global registry as base
- **All chart functions**: Use consistent source registry
- **Legend generation**: Sort by global priority, not local frequency

### Benefits
1. **Cross-User Consistency**: Same source always has same visual representation
2. **Learning Effect**: Users learn to recognize sources by shape/color
3. **Comparative Analysis**: Can compare charts between users meaningfully
4. **System-Wide Standards**: Establishes visual language for data sources

### Migration Path
1. Analyze all existing sources in database to build canonical list
2. Create configuration file with source mappings
3. Update visualization.py to load and use global registry
4. Add override capability for custom sources
5. Document standard source visual representations

## Implementation Changes

### 1. Update `get_rejection_color_map()` function (lines 50-76)
- Replace current color map with red-orange-black gradient
- Order by severity (darkest/reddest for most severe)
- Ensure smooth gradient progression

### 2. Add White Outlines to Accepted Values (lines 559-570)
```python
# Add white outline for accepted values to handle overlaps
ax1.scatter(
    source_groups[source]['timestamps'],
    source_groups[source]['weights'],
    marker=style['marker'],
    s=style['size'],
    alpha=0.9,  # Keep high alpha for clarity
    color=style['color'],
    label=label,
    zorder=5,
    edgecolors='white',  # White outline for overlap visibility
    linewidth=0.5  # Subtle outline
)
```

### 3. Maintain Rejected Value Outlines (lines 626-635)
```python
# Keep colored outlines for rejected values
ax1.scatter(
    [ts], [w],
    marker=style['marker'],
    s=style['size'],
    alpha=0.7,  # Semi-transparent fill
    color=style['color'],
    edgecolors=edge_color,  # Red-orange-black gradient color
    linewidth=2.5,  # Visible but not overwhelming
    zorder=4
)
```

### 4. Standardize Marker Sizes
Create a marker size normalization system to ensure visual consistency:

```python
def get_normalized_marker_size(marker_type: str, base_size: int = 60) -> int:
    """Get normalized size for different marker types to ensure visual consistency."""
    # Visual weight compensation factors
    size_factors = {
        'o': 1.0,   # circle - baseline
        's': 0.9,   # square - appears larger
        '^': 1.1,   # triangle up - appears smaller
        'v': 1.1,   # triangle down
        'D': 0.95,  # diamond - appears larger
        'p': 1.05,  # pentagon
        'h': 1.0,   # hexagon
        '*': 1.2,   # star - appears smaller
        'P': 1.1,   # plus
        'X': 1.1,   # x
        'd': 1.15,  # thin diamond - appears smaller
        '<': 1.1,   # triangle left
        '>': 1.1,   # triangle right
        '8': 1.0,   # octagon
        'H': 1.0,   # hexagon rotated
        '.': 1.5,   # point - needs to be much larger
    }
    
    factor = size_factors.get(marker_type, 1.0)
    return int(base_size * factor)
```

Then update the source registry creation (lines 276-283):
```python
registry[source] = {
    'id': f'source_{priority:03d}',
    'marker': markers[marker_idx],
    'color': selected_color,
    'size': get_normalized_marker_size(markers[marker_idx], 60),  # Normalized size
    'label': truncate_source_label(source),
    'frequency': count,
    'priority': priority,
    'is_primary': True
}
```

### 5. Update Legend Organization
- Maintain existing legend structure
- Update rejection category colors to show new gradient
- Ensure legend reflects white outlines for accepted values

### 6. Visual Hierarchy Summary
- Accepted values: Solid fill, white outline (0.5 width), high opacity (0.9)
- Rejected values: Semi-transparent fill, red-orange-black outline (2.5 width), medium opacity (0.7)
- Source shapes preserved for both
- Marker sizes normalized for visual consistency

## Alternative Approaches Considered

### Option A: Simple Monochrome Gradient
- Use only black to grey gradient for rejections
- Pros: Maximum simplicity
- Cons: Less information conveyed, harder to distinguish categories

### Option B: Full Spectrum Colors (Current)
- Keep existing diverse color palette
- Pros: Maximum distinction between categories
- Cons: Not cohesive, doesn't follow user's requirement

### Option C: Red-Orange-Black Gradient (Chosen)
- Severity-based gradient from red through orange to black
- Pros: Cohesive scheme, intuitive severity mapping, meets requirements
- Cons: Slightly less distinction between adjacent categories

## Testing & Validation

### Visual Tests
1. Generate chart with overlapping accepted values to test white outline effectiveness
2. Create chart with all rejection categories to verify gradient progression
3. Test different marker types at normalized sizes
4. Verify visual weight consistency across all markers
5. Check legend readability with new colors

### Validation Points
1. White outlines improve overlap visibility without adding clutter
2. Red-orange-black gradient is intuitive and cohesive
3. All marker types appear similar in visual prominence
4. Rejection severity is clearly communicated through color darkness

## Rollout Plan
1. Update `get_rejection_color_map()` with red-orange-black gradient
2. Add white outlines to accepted value scatter plots
3. Implement marker size normalization function
4. Update source registry to use normalized sizes
5. Test with sample data containing overlapping points
6. Verify all rejection categories display correctly
7. Fine-tune outline widths and alpha values if needed

## Success Criteria
- Red-orange-black gradient clearly shows rejection severity
- White outlines make overlapping accepted values distinguishable
- All marker types have similar visual weight
- Legend accurately reflects new color scheme
- No performance degradation
- Visualization remains clean and professional

## Risk Mitigation
- Test with dense data to ensure white outlines don't create clutter
- Verify gradient colors are distinguishable on different monitors
- Keep marker size normalization factors adjustable
- Document all color hex codes and size factors
- Test with actual user data before full deployment