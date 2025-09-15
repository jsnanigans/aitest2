# Reset Visualization Implementation Guide

## Quick Implementation Plan

### Phase 1: Basic Reset Markers (Quick Win)
Add vertical lines for each reset type with different styles.

### Phase 2: Adaptation Regions
Add shaded regions showing adaptation periods.

### Phase 3: Interactive Details
Add hover tooltips with reset information.

### Phase 4: Gap Visualization
Show data gaps that triggered hard resets.

## Code Implementation

### 1. Update `src/visualization.py`

#### Add Reset Data Extraction
```python
def extract_reset_insights(results):
    """Extract reset events and adaptation periods from results."""
    reset_events = []
    adaptation_tracking = {}
    
    for i, r in enumerate(results):
        if r.get('reset_event'):
            reset = r['reset_event']
            reset_events.append({
                'index': i,
                'timestamp': r['timestamp'],
                'type': reset.get('type', 'unknown'),
                'reason': reset.get('reason', ''),
                'gap_days': reset.get('gap_days'),
                'weight': r.get('raw_weight'),
                'previous_weight': None  # Will backfill
            })
        
        # Track adaptation progress
        if 'measurements_since_reset' in r:
            adaptation_tracking[i] = {
                'count': r['measurements_since_reset'],
                'in_adaptation': r.get('in_adaptation', False)
            }
    
    return reset_events, adaptation_tracking
```

#### Add Reset Visualization Function
```python
def add_reset_insights(fig, results, config):
    """Add reset markers and adaptation regions to the plot."""
    
    reset_events, adaptation = extract_reset_insights(results)
    
    if not reset_events:
        return
    
    # Color scheme for reset types
    reset_styles = {
        'hard': {'color': '#CC0000', 'dash': 'solid', 'width': 2, 'name': 'Hard Reset'},
        'initial': {'color': '#00AA00', 'dash': 'solid', 'width': 2, 'name': 'Initial'},
        'soft': {'color': '#0066CC', 'dash': 'dash', 'width': 2, 'name': 'Soft Reset'}
    }
    
    # Add vertical lines for each reset
    for reset in reset_events:
        style = reset_styles.get(reset['type'], reset_styles['hard'])
        
        # Add vertical line
        fig.add_vline(
            x=reset['timestamp'],
            line_color=style['color'],
            line_width=style['width'],
            line_dash=style['dash'],
            annotation_text=f"{style['name']}",
            annotation_position="top"
        )
        
        # Add hover text
        hover_text = format_reset_hover(reset)
        fig.add_annotation(
            x=reset['timestamp'],
            y=1.02,  # Above plot
            text=get_reset_symbol(reset['type']),
            showarrow=False,
            hovertext=hover_text,
            hoverlabel=dict(
                bgcolor=style['color'],
                font_color="white"
            )
        )
```

#### Add Adaptation Region Visualization
```python
def add_adaptation_regions(fig, results):
    """Add shaded regions for adaptation periods."""
    
    current_adaptation = None
    adaptation_regions = []
    
    for r in results:
        if r.get('reset_event'):
            # Start new adaptation period
            current_adaptation = {
                'start': r['timestamp'],
                'type': r['reset_event']['type'],
                'measurements': []
            }
        
        if current_adaptation:
            current_adaptation['measurements'].append(r)
            
            # Check if adaptation ended
            measurements_since = r.get('measurements_since_reset', 0)
            params = r.get('reset_parameters', {})
            warmup = params.get('warmup_measurements', 10)
            
            if measurements_since >= warmup:
                current_adaptation['end'] = r['timestamp']
                adaptation_regions.append(current_adaptation)
                current_adaptation = None
    
    # Add regions to plot
    for region in adaptation_regions:
        add_shaded_adaptation(fig, region)
```

### 2. Helper Functions

#### Format Reset Information
```python
def format_reset_hover(reset):
    """Create hover text for reset event."""
    lines = [
        f"<b>{reset['type'].title()} Reset</b>",
        f"Time: {reset['timestamp'].strftime('%Y-%m-%d %H:%M')}"
    ]
    
    if reset['type'] == 'hard':
        lines.append(f"Gap: {reset['gap_days']:.0f} days")
    elif reset['type'] == 'soft':
        if reset['previous_weight']:
            change = reset['weight'] - reset['previous_weight']
            lines.append(f"Weight change: {change:+.1f} kg")
    
    lines.append(f"Reason: {reset['reason']}")
    
    return "<br>".join(lines)

def get_reset_symbol(reset_type):
    """Get symbol for reset type."""
    symbols = {
        'hard': '⟲',     # Reset arrow
        'initial': '▶',  # Start arrow
        'soft': '↻'      # Circular arrow
    }
    return symbols.get(reset_type, '|')
```

#### Add Progress Indicator
```python
def add_adaptation_progress_bar(fig, x_position, progress, reset_type):
    """Add a small progress bar showing adaptation status."""
    colors = {
        'hard': '#CC0000',
        'initial': '#00AA00',
        'soft': '#0066CC'
    }
    
    # Add as a small horizontal bar chart
    fig.add_trace(go.Bar(
        x=[progress * 100],
        y=['Adaptation'],
        orientation='h',
        marker_color=colors.get(reset_type, '#999'),
        opacity=0.6,
        showlegend=False,
        hovertemplate=f"Adaptation: {progress*100:.0f}%<extra></extra>"
    ))
```

### 3. Integration with Main Plot

#### Update `create_weight_timeline` function
```python
def create_weight_timeline(results, user_id, config):
    """Create enhanced timeline with reset insights."""
    
    # ... existing code ...
    
    # Add reset visualizations if enabled
    reset_config = config.get('visualization', {}).get('reset_insights', {})
    if reset_config.get('show_reset_lines', True):
        add_reset_insights(fig, results, config)
    
    if reset_config.get('show_adaptation_regions', True):
        add_adaptation_regions(fig, results)
    
    if reset_config.get('show_gap_regions', True):
        add_gap_regions(fig, results)
    
    # Add legend for reset types
    if reset_config.get('show_in_legend', True):
        add_reset_legend(fig)
    
    # ... rest of existing code ...
```

### 4. Visual Styling

#### CSS-like Styling for Regions
```python
ADAPTATION_STYLES = {
    'hard': {
        'fillcolor': 'rgba(255, 200, 200, 0.2)',  # Light red
        'pattern': {'shape': '/', 'size': 4},      # Diagonal lines
        'border': {'color': '#CC0000', 'width': 1}
    },
    'soft': {
        'fillcolor': 'rgba(200, 220, 255, 0.15)',  # Light blue
        'pattern': None,                            # Solid fill
        'border': {'color': '#0066CC', 'width': 1, 'dash': 'dot'}
    },
    'initial': {
        'fillcolor': 'rgba(200, 255, 200, 0.15)',  # Light green
        'pattern': None,
        'border': {'color': '#00AA00', 'width': 1}
    }
}
```

## Testing the Visualization

### Test Script
```python
def test_reset_visualization():
    """Test reset visualization with sample data."""
    
    # Create sample results with different reset types
    results = [
        {'timestamp': datetime(2024, 1, 1), 'raw_weight': 85, 
         'reset_event': {'type': 'initial', 'reason': 'first_measurement'}},
        
        # ... normal measurements ...
        
        {'timestamp': datetime(2024, 2, 1), 'raw_weight': 90,
         'reset_event': {'type': 'hard', 'gap_days': 30, 'reason': 'gap_exceeded'}},
        
        # ... more measurements ...
        
        {'timestamp': datetime(2024, 2, 15), 'raw_weight': 95,
         'reset_event': {'type': 'soft', 'reason': 'manual_entry_5kg'}}
    ]
    
    fig = create_weight_timeline(results, 'test_user', config)
    fig.show()
```

## Expected Output

### Visual Hierarchy
1. **Background**: Adaptation regions (subtle shading)
2. **Middle**: Data points and trend lines
3. **Foreground**: Reset markers (vertical lines)
4. **Top**: Annotations and symbols

### Information Density
- **At a glance**: See reset events as vertical lines
- **On hover**: Get detailed reset information
- **In legend**: Understand reset types
- **In regions**: See adaptation periods

## Performance Considerations
- Limit annotations if >10 resets to avoid clutter
- Use efficient shapes for regions (rectangles vs complex paths)
- Cache reset calculations if results don't change
- Consider pagination for users with many resets

## Configuration
Add to `config.toml`:
```toml
[visualization.reset_insights]
show_reset_lines = true
show_adaptation_regions = true
show_gap_regions = true
show_annotations = true
max_annotations = 10
adaptation_opacity = 0.2
```

## Success Criteria
✓ Reset events clearly visible
✓ Different reset types distinguishable
✓ Adaptation periods shown
✓ Hover information available
✓ Doesn't obscure main data
✓ Legend explains symbols