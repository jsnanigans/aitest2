# Plan: Visualize Reset Insights in Dashboard

## Summary
Add comprehensive reset visualization to help users understand when resets occur, why they happen, and how the system is adapting.

## Visual Requirements

### 1. Reset Event Markers
Show different reset types with distinct visual indicators:
- **Hard Reset** (30+ day gap): Bold red vertical line
- **Initial Reset** (first measurement): Green vertical line
- **Soft Reset** (manual entry): Blue dashed vertical line

### 2. Adaptation Period Visualization
Show when system is in adaptive mode:
- Shaded background region during adaptation
- Gradient opacity (darker at reset, fading over time)
- Different colors per reset type

### 3. Reset Information Panel
Hoverable/clickable details showing:
- Reset type and trigger reason
- Adaptation parameters (boost factors, decay rate)
- Progress through adaptation (e.g., "Day 3 of 7")
- Measurements since reset counter

### 4. Gap Visualization
For hard resets, show the gap period:
- Gray shaded region with diagonal stripes
- Text overlay: "No data - 35 days"
- Connect last pre-gap to first post-gap with dotted line

## Implementation Design

### Visual Components

```
Timeline View:
     
     Hard Reset (Gap)          Soft Reset           
          ↓                        ↓               
    ━━━━━|════════════━━━━━━━━━━━|┅┅┅┅┅━━━━━━━
    ←gap→|←─adaptation─→          |←adapt→       
    ░░░░░|▓▓▓▓▓▓▒▒▒▒░░░          |▒▒▒░░░       
         |                        |               
    Last │ Reset                  │ Manual       
    85kg │ 120kg                  │ 92kg         
```

### Color Scheme
```python
RESET_COLORS = {
    'hard': {
        'line': '#CC0000',      # Dark red
        'adaptation': '#FFE5E5', # Light red
        'opacity': 0.3
    },
    'initial': {
        'line': '#00AA00',      # Green
        'adaptation': '#E5FFE5', # Light green
        'opacity': 0.25
    },
    'soft': {
        'line': '#0066CC',      # Blue
        'adaptation': '#E5F2FF', # Light blue
        'opacity': 0.2
    }
}
```

### Information Layers

#### Layer 1: Reset Markers
```python
# Vertical lines at reset points
for reset in reset_events:
    add_vline(
        x=reset['timestamp'],
        line_color=RESET_COLORS[reset['type']]['line'],
        line_width=2,
        line_dash='solid' if reset['type'] == 'hard' else 'dash'
    )
```

#### Layer 2: Adaptation Regions
```python
# Shaded regions during adaptation
for reset in reset_events:
    end_time = reset['timestamp'] + timedelta(days=reset['adaptive_days'])
    add_vrect(
        x0=reset['timestamp'],
        x1=end_time,
        fillcolor=RESET_COLORS[reset['type']]['adaptation'],
        opacity=calculate_opacity(reset, current_time),
        layer='below'
    )
```

#### Layer 3: Annotations
```python
# Text annotations for reset events
for reset in reset_events:
    add_annotation(
        x=reset['timestamp'],
        y=y_position,
        text=get_reset_label(reset),
        showarrow=True,
        arrowhead=2
    )
```

### Hover Information Template
```html
<b>{{reset_type}} Reset</b><br>
<b>Trigger:</b> {{trigger_reason}}<br>
<b>Weight Change:</b> {{weight_before}} → {{weight_after}} kg<br>
<br>
<b>Adaptation Status:</b><br>
• Day {{current_day}} of {{total_days}}<br>
• Measurements: {{measurements_since_reset}}<br>
• Progress: {{progress_bar}}<br>
<br>
<b>Current Parameters:</b><br>
• Weight flexibility: {{weight_boost}}x<br>
• Trend flexibility: {{trend_boost}}x<br>
• Quality threshold: {{quality_threshold}}<br>
```

### Legend Enhancement
```python
# Add reset types to legend
legend_items = [
    {'name': 'Hard Reset (Gap)', 'symbol': '━', 'color': '#CC0000'},
    {'name': 'Soft Reset (Manual)', 'symbol': '┅', 'color': '#0066CC'},
    {'name': 'Initial Reset', 'symbol': '━', 'color': '#00AA00'},
    {'name': 'Adaptation Period', 'symbol': '▒', 'color': '#FFE5E5'}
]
```

## Implementation Steps

### Step 1: Extend Visualization Data Structure
```python
def prepare_reset_data(results):
    """Extract and enrich reset information from results."""
    reset_events = []
    adaptation_periods = []
    
    for r in results:
        if r.get('reset_event'):
            event = r['reset_event']
            reset_events.append({
                'timestamp': r['timestamp'],
                'type': event['type'],
                'reason': event['reason'],
                'gap_days': event.get('gap_days'),
                'weight_before': event.get('last_weight'),
                'weight_after': r['raw_weight'],
                'parameters': event.get('parameters', {})
            })
            
            # Calculate adaptation period
            adaptation_periods.append({
                'start': r['timestamp'],
                'end': r['timestamp'] + timedelta(days=params['adaptive_days']),
                'type': event['type'],
                'measurements_count': 0
            })
    
    return reset_events, adaptation_periods
```

### Step 2: Add Visual Elements
```python
def add_reset_visualizations(fig, reset_events, adaptation_periods):
    """Add reset markers and adaptation regions to plot."""
    
    # Add adaptation regions first (background)
    for period in adaptation_periods:
        add_adaptation_region(fig, period)
    
    # Add gap regions for hard resets
    for event in reset_events:
        if event['type'] == 'hard' and event.get('gap_days'):
            add_gap_region(fig, event)
    
    # Add reset lines (foreground)
    for event in reset_events:
        add_reset_line(fig, event)
    
    # Add annotations
    for event in reset_events:
        add_reset_annotation(fig, event)
```

### Step 3: Interactive Features
```python
def create_reset_info_box(reset_event, current_state):
    """Create detailed info box for reset event."""
    return {
        'visible': False,  # Show on hover/click
        'x': reset_event['timestamp'],
        'y': 1.1,  # Above main plot
        'content': format_reset_details(reset_event, current_state),
        'bgcolor': 'white',
        'bordercolor': RESET_COLORS[reset_event['type']]['line']
    }
```

### Step 4: Progress Indicators
```python
def add_adaptation_progress(fig, state, timestamp):
    """Show current adaptation progress if active."""
    if state.get('reset_timestamp'):
        days_since = (timestamp - state['reset_timestamp']).days
        total_days = state['reset_parameters']['adaptive_days']
        
        if days_since < total_days:
            # Add progress bar or indicator
            add_progress_indicator(
                fig,
                progress=days_since/total_days,
                text=f"Adapting: Day {days_since}/{total_days}"
            )
```

## Visual Examples

### Example 1: Hard Reset After Gap
```
Before Gap        Gap Period        After Reset
    ●●●●●    ░░░░░░░░░░░░░░░░    |▓▓▓▓●●●●●
    105kg    [No Data - 35d]     |  107kg→
                                  ↑
                              Hard Reset
                              Gap: 35 days
```

### Example 2: Soft Reset from Manual Entry
```
Auto Measurements    Manual Entry    Adaptation
    ●●●●●●●●●●●        |┅┅┅┅┅●●●●●●●
    85kg→86kg          |   92kg→91kg
                       ↑
                   Soft Reset
                   Manual: +6kg
```

### Example 3: Multiple Resets
```
Initial    Normal     Gap/Hard    Manual/Soft
  |●●●●●●●●●●●●●  ░░░░|▓▓▓●●●●    |▒▒●●●●
  ↑                   ↑            ↑
Initial            Hard Reset   Soft Reset
Start              Gap: 30d     Manual: +5kg
```

## Configuration Options
```toml
[visualization.reset_insights]
# Enable/disable reset visualizations
show_reset_lines = true
show_adaptation_regions = true
show_gap_regions = true
show_annotations = true
show_progress = true

# Visual styling
reset_line_width = 2
adaptation_opacity = 0.2
gap_pattern = "diagonal"  # diagonal, dots, solid

# Information display
hover_details = true
click_for_details = true
show_in_legend = true

# Annotation settings
annotation_position = "top"  # top, bottom, auto
annotation_rotation = 0
max_annotations = 10  # Limit to avoid clutter
```

## Benefits
1. **Transparency**: Users see exactly when and why resets occur
2. **Understanding**: Clear visualization of adaptation periods
3. **Trust**: Shows system is responding appropriately to gaps/manual data
4. **Debugging**: Helps identify patterns in reset behavior
5. **Education**: Users learn how the system adapts

## Success Metrics
- Users can identify reset events at a glance
- Adaptation periods are clearly visible
- Reset reasons are easily accessible
- Visual doesn't clutter the main data
- Performance remains smooth with many resets

## Future Enhancements
- Animation showing adaptation decay over time
- Comparison view (with/without resets)
- Reset statistics panel
- Export reset history
- Predictive indicators for upcoming resets