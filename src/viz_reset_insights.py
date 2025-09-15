"""
Reset Insights Visualization Module
Provides simple, clean visualization of reset events and adaptation periods.
Designed to work with the basic visualization, not the enhanced one.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import plotly.graph_objects as go

# Color schemes for different reset types
RESET_COLORS = {
    'hard': {
        'line': '#CC0000',       # Dark red
        'adaptation': 'rgba(255, 200, 200, 0.2)',  # Light red with transparency
        'symbol': '⟲',
        'name': 'Hard Reset',
        'dash': 'solid',
        'width': 3
    },
    'initial': {
        'line': '#00AA00',       # Green
        'adaptation': 'rgba(200, 255, 200, 0.15)',  # Light green
        'symbol': '▶',
        'name': 'Initial',
        'dash': 'solid',
        'width': 2
    },
    'soft': {
        'line': '#0066CC',       # Blue
        'adaptation': 'rgba(200, 220, 255, 0.15)',  # Light blue
        'symbol': '↻',
        'name': 'Soft Reset',
        'dash': 'dash',
        'width': 2
    },
    'unknown': {
        'line': '#666666',       # Gray
        'adaptation': 'rgba(200, 200, 200, 0.1)',
        'symbol': '|',
        'name': 'Reset',
        'dash': 'dot',
        'width': 1
    }
}


def extract_reset_insights(results: List[Dict[str, Any]]) -> Tuple[List[Dict], List[Dict], Dict]:
    """
    Extract reset events, adaptation periods, and gaps from results.
    
    Returns:
        Tuple of (reset_events, adaptation_periods, gap_regions)
    """
    reset_events = []
    adaptation_periods = []
    gap_regions = []
    
    last_timestamp = None
    last_weight = None
    current_adaptation = None
    
    for i, r in enumerate(results):
        timestamp = r.get('timestamp')
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        
        # Check for reset event
        if r.get('reset_event'):
            reset = r['reset_event']
            reset_type = reset.get('type', 'unknown')
            
            # Record reset event
            reset_event = {
                'index': i,
                'timestamp': timestamp,
                'type': reset_type,
                'reason': reset.get('reason', ''),
                'gap_days': reset.get('gap_days'),
                'weight': r.get('raw_weight'),
                'previous_weight': last_weight,
                'previous_timestamp': last_timestamp
            }
            reset_events.append(reset_event)
            
            # Start new adaptation period
            if current_adaptation:
                adaptation_periods.append(current_adaptation)
            
            # Get adaptation parameters
            params = get_reset_parameters(reset_type)
            current_adaptation = {
                'start': timestamp,
                'type': reset_type,
                'adaptive_days': params.get('adaptive_days', 7),
                'warmup_measurements': params.get('warmup_measurements', 10),
                'measurements': [r],
                'end': None
            }
            
            # Add gap region for hard resets
            if reset_type == 'hard' and reset.get('gap_days') and last_timestamp:
                gap_regions.append({
                    'start': last_timestamp,
                    'end': timestamp,
                    'days': reset.get('gap_days'),
                    'last_weight': last_weight,
                    'next_weight': r.get('raw_weight')
                })
        
        # Track adaptation progress
        if current_adaptation:
            current_adaptation['measurements'].append(r)
            
            # Check if adaptation period ended
            days_since = (timestamp - current_adaptation['start']).total_seconds() / 86400
            measurements_count = len(current_adaptation['measurements'])
            
            if (days_since >= current_adaptation['adaptive_days'] or 
                measurements_count >= current_adaptation['warmup_measurements']):
                current_adaptation['end'] = timestamp
                adaptation_periods.append(current_adaptation)
                current_adaptation = None
        
        # Update tracking
        last_timestamp = timestamp
        last_weight = r.get('raw_weight', r.get('filtered_weight'))
    
    # Close any open adaptation period
    if current_adaptation:
        current_adaptation['end'] = last_timestamp
        adaptation_periods.append(current_adaptation)
    
    return reset_events, adaptation_periods, gap_regions


def get_reset_parameters(reset_type: str) -> Dict[str, Any]:
    """Get default parameters for reset type."""
    defaults = {
        'hard': {
            'adaptive_days': 7,
            'warmup_measurements': 10,
            'weight_boost': 10,
            'trend_boost': 100
        },
        'initial': {
            'adaptive_days': 7,
            'warmup_measurements': 10,
            'weight_boost': 10,
            'trend_boost': 100
        },
        'soft': {
            'adaptive_days': 10,
            'warmup_measurements': 15,
            'weight_boost': 3,
            'trend_boost': 10
        }
    }
    return defaults.get(reset_type, defaults['hard'])


def add_reset_markers(fig: go.Figure, reset_events: List[Dict], y_range: Tuple[float, float]):
    """Add vertical lines for reset events."""
    
    for reset in reset_events:
        style = RESET_COLORS.get(reset['type'], RESET_COLORS['unknown'])
        
        # Add vertical line as a trace
        fig.add_trace(go.Scatter(
            x=[reset['timestamp'], reset['timestamp']],
            y=[y_range[0], y_range[1]],
            mode='lines',
            name=style['name'],
            line=dict(
                color=style['line'],
                width=style['width'],
                dash=style['dash']
            ),
            showlegend=False,
            hovertemplate=format_reset_hover(reset)
        ))
        
        # Add annotation at top
        fig.add_annotation(
            x=reset['timestamp'],
            y=y_range[1],
            text=style['symbol'],
            showarrow=False,
            yshift=10,
            font=dict(size=14, color=style['line']),
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor=style['line'],
            borderwidth=1
        )


def add_adaptation_regions(fig: go.Figure, adaptation_periods: List[Dict], y_range: Tuple[float, float]):
    """Add shaded regions for adaptation periods."""
    
    for period in adaptation_periods:
        style = RESET_COLORS.get(period['type'], RESET_COLORS['unknown'])
        
        # Calculate opacity gradient (stronger at start, fading at end)
        if period['end']:
            # Add shaded rectangle
            fig.add_shape(
                type="rect",
                x0=period['start'],
                x1=period['end'],
                y0=y_range[0],
                y1=y_range[1],
                fillcolor=style['adaptation'],
                layer="below",
                line_width=0,
            )
            
            # Add subtle border
            fig.add_shape(
                type="line",
                x0=period['start'],
                x1=period['start'],
                y0=y_range[0],
                y1=y_range[1],
                line=dict(
                    color=style['line'],
                    width=1,
                    dash="dot"
                ),
                layer="below"
            )


def add_gap_regions(fig: go.Figure, gap_regions: List[Dict], y_range: Tuple[float, float]):
    """Add visualization for data gaps."""
    
    for gap in gap_regions:
        # Add gray shaded region
        fig.add_shape(
            type="rect",
            x0=gap['start'],
            x1=gap['end'],
            y0=y_range[0],
            y1=y_range[1],
            fillcolor='rgba(128, 128, 128, 0.1)',
            layer="below",
            line=dict(
                color='rgba(128, 128, 128, 0.3)',
                width=1,
                dash="dash"
            )
        )
        
        # Add text label in middle of gap
        gap_middle = gap['start'] + (gap['end'] - gap['start']) / 2
        fig.add_annotation(
            x=gap_middle,
            y=(y_range[0] + y_range[1]) / 2,
            text=f"No Data<br>{gap['days']:.0f} days",
            showarrow=False,
            font=dict(size=10, color='rgba(128, 128, 128, 0.7)'),
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='rgba(128, 128, 128, 0.3)',
            borderwidth=1
        )
        
        # Add connecting line from last to next weight
        if gap.get('last_weight') and gap.get('next_weight'):
            fig.add_trace(go.Scatter(
                x=[gap['start'], gap['end']],
                y=[gap['last_weight'], gap['next_weight']],
                mode='lines',
                line=dict(
                    color='rgba(128, 128, 128, 0.3)',
                    width=1,
                    dash='dot'
                ),
                showlegend=False,
                hoverinfo='skip'
            ))


def format_reset_hover(reset: Dict) -> str:
    """Format hover text for reset event."""
    
    reset_type = reset['type'].replace('_', ' ').title()
    lines = [
        f"<b>{reset_type}</b>",
        f"Time: %{{x|%Y-%m-%d %H:%M}}"
    ]
    
    if reset['type'] == 'hard' and reset.get('gap_days'):
        lines.append(f"Gap: {reset['gap_days']:.0f} days")
    
    if reset['type'] == 'soft':
        if reset.get('previous_weight') and reset.get('weight'):
            change = reset['weight'] - reset['previous_weight']
            lines.append(f"Weight change: {change:+.1f} kg")
    
    if reset.get('reason'):
        reason = reset['reason'].replace('_', ' ')
        lines.append(f"Reason: {reason}")
    
    lines.append("<extra></extra>")
    return "<br>".join(lines)


def add_reset_legend(fig: go.Figure):
    """Add legend entries for reset types."""
    
    # Add dummy traces for legend
    for reset_type, style in RESET_COLORS.items():
        if reset_type != 'unknown':
            fig.add_trace(go.Scatter(
                x=[None],
                y=[None],
                mode='lines',
                name=f"{style['symbol']} {style['name']}",
                line=dict(
                    color=style['line'],
                    width=style['width'],
                    dash=style['dash']
                ),
                showlegend=True
            ))


def add_adaptation_progress_indicator(fig: go.Figure, current_state: Dict, position: Dict):
    """Add a progress indicator for current adaptation."""
    
    if not current_state.get('reset_timestamp'):
        return
    
    reset_timestamp = current_state['reset_timestamp']
    if isinstance(reset_timestamp, str):
        reset_timestamp = datetime.fromisoformat(reset_timestamp)
    
    reset_type = current_state.get('reset_type', 'unknown')
    params = current_state.get('reset_parameters', {})
    
    # Calculate progress
    now = datetime.now()
    days_since = (now - reset_timestamp).total_seconds() / 86400
    adaptive_days = params.get('adaptive_days', 7)
    measurements_since = current_state.get('measurements_since_reset', 0)
    warmup_measurements = params.get('warmup_measurements', 10)
    
    # Determine if still adapting
    if days_since < adaptive_days or measurements_since < warmup_measurements:
        time_progress = min(1.0, days_since / adaptive_days)
        measurement_progress = min(1.0, measurements_since / warmup_measurements)
        overall_progress = max(time_progress, measurement_progress)
        
        style = RESET_COLORS.get(reset_type, RESET_COLORS['unknown'])
        
        # Add progress annotation
        fig.add_annotation(
            x=position.get('x', 0.98),
            y=position.get('y', 0.98),
            xref="paper",
            yref="paper",
            text=f"<b>Adapting</b><br>{style['symbol']} {int(overall_progress*100)}%<br>Day {int(days_since)}/{adaptive_days}",
            showarrow=False,
            font=dict(size=10, color=style['line']),
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor=style['line'],
            borderwidth=2,
            align="right"
        )


def enhance_with_reset_insights(fig: go.Figure, results: List[Dict[str, Any]], config: Dict[str, Any] = None):
    """
    Main function to enhance a figure with reset insights.
    
    Args:
        fig: Existing plotly figure to enhance
        results: List of measurement results
        config: Configuration dictionary
    """
    
    if not results:
        return
    
    # Get configuration
    reset_config = config.get('visualization', {}).get('reset_insights', {}) if config else {}
    
    if not reset_config.get('enabled', True):
        return
    
    # Extract reset information
    reset_events, adaptation_periods, gap_regions = extract_reset_insights(results)
    
    if not reset_events and not adaptation_periods and not gap_regions:
        return  # Nothing to add
    
    # Calculate y-axis range from existing data
    y_min, y_max = calculate_y_range(results)
    y_range = (y_min, y_max)
    
    # Add visualizations in order (background to foreground)
    if reset_config.get('show_gap_regions', True) and gap_regions:
        add_gap_regions(fig, gap_regions, y_range)
    
    if reset_config.get('show_adaptation_regions', True) and adaptation_periods:
        add_adaptation_regions(fig, adaptation_periods, y_range)
    
    if reset_config.get('show_reset_lines', True) and reset_events:
        add_reset_markers(fig, reset_events, y_range)
    
    # Add legend if requested
    if reset_config.get('show_in_legend', True) and reset_events:
        add_reset_legend(fig)
    
    # Add current adaptation progress if applicable
    if reset_config.get('show_progress', True):
        # Would need current state passed in for this
        pass


def calculate_y_range(results: List[Dict[str, Any]]) -> Tuple[float, float]:
    """Calculate appropriate y-axis range from results."""
    
    weights = []
    for r in results:
        if r.get('accepted'):
            weights.append(r.get('filtered_weight', r.get('raw_weight')))
        else:
            weights.append(r.get('raw_weight'))
    
    if weights:
        y_min = min(weights) * 0.98
        y_max = max(weights) * 1.02
    else:
        y_min, y_max = 0, 100
    
    return y_min, y_max