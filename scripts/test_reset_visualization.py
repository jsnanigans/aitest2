#!/usr/bin/env python3
"""
Test script for reset insights visualization.
Uses the basic visualization with reset insights overlay.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime, timedelta
import plotly.graph_objects as go
from src.viz_reset_insights import (
    extract_reset_insights,
    add_reset_markers,
    add_adaptation_regions,
    add_gap_regions,
    RESET_COLORS
)

def create_test_data_with_resets():
    """Create test data with various reset scenarios."""
    results = []
    base_time = datetime(2024, 1, 1)
    base_weight = 85.0
    
    # Initial measurements (will trigger initial reset)
    for i in range(5):
        results.append({
            'timestamp': base_time + timedelta(days=i),
            'raw_weight': base_weight + i * 0.1,
            'filtered_weight': base_weight + i * 0.1,
            'accepted': True,
            'reset_event': {'type': 'initial', 'reason': 'first_measurement'} if i == 0 else None
        })
    
    # Normal measurements
    for i in range(5, 20):
        results.append({
            'timestamp': base_time + timedelta(days=i),
            'raw_weight': base_weight + i * 0.05,
            'filtered_weight': base_weight + i * 0.05,
            'accepted': True
        })
    
    # Gap of 35 days (hard reset)
    gap_start = base_time + timedelta(days=20)
    gap_end = base_time + timedelta(days=55)
    
    # Measurements after gap
    for i in range(10):
        results.append({
            'timestamp': gap_end + timedelta(days=i),
            'raw_weight': base_weight + 5 + i * 0.1,
            'filtered_weight': base_weight + 5 + i * 0.1,
            'accepted': True,
            'reset_event': {
                'type': 'hard',
                'reason': 'gap_exceeded',
                'gap_days': 35,
                'last_timestamp': gap_start
            } if i == 0 else None
        })
    
    # Manual entry (soft reset)
    manual_time = gap_end + timedelta(days=15)
    results.append({
        'timestamp': manual_time,
        'raw_weight': base_weight + 10,
        'filtered_weight': base_weight + 6,
        'accepted': True,
        'reset_event': {
            'type': 'soft',
            'reason': 'manual_entry',
            'last_timestamp': manual_time - timedelta(days=1)
        }
    })
    
    # More measurements after soft reset
    for i in range(1, 10):
        results.append({
            'timestamp': manual_time + timedelta(days=i),
            'raw_weight': base_weight + 10 - i * 0.2,
            'filtered_weight': base_weight + 10 - i * 0.2,
            'accepted': True
        })
    
    return results

def create_simple_visualization_with_resets():
    """Create a simple weight timeline with reset insights."""
    
    # Get test data
    results = create_test_data_with_resets()
    
    # Extract reset information
    reset_events, adaptation_periods, gap_regions = extract_reset_insights(results)
    
    print(f"Found {len(reset_events)} reset events")
    print(f"Found {len(adaptation_periods)} adaptation periods")
    print(f"Found {len(gap_regions)} gap regions")
    
    # Create figure
    fig = go.Figure()
    
    # Calculate y-axis range
    weights = [r['raw_weight'] for r in results]
    y_min = min(weights) * 0.98
    y_max = max(weights) * 1.02
    y_range = (y_min, y_max)
    
    # Add gap regions (background layer)
    for gap in gap_regions:
        fig.add_shape(
            type="rect",
            x0=gap['start'],
            x1=gap['end'],
            y0=y_min,
            y1=y_max,
            fillcolor='rgba(128, 128, 128, 0.1)',
            layer="below",
            line=dict(
                color='rgba(128, 128, 128, 0.3)',
                width=1,
                dash="dash"
            )
        )
        
        gap_middle = gap['start'] + (gap['end'] - gap['start']) / 2
        fig.add_annotation(
            x=gap_middle,
            y=(y_min + y_max) / 2,
            text=f"No Data<br>{gap['days']:.0f} days",
            showarrow=False,
            font=dict(size=10, color='rgba(128, 128, 128, 0.7)'),
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='rgba(128, 128, 128, 0.3)',
            borderwidth=1
        )
    
    # Add adaptation regions (middle layer)
    for period in adaptation_periods:
        style = RESET_COLORS.get(period['type'], RESET_COLORS['unknown'])
        if period.get('end'):
            fig.add_shape(
                type="rect",
                x0=period['start'],
                x1=period['end'],
                y0=y_min,
                y1=y_max,
                fillcolor=style['adaptation'],
                layer="below",
                line_width=0,
            )
    
    # Add weight data
    timestamps = [r['timestamp'] for r in results]
    weights = [r['raw_weight'] for r in results]
    
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=weights,
        mode='lines+markers',
        name='Weight',
        line=dict(color='#2E7D32', width=2),
        marker=dict(size=8, color='#4CAF50')
    ))
    
    # Add reset markers (foreground layer)
    for reset in reset_events:
        style = RESET_COLORS.get(reset['type'], RESET_COLORS['unknown'])
        
        # Vertical line
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
            hovertemplate=f"<b>{style['name']}</b><br>Time: %{{x|%Y-%m-%d}}<br><extra></extra>"
        ))
        
        # Symbol annotation
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
    
    # Add legend entries for reset types
    for reset_type in ['initial', 'hard', 'soft']:
        style = RESET_COLORS[reset_type]
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
    
    # Update layout
    fig.update_layout(
        title="Weight Timeline with Reset Insights",
        xaxis_title="Date",
        yaxis_title="Weight (kg)",
        height=600,
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255, 255, 255, 0.95)',
            bordercolor='rgba(0, 0, 0, 0.3)',
            borderwidth=1
        )
    )
    
    # Save to file
    output_file = "output/test_reset_visualization.html"
    fig.write_html(output_file)
    print(f"\nVisualization saved to: {output_file}")
    
    return fig

if __name__ == "__main__":
    create_simple_visualization_with_resets()