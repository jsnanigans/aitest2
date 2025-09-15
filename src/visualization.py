"""
Weight Stream Processor Visualization
Single timeline chart with detailed hover information
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from .constants import SOURCE_MARKER_SYMBOLS, REJECTION_SEVERITY_COLORS, get_rejection_severity
except ImportError:
    from constants import SOURCE_MARKER_SYMBOLS, REJECTION_SEVERITY_COLORS, get_rejection_severity

try:
    from .viz_reset_insights import extract_reset_insights, RESET_COLORS
except ImportError:
    try:
        from viz_reset_insights import extract_reset_insights, RESET_COLORS
    except ImportError:
        extract_reset_insights = None
        RESET_COLORS = None


def create_weight_timeline(results: List[Dict[str, Any]], 
                          user_id: str,
                          output_dir: str = "output",
                          use_enhanced: bool = True,
                          config: Optional[Dict[str, Any]] = None) -> str:
    """
    Create an interactive weight timeline with quality analysis hover details.
    
    Args:
        results: List of measurement results
        user_id: User identifier
        output_dir: Output directory for HTML file
        use_enhanced: Use enhanced visualization with subplots
        
    Returns:
        Path to generated HTML file
    """
    # Try to use enhanced visualization if available and requested
    if use_enhanced and create_enhanced_weight_timeline is not None:
        try:
            return create_enhanced_weight_timeline(results, user_id, output_dir, config)
        except Exception as e:
            # Fall back to basic visualization on error
            print(f"Warning: Enhanced visualization failed, using basic: {e}")
    
    # Basic visualization fallback
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly is required for visualization")
    
    if not results:
        raise ValueError("No results to visualize")
    
    # Get configuration options
    viz_config = config.get('visualization', {}) if config else {}
    marker_config = viz_config.get('markers', {})
    rejection_config = viz_config.get('rejection', {})
    
    show_source_icons = marker_config.get('show_source_icons', True)
    show_source_legend = marker_config.get('show_source_legend', True)
    show_reset_markers = marker_config.get('show_reset_markers', True)
    reset_marker_color = marker_config.get('reset_marker_color', '#FF6600')
    reset_marker_opacity = marker_config.get('reset_marker_opacity', 0.2)
    reset_marker_width = marker_config.get('reset_marker_width', 1)
    reset_marker_style = marker_config.get('reset_marker_style', 'dot')
    show_severity_colors = rejection_config.get('show_severity_colors', True)
    group_by_severity = rejection_config.get('group_by_severity', True)
    
    accepted_data = []
    rejected_data = []
    reset_events = []
    
    for r in results:
        timestamp = r.get('timestamp')
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        
        quality_info = _extract_quality_info(r)
        source = r.get('source', 'unknown')
        
        # Get marker symbol for source (if enabled)
        marker_symbol = SOURCE_MARKER_SYMBOLS.get(source, SOURCE_MARKER_SYMBOLS['default']) if show_source_icons else 'circle'
        
        # Check for reset events (if enabled)
        if show_reset_markers and r.get('reset_event'):
            reset_event = r['reset_event']
            reset_events.append({
                'timestamp': timestamp,
                'reason': reset_event.get('reason', 'gap_exceeded'),
                'gap_days': reset_event.get('gap_days', 0),
                'last_timestamp': reset_event.get('last_timestamp')
            })
        
        if r.get('accepted', False):
            accepted_data.append({
                'timestamp': timestamp,
                'weight': r.get('filtered_weight', r.get('raw_weight')),
                'raw_weight': r.get('raw_weight'),
                'confidence': r.get('confidence', 0),
                'innovation': r.get('innovation', 0),
                'quality_score': quality_info['score'],
                'quality_details': quality_info['details'],
                'source': source,
                'marker_symbol': marker_symbol,  # Always use source-based marker
                'was_reset': r.get('was_reset', False),
                'reset_reason': r.get('reset_reason'),
                'gap_days': r.get('gap_days'),
                'hover_text': _create_accepted_hover(r, quality_info)
            })
        else:
            # Calculate rejection severity for color
            reason = r.get('reason', 'Unknown')
            weight_change = abs(r.get('raw_weight', 0) - r.get('filtered_weight', r.get('raw_weight', 0)))
            severity = get_rejection_severity(reason, weight_change)
            
            rejected_data.append({
                'timestamp': timestamp,
                'weight': r.get('raw_weight'),
                'reason': reason,
                'severity': severity,
                'quality_score': quality_info['score'],
                'quality_details': quality_info['details'],
                'source': source,
                'marker_symbol': marker_symbol,
                'hover_text': _create_rejected_hover(r, quality_info)
            })
    
    fig = go.Figure()
    
    # Use new reset insights module if available
    if extract_reset_insights and RESET_COLORS:
        reset_events_new, adaptation_periods, gap_regions = extract_reset_insights(results)
        
        # Calculate y-axis range
        all_weights = []
        if accepted_data:
            all_weights.extend([d['weight'] for d in accepted_data])
        if rejected_data:
            all_weights.extend([d['weight'] for d in rejected_data])
        
        if all_weights and (reset_events_new or adaptation_periods or gap_regions):
            y_min = min(all_weights) * 0.98
            y_max = max(all_weights) * 1.02
            y_range = (y_min, y_max)
            
            reset_config = config.get('visualization', {}).get('reset_insights', {}) if config else {}
            
            # Add gap regions (background)
            if gap_regions and reset_config.get('show_gap_regions', True):
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
            if adaptation_periods and reset_config.get('show_adaptation_regions', True):
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
    
    # Fallback to old reset visualization if new module not available
    elif reset_events and (accepted_data or rejected_data):
        # Get y-axis range from data
        all_weights = []
        if accepted_data:
            all_weights.extend([d['weight'] for d in accepted_data])
        if rejected_data:
            all_weights.extend([d['weight'] for d in rejected_data])
        
        if all_weights:
            y_min = min(all_weights) * 0.98
            y_max = max(all_weights) * 1.02
            
            # Get reset visualization config
            reset_config = config.get('visualization', {}).get('reset', {})
            show_gap_regions = reset_config.get('show_gap_regions', True)
            gap_region_color = reset_config.get('gap_region_color', '#F0F0F0')
            gap_region_opacity = reset_config.get('gap_region_opacity', 0.5)
            show_gap_labels = reset_config.get('show_gap_labels', True)
            
            for reset in reset_events:
                # Add gap region if we have the last timestamp
                if show_gap_regions and reset.get('last_timestamp'):
                    last_ts = reset['last_timestamp']
                    if isinstance(last_ts, str):
                        last_ts = datetime.fromisoformat(last_ts)
                    
                    # Add shaded rectangle for gap period
                    fig.add_shape(
                        type="rect",
                        x0=last_ts,
                        x1=reset['timestamp'],
                        y0=y_min,
                        y1=y_max,
                        fillcolor=gap_region_color,
                        opacity=gap_region_opacity,
                        layer="below",
                        line_width=0,
                    )
                    
                    # Add text label in the middle of the gap
                    if show_gap_labels:
                        gap_middle = last_ts + (reset['timestamp'] - last_ts) / 2
                        fig.add_annotation(
                            x=gap_middle,
                            y=(y_min + y_max) / 2,
                            text=f"No Data<br>{reset['gap_days']:.0f} days",
                            showarrow=False,
                            font=dict(size=10, color='rgba(128, 128, 128, 0.7)'),
                            bgcolor='rgba(255, 255, 255, 0.8)',
                            bordercolor='rgba(128, 128, 128, 0.3)',
                            borderwidth=1
                        )
                
                # Add vertical line at reset point
                fig.add_trace(go.Scatter(
                    x=[reset['timestamp'], reset['timestamp']],
                    y=[y_min, y_max],
                    mode='lines',
                    name=f"Reset ({reset['gap_days']:.0f}d gap)",
                    line=dict(
                        color=f'rgba({int(reset_marker_color[1:3], 16)}, {int(reset_marker_color[3:5], 16)}, {int(reset_marker_color[5:7], 16)}, {reset_marker_opacity})',
                        width=reset_marker_width,
                        dash=reset_marker_style
                    ),
                    showlegend=False,  # Don't show in legend
                    hovertemplate=f"<b>System Reset</b><br>Gap: {reset['gap_days']:.0f} days<br>Reason: {reset['reason']}<br>%{{x}}<extra></extra>"
                ))
                
                # Add subtle annotation at the top
                fig.add_annotation(
                    x=reset['timestamp'],
                    y=y_max,
                    text=f"Reset",
                    showarrow=False,
                    yshift=5,
                    font=dict(size=8, color='rgba(255, 102, 0, 0.6)'),
                    bgcolor='rgba(255, 255, 255, 0.7)',
                    bordercolor='rgba(255, 102, 0, 0.3)',
                    borderwidth=0.5
                )
    
    if accepted_data:
        fig.add_trace(go.Scatter(
            x=[d['timestamp'] for d in accepted_data],
            y=[d['weight'] for d in accepted_data],
            mode='lines+markers',
            name='Accepted',
            line=dict(color='#2E7D32', width=2),
            marker=dict(
                size=10,
                color=[d['quality_score'] for d in accepted_data],
                colorscale='Viridis',
                cmin=0,
                cmax=1,
                showscale=True,
                colorbar=dict(
                    title="Quality<br>Score",
                    x=1.05,
                    y=0.5,
                    len=0.5
                ),
                symbol=[d['marker_symbol'] for d in accepted_data],
                line=dict(color='#1B5E20', width=1)
            ),
            text=[d['hover_text'] for d in accepted_data],
            hovertemplate='%{text}<extra></extra>'
        ))
    
    if rejected_data:
        if group_by_severity and show_severity_colors:
            # Create separate traces for each severity level to show in legend
            severity_groups = {}
            for d in rejected_data:
                severity = d['severity']
                if severity not in severity_groups:
                    severity_groups[severity] = []
                severity_groups[severity].append(d)
            
            for severity, data_points in severity_groups.items():
                fig.add_trace(go.Scatter(
                    x=[d['timestamp'] for d in data_points],
                    y=[d['weight'] for d in data_points],
                    mode='markers',
                    name=f'Rejected ({severity})',
                    marker=dict(
                        size=12,
                        color=REJECTION_SEVERITY_COLORS.get(severity, '#FF0000') if show_severity_colors else '#C62828',
                        symbol=[d['marker_symbol'] for d in data_points],
                        line=dict(color='#8B0000', width=1)
                    ),
                    text=[d['hover_text'] for d in data_points],
                    hovertemplate='%{text}<extra></extra>'
                ))
        else:
            # Single trace for all rejected points
            fig.add_trace(go.Scatter(
                x=[d['timestamp'] for d in rejected_data],
                y=[d['weight'] for d in rejected_data],
                mode='markers',
                name='Rejected',
                marker=dict(
                    size=12,
                    color=[REJECTION_SEVERITY_COLORS.get(d['severity'], '#FF0000') for d in rejected_data] if show_severity_colors else '#C62828',
                    symbol=[d['marker_symbol'] for d in rejected_data],
                    line=dict(color='#8B0000', width=1)
                ),
                text=[d['hover_text'] for d in rejected_data],
                hovertemplate='%{text}<extra></extra>'
            ))
    
    # Add invisible traces for source type legend (if enabled)
    if show_source_legend:
        source_types = {
            'care-team-upload': ('Care Team', 'triangle-up'),
            'patient-upload': ('Patient Upload', 'circle'),
            'internal-questionnaire': ('Questionnaire', 'square'),
            'patient-device': ('Device', 'diamond'),
            'https://connectivehealth.io': ('ConnectiveHealth', 'hexagon'),
            'https://api.iglucose.com': ('iGlucose', 'hexagon')
        }
        
        # Add legend entries for source types
        for source_key, (display_name, symbol) in source_types.items():
            fig.add_trace(go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                name=f"Source: {display_name}",
                marker=dict(
                    size=10,
                    symbol=symbol,
                    color='gray',
                    line=dict(color='darkgray', width=1)
                ),
                showlegend=True,
                legendgroup='sources',
                legendgrouptitle_text="Source Types"
            ))
    
    # Add reset markers (foreground layer) if using new module
    if extract_reset_insights and RESET_COLORS and 'reset_events_new' in locals():
        reset_config = config.get('visualization', {}).get('reset_insights', {}) if config else {}
        
        if reset_events_new and reset_config.get('show_reset_lines', True):
            for reset in reset_events_new:
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
                    hovertemplate=f"<b>{style['name']}</b><br>Time: %{{x|%Y-%m-%d %H:%M}}<br><extra></extra>"
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
        if reset_config.get('show_in_legend', True) and reset_events_new:
            seen_types = set(r['type'] for r in reset_events_new)
            for reset_type in seen_types:
                if reset_type in RESET_COLORS:
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
                        showlegend=True,
                        legendgroup='resets',
                        legendgrouptitle_text="Reset Events"
                    ))
    
    stats = _calculate_stats(results, accepted_data, rejected_data)
    
    fig.update_layout(
        title={
            'text': f'Weight Measurements - {user_id}<br><sub>{stats}</sub>',
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title='Date',
        yaxis_title='Weight (kg)',
        height=700,
        hovermode='closest',
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255, 255, 255, 0.95)',
            bordercolor='rgba(0, 0, 0, 0.3)',
            borderwidth=1,
            font=dict(size=10),
            tracegroupgap=20,
            itemsizing='constant'
        ),
        plot_bgcolor='#FAFAFA',
        paper_bgcolor='white',
        xaxis=dict(
            showgrid=True,
            gridcolor='#E0E0E0',
            showline=True,
            linecolor='#BDBDBD'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='#E0E0E0',
            showline=True,
            linecolor='#BDBDBD'
        )
    )
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    html_file = output_path / f"{user_id}_timeline.html"
    
    config = {
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
        'toImageButtonOptions': {
            'format': 'png',
            'filename': f'weight_timeline_{user_id}',
            'height': 700,
            'width': 1200,
            'scale': 2
        }
    }
    
    fig.write_html(str(html_file), config=config)
    
    return str(html_file)


def _extract_quality_info(result: Dict[str, Any]) -> Dict[str, Any]:
    """Extract quality score information from result."""
    quality_info = {
        'score': 0,
        'details': {}
    }
    
    if 'quality_score' in result:
        qs = result['quality_score']
        if isinstance(qs, dict):
            quality_info['score'] = qs.get('overall', 0)
            quality_info['details'] = qs.get('components', {})
        elif isinstance(qs, (int, float)):
            quality_info['score'] = qs
    
    return quality_info


def _create_accepted_hover(result: Dict[str, Any], quality_info: Dict[str, Any]) -> str:
    """Create hover text for accepted measurements."""
    timestamp = result.get('timestamp')
    if isinstance(timestamp, str):
        timestamp = datetime.fromisoformat(timestamp)
    
    lines = [
        f"<b>ACCEPTED</b>",
        f"<b>Date:</b> {timestamp.strftime('%Y-%m-%d %H:%M')}",
        f"<b>Weight:</b> {result.get('filtered_weight', 0):.2f} kg",
        f"<b>Raw:</b> {result.get('raw_weight', 0):.2f} kg"
    ]
    
    # Add reset information if applicable
    if result.get('was_reset', False):
        lines.append(f"<b>⚠️ KALMAN FILTER RESET</b>")
        if result.get('gap_days'):
            lines.append(f"<b>Data Gap:</b> {result['gap_days']:.1f} days")
        lines.append(f"<b>Filter reinitialized from this measurement</b>")
    
    if 'confidence' in result:
        lines.append(f"<b>Confidence:</b> {result['confidence']:.3f}")
    
    if 'innovation' in result:
        lines.append(f"<b>Innovation:</b> {result['innovation']:.3f} kg")
    
    lines.append(f"<b>Quality Score:</b> {quality_info['score']:.2f}")
    
    if quality_info['details']:
        lines.append("<b>Quality Components:</b>")
        for key, value in quality_info['details'].items():
            lines.append(f"  • {key}: {value:.2f}")
    
    if 'source' in result:
        lines.append(f"<b>Source:</b> {_format_source(result['source'])}")
    
    return "<br>".join(lines)


def _create_rejected_hover(result: Dict[str, Any], quality_info: Dict[str, Any]) -> str:
    """Create hover text for rejected measurements."""
    timestamp = result.get('timestamp')
    if isinstance(timestamp, str):
        timestamp = datetime.fromisoformat(timestamp)
    
    lines = [
        f"<b>REJECTED</b>",
        f"<b>Date:</b> {timestamp.strftime('%Y-%m-%d %H:%M')}",
        f"<b>Weight:</b> {result.get('raw_weight', 0):.2f} kg",
        f"<b>Reason:</b> {result.get('reason', 'Unknown')}"
    ]
    
    lines.append(f"<b>Quality Score:</b> {quality_info['score']:.2f}")
    
    if quality_info['details']:
        lines.append("<b>Quality Components:</b>")
        for key, value in quality_info['details'].items():
            lines.append(f"  • {key}: {value:.2f}")
    
    if 'source' in result:
        lines.append(f"<b>Source:</b> {_format_source(result['source'])}")
    
    return "<br>".join(lines)


def _format_source(source: str) -> str:
    """Format source string for display."""
    source_map = {
        'care-team-upload': 'Care Team',
        'patient-upload': 'Patient Upload',
        'patient-device': 'Patient Device',
        'internal-questionnaire': 'Internal Questionnaire',
        'initial-questionnaire': 'Initial Questionnaire',
        'https://connectivehealth.io': 'ConnectiveHealth',
        'https://api.iglucose.com': 'iGlucose'
    }
    return source_map.get(source, source)


def _calculate_stats(results: List[Dict[str, Any]], 
                     accepted_data: List[Dict[str, Any]], 
                     rejected_data: List[Dict[str, Any]]) -> str:
    """Calculate summary statistics for display."""
    total = len(results)
    accepted = len(accepted_data)
    rejected = len(rejected_data)
    
    if total > 0:
        acceptance_rate = (accepted / total) * 100
    else:
        acceptance_rate = 0
    
    stats_parts = [
        f"Total: {total}",
        f"Accepted: {accepted} ({acceptance_rate:.1f}%)",
        f"Rejected: {rejected}"
    ]
    
    if accepted_data:
        weights = [d['weight'] for d in accepted_data]
        avg_weight = sum(weights) / len(weights)
        stats_parts.append(f"Avg Weight: {avg_weight:.1f} kg")
    
    return " | ".join(stats_parts)


# Import the comprehensive index function from viz_index
try:
    from .viz_index import create_index_from_results
except ImportError:
    from viz_index import create_index_from_results

# Import enhanced visualization
try:
    from .viz_enhanced import create_enhanced_weight_timeline
except ImportError:
    try:
        from viz_enhanced import create_enhanced_weight_timeline
    except ImportError:
        create_enhanced_weight_timeline = None