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


def create_weight_timeline(results: List[Dict[str, Any]], 
                          user_id: str,
                          output_dir: str = "output") -> str:
    """
    Create an interactive weight timeline with quality analysis hover details.
    
    Args:
        results: List of measurement results
        user_id: User identifier
        output_dir: Output directory for HTML file
        
    Returns:
        Path to generated HTML file
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly is required for visualization")
    
    if not results:
        raise ValueError("No results to visualize")
    
    accepted_data = []
    rejected_data = []
    
    for r in results:
        timestamp = r.get('timestamp')
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        
        quality_info = _extract_quality_info(r)
        
        if r.get('accepted', False):
            accepted_data.append({
                'timestamp': timestamp,
                'weight': r.get('filtered_weight', r.get('raw_weight')),
                'raw_weight': r.get('raw_weight'),
                'confidence': r.get('confidence', 0),
                'innovation': r.get('innovation', 0),
                'quality_score': quality_info['score'],
                'quality_details': quality_info['details'],
                'source': r.get('source', 'unknown'),
                'hover_text': _create_accepted_hover(r, quality_info)
            })
        else:
            rejected_data.append({
                'timestamp': timestamp,
                'weight': r.get('raw_weight'),
                'reason': r.get('reason', 'Unknown'),
                'quality_score': quality_info['score'],
                'quality_details': quality_info['details'],
                'source': r.get('source', 'unknown'),
                'hover_text': _create_rejected_hover(r, quality_info)
            })
    
    fig = go.Figure()
    
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
                line=dict(color='#1B5E20', width=1)
            ),
            text=[d['hover_text'] for d in accepted_data],
            hovertemplate='%{text}<extra></extra>'
        ))
    
    if rejected_data:
        fig.add_trace(go.Scatter(
            x=[d['timestamp'] for d in rejected_data],
            y=[d['weight'] for d in rejected_data],
            mode='markers',
            name='Rejected',
            marker=dict(
                size=12,
                color='#C62828',
                symbol='x',
                line=dict(color='#B71C1C', width=2)
            ),
            text=[d['hover_text'] for d in rejected_data],
            hovertemplate='%{text}<extra></extra>'
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
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='rgba(0, 0, 0, 0.2)',
            borderwidth=1
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