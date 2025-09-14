import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import json

from .viz_quality import (
    extract_quality_data,
    calculate_quality_statistics,
    detect_quality_patterns,
    generate_quality_insights,
    analyze_rejection_reasons,
    analyze_source_quality,
    calculate_quality_trends
)

from .viz_kalman import (
    extract_kalman_data,
    calculate_kalman_metrics,
    calculate_uncertainty_bands,
    identify_resets_and_gaps,
    calculate_filter_performance,
    calculate_baseline_and_thresholds,
    prepare_kalman_visualization_data,
    format_kalman_hover_text
)

try:
    from .viz_logger import get_logger
    logger = get_logger()
except ImportError:
    # Fallback if logger not available
    class SimpleLogger:
        def debug(self, msg): pass
        def info(self, msg): pass
        def progress(self, msg): pass
        def success(self, msg): print(msg)
        def warning(self, msg): print(f"Warning: {msg}")
        def error(self, msg): print(f"Error: {msg}")
    logger = SimpleLogger()

def create_enhanced_dashboard(results: List[Dict[str, Any]], 
                            user_id: str,
                            output_dir: str = "output",
                            config: Optional[Dict[str, Any]] = None) -> str:
    """Create enhanced dashboard with Kalman filter insights."""
    
    # Extract both quality and Kalman data
    quality_df = extract_quality_data(results)
    kalman_df = extract_kalman_data(results)
    
    if kalman_df.empty and quality_df.empty:
        logger.debug(f"No data to visualize for user {user_id}")
        return None
    
    # Use Kalman df as primary if available, otherwise fall back to quality df
    df = kalman_df if not kalman_df.empty else quality_df
    
    # Calculate all metrics
    stats = calculate_quality_statistics(quality_df) if not quality_df.empty else {}
    patterns = detect_quality_patterns(quality_df) if not quality_df.empty else []
    insights = generate_quality_insights(quality_df, stats, patterns) if not quality_df.empty else []
    
    # Kalman-specific data
    kalman_viz_data = prepare_kalman_visualization_data(kalman_df, config) if not kalman_df.empty else {}
    kalman_metrics = kalman_viz_data.get("metrics", {})
    thresholds = kalman_viz_data.get("thresholds", {})
    
    # Get config sections
    viz_config = config.get("visualization", {}) if config else {}
    interactive_config = viz_config.get("interactive", {})
    quality_config = viz_config.get("quality", {})
    
    # Create the enhanced dashboard
    fig = create_enhanced_layout(
        df, kalman_viz_data, stats, insights,
        kalman_metrics, thresholds,
        viz_config, interactive_config, quality_config
    )
    
    # Save to file
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    html_file = output_path / f"dashboard_enhanced_{user_id}.html"
    
    fig.write_html(
        str(html_file),
        include_plotlyjs='directory',
        config={
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
            'toImageButtonOptions': {
                'format': 'png',
                'filename': f'dashboard_kalman_{user_id}',
                'height': 1080,
                'width': 1920,
                'scale': 2
            }
        }
    )
    
    logger.success(f"✓ Enhanced: {html_file.name}")
    return str(html_file)

def create_enhanced_layout(df: pd.DataFrame,
                          kalman_data: Dict[str, Any],
                          stats: Dict[str, Any],
                          insights: List[Dict[str, Any]],
                          kalman_metrics: Dict[str, Any],
                          thresholds: Dict[str, Any],
                          viz_config: Dict[str, Any],
                          interactive_config: Dict[str, Any],
                          quality_config: Dict[str, Any]) -> go.Figure:
    """Create enhanced dashboard layout with Kalman insights."""
    
    theme = viz_config.get("theme", "plotly_white")
    
    # Create subplot layout - enhanced for Kalman
    fig = make_subplots(
        rows=4, cols=4,
        row_heights=[0.35, 0.25, 0.20, 0.20],
        column_widths=[0.25, 0.25, 0.25, 0.25],
        subplot_titles=(
            "Weight Measurements with Kalman Filter", None, None, "Filter Statistics",
            "Innovation (Measurement - Prediction)", "Normalized Innovation", "Confidence Evolution", "Daily Changes",
            "Quality Score Timeline", "Component Breakdown", "Quality Distribution", "Insights",
            "Filter Performance", None, "Processing Overview", None
        ),
        specs=[
            [{"colspan": 3}, None, None, {"type": "table", "rowspan": 2}],
            [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}, None],
            [{"type": "scatter"}, {"type": "bar"}, {"type": "histogram"}, {"type": "table"}],
            [{"colspan": 2, "type": "table"}, None, {"colspan": 2, "type": "bar"}, None]
        ],
        horizontal_spacing=0.08,
        vertical_spacing=0.10
    )
    
    # Add main Kalman filter chart
    add_kalman_main_chart(fig, df, kalman_data, thresholds, quality_config, row=1, col=1)
    
    # Add Kalman analytics subplots
    add_innovation_chart(fig, df, row=2, col=1)
    add_normalized_innovation_chart(fig, df, row=2, col=2)
    add_confidence_evolution_chart(fig, df, row=2, col=3)
    
    # Add quality analysis charts
    add_quality_timeline_simple(fig, df, quality_config, row=3, col=1)
    add_component_breakdown_simple(fig, df, row=3, col=2)
    add_daily_change_distribution(fig, kalman_data, row=3, col=3)
    
    # Add statistics and insights
    add_kalman_statistics_table(fig, stats, kalman_metrics, thresholds, row=1, col=4)
    add_insights_table(fig, insights, row=3, col=4)
    
    # Add filter performance panel
    add_filter_performance_table(fig, kalman_metrics, row=4, col=1)
    
    # Add processing overview
    add_processing_overview(fig, df, row=4, col=3)
    
    # Update layout
    fig.update_layout(
        title={
            'text': f"Enhanced Weight Analytics with Kalman Filter",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24}
        },
        template=theme,
        height=1200,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.12,
            xanchor="center",
            x=0.5
        ),
        hovermode='x unified'
    )
    
    return fig

def add_kalman_main_chart(fig: go.Figure, df: pd.DataFrame, kalman_data: Dict[str, Any], 
                          thresholds: Dict[str, Any], quality_config: Dict[str, Any], 
                          row: int, col: int):
    """Add main weight chart with Kalman filter visualization."""
    
    if df.empty:
        return
    
    # Add uncertainty bands (2-sigma)
    if "upper_2sigma" in df.columns and "lower_2sigma" in df.columns:
        x_band = pd.concat([df["timestamp"], df["timestamp"][::-1]])
        y_band = pd.concat([df["upper_2sigma"], df["lower_2sigma"][::-1]])
        
        fig.add_trace(
            go.Scatter(
                x=x_band,
                y=y_band,
                fill='toself',
                fillcolor='rgba(0, 100, 200, 0.1)',
                line=dict(color='rgba(255,255,255,0)'),
                name='±2σ Band',
                showlegend=True,
                hoverinfo='skip'
            ),
            row=row, col=col
        )
    
    # Add uncertainty bands (1-sigma)
    if "upper_1sigma" in df.columns and "lower_1sigma" in df.columns:
        x_band = pd.concat([df["timestamp"], df["timestamp"][::-1]])
        y_band = pd.concat([df["upper_1sigma"], df["lower_1sigma"][::-1]])
        
        fig.add_trace(
            go.Scatter(
                x=x_band,
                y=y_band,
                fill='toself',
                fillcolor='rgba(0, 100, 200, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='±1σ Band',
                showlegend=True,
                hoverinfo='skip'
            ),
            row=row, col=col
        )
    
    # Add Kalman filtered line
    if "filtered_weight" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["filtered_weight"],
                mode='lines',
                name='Kalman Filtered',
                line=dict(color='blue', width=2),
                hovertemplate=(
                    "<b>%{x|%Y-%m-%d %H:%M}</b><br>" +
                    "Filtered: <b>%{y:.1f} kg</b><extra></extra>"
                )
            ),
            row=row, col=col
        )
    
    # Add baseline reference
    if "baseline" in thresholds:
        fig.add_hline(
            y=thresholds["baseline"],
            line_dash="dash",
            line_color="gray",
            annotation_text="Baseline",
            row=row, col=col
        )
    
    # Add extreme thresholds
    if "upper_extreme" in thresholds:
        fig.add_hline(
            y=thresholds["upper_extreme"],
            line_dash="dot",
            line_color="orange",
            annotation_text="Upper Extreme",
            row=row, col=col
        )
    
    if "lower_extreme" in thresholds:
        fig.add_hline(
            y=thresholds["lower_extreme"],
            line_dash="dot",
            line_color="orange",
            annotation_text="Lower Extreme",
            row=row, col=col
        )
    
    # Add accepted measurements
    accepted_df = df[df["accepted"] == True]
    if not accepted_df.empty:
        # Prepare hover text
        hover_texts = accepted_df.apply(format_kalman_hover_text, axis=1)
        
        # Color by quality score if available
        if quality_config.get("show_quality_scores", True) and "quality_score" in accepted_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=accepted_df["timestamp"],
                    y=accepted_df["raw_weight"],
                    mode='markers',
                    name='Accepted',
                    marker=dict(
                        size=8,
                        color=accepted_df["quality_score"],
                        colorscale='RdYlGn',
                        cmin=0,
                        cmax=1,
                        colorbar=dict(
                            title="Quality",
                            thickness=10,
                            len=0.3,
                            x=0.73,
                            y=0.85
                        ),
                        line=dict(width=1, color='DarkSlateGrey'),
                        showscale=True
                    ),
                    text=hover_texts,
                    hovertemplate="%{text}<extra></extra>"
                ),
                row=row, col=col
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=accepted_df["timestamp"],
                    y=accepted_df["raw_weight"],
                    mode='markers',
                    name='Accepted',
                    marker=dict(size=8, color='green', symbol='circle'),
                    text=hover_texts,
                    hovertemplate="%{text}<extra></extra>"
                ),
                row=row, col=col
            )
    
    # Add rejected measurements
    rejected_df = df[df["accepted"] == False]
    if not rejected_df.empty:
        hover_texts = rejected_df.apply(format_kalman_hover_text, axis=1)
        
        fig.add_trace(
            go.Scatter(
                x=rejected_df["timestamp"],
                y=rejected_df["raw_weight"],
                mode='markers',
                name='Rejected',
                marker=dict(
                    size=8,
                    color='red',
                    symbol='circle-open',
                    line=dict(width=2, color='darkred')
                ),
                text=hover_texts,
                hovertemplate="%{text}<extra></extra>"
            ),
            row=row, col=col
        )
    
    # Add reset points
    reset_df = df[df.get("is_reset", False) == True]
    if not reset_df.empty:
        fig.add_trace(
            go.Scatter(
                x=reset_df["timestamp"],
                y=reset_df["filtered_weight"],
                mode='markers',
                name='Reset Points',
                marker=dict(
                    size=12,
                    color='purple',
                    symbol='triangle-down',
                    line=dict(width=2, color='darkviolet')
                ),
                hovertemplate=(
                    "<b>Filter Reset</b><br>" +
                    "%{x|%Y-%m-%d %H:%M}<br>" +
                    "Weight: %{y:.1f} kg<extra></extra>"
                )
            ),
            row=row, col=col
        )
    
    # Mark gap regions
    if "gaps" in kalman_data:
        for gap_start, gap_end in kalman_data["gaps"]:
            fig.add_vrect(
                x0=gap_start, x1=gap_end,
                fillcolor="gray", opacity=0.1,
                layer="below", line_width=0,
                row=row, col=col
            )
    
    fig.update_xaxes(title_text="Date", row=row, col=col)
    fig.update_yaxes(title_text="Weight (kg)", row=row, col=col)

def add_innovation_chart(fig: go.Figure, df: pd.DataFrame, row: int, col: int):
    """Add innovation (measurement - prediction) chart."""
    
    if "innovation" not in df.columns:
        return
    
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["innovation"],
            mode='markers+lines',
            name='Innovation',
            line=dict(color='purple', width=1),
            marker=dict(size=4, color='purple'),
            hovertemplate=(
                "%{x|%Y-%m-%d}<br>" +
                "Innovation: %{y:.2f} kg<extra></extra>"
            )
        ),
        row=row, col=col
    )
    
    # Add zero reference line
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="gray",
        row=row, col=col
    )
    
    # Add moving average if available
    if "innovation_ma" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["innovation_ma"],
                mode='lines',
                name='MA(10)',
                line=dict(color='orange', width=2),
                hovertemplate=(
                    "%{x|%Y-%m-%d}<br>" +
                    "MA: %{y:.2f} kg<extra></extra>"
                )
            ),
            row=row, col=col
        )
    
    fig.update_xaxes(title_text="Date", row=row, col=col)
    fig.update_yaxes(title_text="Innovation (kg)", row=row, col=col)

def add_normalized_innovation_chart(fig: go.Figure, df: pd.DataFrame, row: int, col: int):
    """Add normalized innovation chart with significance bands."""
    
    if "normalized_innovation" not in df.columns:
        return
    
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["normalized_innovation"],
            mode='markers',
            name='Normalized Innovation',
            marker=dict(
                size=6,
                color=df["normalized_innovation"].abs(),
                colorscale='Viridis',
                showscale=False
            ),
            hovertemplate=(
                "%{x|%Y-%m-%d}<br>" +
                "NI: %{y:.2f}<extra></extra>"
            )
        ),
        row=row, col=col
    )
    
    # Add significance bands (±2σ)
    fig.add_hline(y=2, line_dash="dash", line_color="orange", row=row, col=col)
    fig.add_hline(y=-2, line_dash="dash", line_color="orange", row=row, col=col)
    fig.add_hline(y=0, line_dash="solid", line_color="gray", row=row, col=col)
    
    # Shade extreme regions
    fig.add_hrect(
        y0=2, y1=5,
        fillcolor="red", opacity=0.1,
        layer="below", line_width=0,
        row=row, col=col
    )
    fig.add_hrect(
        y0=-5, y1=-2,
        fillcolor="red", opacity=0.1,
        layer="below", line_width=0,
        row=row, col=col
    )
    
    fig.update_xaxes(title_text="Date", row=row, col=col)
    fig.update_yaxes(title_text="Normalized Innovation", range=[-4, 4], row=row, col=col)

def add_confidence_evolution_chart(fig: go.Figure, df: pd.DataFrame, row: int, col: int):
    """Add confidence evolution chart."""
    
    if "confidence" not in df.columns:
        return
    
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["confidence"] * 100,  # Convert to percentage
            mode='lines+markers',
            name='Confidence',
            line=dict(color='green', width=2),
            marker=dict(size=4),
            fill='tozeroy',
            fillcolor='rgba(0, 255, 0, 0.1)',
            hovertemplate=(
                "%{x|%Y-%m-%d}<br>" +
                "Confidence: %{y:.1f}%<extra></extra>"
            )
        ),
        row=row, col=col
    )
    
    # Add threshold lines
    fig.add_hline(y=95, line_dash="dash", line_color="green", annotation_text="High", row=row, col=col)
    fig.add_hline(y=80, line_dash="dash", line_color="orange", annotation_text="Medium", row=row, col=col)
    fig.add_hline(y=60, line_dash="dash", line_color="red", annotation_text="Low", row=row, col=col)
    
    fig.update_xaxes(title_text="Date", row=row, col=col)
    fig.update_yaxes(title_text="Confidence (%)", range=[0, 100], row=row, col=col)

def add_daily_change_distribution(fig: go.Figure, kalman_data: Dict[str, Any], row: int, col: int):
    """Add daily change distribution histogram."""
    
    daily_change_data = kalman_data.get("daily_change_hist")
    if not daily_change_data:
        return
    
    values = daily_change_data["values"]
    mean = daily_change_data["mean"]
    std = daily_change_data["std"]
    
    fig.add_trace(
        go.Histogram(
            x=values,
            name='Daily Changes',
            marker_color='lightblue',
            opacity=0.7,
            nbinsx=30,
            hovertemplate="Change: %{x:.2f} kg<br>Count: %{y}<extra></extra>"
        ),
        row=row, col=col
    )
    
    # Add normal distribution overlay
    x_range = np.linspace(min(values), max(values), 100)
    normal_dist = stats.norm.pdf(x_range, mean, std) * len(values) * (max(values) - min(values)) / 30
    
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=normal_dist,
            mode='lines',
            name='Normal Fit',
            line=dict(color='red', width=2),
            hovertemplate="Expected: %{y:.1f}<extra></extra>"
        ),
        row=row, col=col
    )
    
    # Add mean line
    fig.add_vline(
        x=mean,
        line_dash="dash",
        line_color="green",
        annotation_text=f"Mean: {mean:.2f}",
        row=row, col=col
    )
    
    fig.update_xaxes(title_text="Daily Change (kg)", row=row, col=col)
    fig.update_yaxes(title_text="Frequency", row=row, col=col)

def add_kalman_statistics_table(fig: go.Figure, stats: Dict[str, Any], 
                                kalman_metrics: Dict[str, Any], 
                                thresholds: Dict[str, Any], 
                                row: int, col: int):
    """Add Kalman filter statistics table."""
    
    table_data = []
    
    # Current state
    table_data.append(["<b>CURRENT STATE</b>", ""])
    if "current_weight" in thresholds:
        table_data.append(["Current Weight", f"{thresholds['current_weight']:.1f} kg"])
    if "baseline" in thresholds:
        table_data.append(["Baseline", f"{thresholds['baseline']:.1f} kg"])
    
    # Kalman metrics
    table_data.append(["", ""])
    table_data.append(["<b>FILTER METRICS</b>", ""])
    if "innovation_mean" in kalman_metrics:
        table_data.append(["Innovation Mean", f"{kalman_metrics['innovation_mean']:.2f}"])
    if "innovation_std" in kalman_metrics:
        table_data.append(["Innovation Std", f"{kalman_metrics['innovation_std']:.2f}"])
    if "avg_confidence" in kalman_metrics:
        table_data.append(["Avg Confidence", f"{kalman_metrics['avg_confidence']:.1%}"])
    if "tracking_error" in kalman_metrics:
        table_data.append(["Tracking Error", f"{kalman_metrics['tracking_error']:.2f}"])
    
    # Quality metrics
    if stats:
        table_data.append(["", ""])
        table_data.append(["<b>QUALITY METRICS</b>", ""])
        if "overall_quality" in stats:
            table_data.append(["Quality Score", f"{stats['overall_quality']:.2f}"])
        if "acceptance_rate" in stats:
            table_data.append(["Acceptance Rate", f"{stats['acceptance_rate']:.1f}%"])
    
    # Filter health
    table_data.append(["", ""])
    table_data.append(["<b>FILTER HEALTH</b>", ""])
    
    # Determine filter health status
    health_status = "Good"
    health_color = "green"
    if kalman_metrics.get("innovation_mean", 0) > 1.0:
        health_status = "Check Bias"
        health_color = "orange"
    if kalman_metrics.get("tracking_error", 0) > 3.0:
        health_status = "Poor Tracking"
        health_color = "red"
    
    table_data.append(["Status", f"<span style='color:{health_color}'>{health_status}</span>"])
    
    labels = [row[0] for row in table_data]
    values = [row[1] for row in table_data]
    
    colors = []
    for label in labels:
        if "<b>" in label:
            colors.append("lightblue")
        else:
            colors.append("white")
    
    fig.add_trace(
        go.Table(
            header=dict(
                values=["<b>Metric</b>", "<b>Value</b>"],
                fill_color='darkblue',
                font=dict(color='white', size=12),
                align='left'
            ),
            cells=dict(
                values=[labels, values],
                fill_color=[colors, colors],
                align=['left', 'right'],
                font=dict(size=11),
                height=20
            )
        ),
        row=row, col=col
    )

def add_filter_performance_table(fig: go.Figure, kalman_metrics: Dict[str, Any], row: int, col: int):
    """Add filter performance metrics table."""
    
    table_data = []
    
    table_data.append(["<b>FILTER PERFORMANCE</b>", ""])
    
    # Innovation tests
    table_data.append(["Innovation Tests", ""])
    if "innovation_is_normal" in kalman_metrics:
        status = "✓ Pass" if kalman_metrics["innovation_is_normal"] else "✗ Fail"
        table_data.append(["  Normality", status])
    if "innovation_mean" in kalman_metrics:
        bias_ok = abs(kalman_metrics["innovation_mean"]) < 0.5
        status = "✓ Pass" if bias_ok else "✗ Fail"
        table_data.append(["  Zero Mean", status])
    
    # NIS consistency
    if "nis_consistency" in kalman_metrics:
        status = "✓ Pass" if kalman_metrics["nis_consistency"] else "✗ Fail"
        table_data.append(["NIS Consistency", status])
    
    # Convergence
    if "confidence_trend" in kalman_metrics:
        trend = kalman_metrics["confidence_trend"]
        if trend > 0:
            status = "↑ Improving"
        elif trend < -0.1:
            status = "↓ Degrading"
        else:
            status = "→ Stable"
        table_data.append(["Convergence", status])
    
    # Overall assessment
    table_data.append(["", ""])
    if "tracking_quality" in kalman_metrics:
        quality = kalman_metrics["tracking_quality"]
        color = "green" if quality == "good" else "red"
        table_data.append(["<b>Overall</b>", f"<span style='color:{color}'>{quality.upper()}</span>"])
    
    labels = [row[0] for row in table_data]
    values = [row[1] for row in table_data]
    
    fig.add_trace(
        go.Table(
            cells=dict(
                values=[labels, values],
                align=['left', 'center'],
                font=dict(size=11),
                height=25
            )
        ),
        row=row, col=col
    )

def add_processing_overview(fig: go.Figure, df: pd.DataFrame, row: int, col: int):
    """Add processing overview bar chart."""
    
    if df.empty:
        return
    
    # Count by source
    source_counts = df.groupby("source").agg({
        "accepted": ["sum", "count"]
    })
    source_counts.columns = ["accepted", "total"]
    source_counts["rejected"] = source_counts["total"] - source_counts["accepted"]
    source_counts["acceptance_rate"] = source_counts["accepted"] / source_counts["total"] * 100
    
    fig.add_trace(
        go.Bar(
            x=source_counts.index,
            y=source_counts["accepted"],
            name='Accepted',
            marker_color='green',
            hovertemplate="%{x}<br>Accepted: %{y}<extra></extra>"
        ),
        row=row, col=col
    )
    
    fig.add_trace(
        go.Bar(
            x=source_counts.index,
            y=source_counts["rejected"],
            name='Rejected',
            marker_color='red',
            hovertemplate="%{x}<br>Rejected: %{y}<extra></extra>"
        ),
        row=row, col=col
    )
    
    fig.update_xaxes(title_text="Source", row=row, col=col)
    fig.update_yaxes(title_text="Count", row=row, col=col)
    fig.update_layout(barmode='stack')

def add_insights_table(fig: go.Figure, insights: List[Dict[str, Any]], row: int, col: int):
    """Add insights table."""
    
    if not insights:
        insights = [{"title": "No insights available", "description": "Process more data"}]
    
    table_data = []
    table_data.append(["<b>KEY INSIGHTS</b>", ""])
    
    for i, insight in enumerate(insights[:5], 1):
        icon = {"alert": "⚠️", "warning": "⚡", "info": "ℹ️"}.get(insight.get("type", "info"), "•")
        table_data.append([f"{icon} {insight.get('title', 'Insight')}", ""])
        if "description" in insight:
            table_data.append([f"  {insight['description'][:50]}...", ""])
    
    labels = [row[0] for row in table_data]
    values = [row[1] for row in table_data]
    
    fig.add_trace(
        go.Table(
            cells=dict(
                values=[labels],
                align='left',
                font=dict(size=10),
                height=20
            )
        ),
        row=row, col=col
    )

# Import scipy for statistics
import scipy.stats as stats

def add_quality_timeline_simple(fig: go.Figure, df: pd.DataFrame, quality_config: Dict[str, Any], row: int, col: int):
    """Add simple quality timeline."""
    if "quality_score" not in df.columns or df["quality_score"].isna().all():
        return
    
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["quality_score"],
            mode='lines+markers',
            name='Quality Score',
            line=dict(color='green', width=2),
            marker=dict(size=4),
            hovertemplate="%{x|%Y-%m-%d}<br>Quality: %{y:.2f}<extra></extra>"
        ),
        row=row, col=col
    )
    
    fig.add_hline(y=0.6, line_dash="dash", line_color="red", annotation_text="Threshold", row=row, col=col)
    fig.update_xaxes(title_text="Date", row=row, col=col)
    fig.update_yaxes(title_text="Quality Score", range=[0, 1], row=row, col=col)

def add_component_breakdown_simple(fig: go.Figure, df: pd.DataFrame, row: int, col: int):
    """Add simple component breakdown."""
    components = ["safety", "plausibility", "consistency", "reliability"]
    colors = ["green", "blue", "orange", "purple"]
    
    means = []
    for comp in components:
        if comp in df.columns and not df[comp].isna().all():
            means.append(df[comp].mean())
        else:
            means.append(0)
    
    if any(means):
        fig.add_trace(
            go.Bar(
                x=components,
                y=means,
                name='Component Scores',
                marker_color=colors,
                text=[f"{m:.2f}" for m in means],
                textposition='auto',
                hovertemplate="<b>%{x}</b><br>Score: %{y:.2f}<extra></extra>"
            ),
            row=row, col=col
        )
        
        fig.add_hline(y=0.6, line_dash="dash", line_color="red", row=row, col=col)
        fig.update_xaxes(title_text="Component", row=row, col=col)
        fig.update_yaxes(title_text="Average Score", range=[0, 1], row=row, col=col)