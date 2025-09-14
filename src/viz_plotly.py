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

def create_interactive_dashboard(results: List[Dict[str, Any]], 
                                user_id: str,
                                output_dir: str = "output",
                                config: Optional[Dict[str, Any]] = None) -> str:
    
    df = extract_quality_data(results)
    if df.empty:
        logger.debug(f"No data to visualize for user {user_id}")
        return None
    
    stats = calculate_quality_statistics(df)
    patterns = detect_quality_patterns(df)
    insights = generate_quality_insights(df, stats, patterns)
    rejection_analysis = analyze_rejection_reasons(df)
    source_analysis = analyze_source_quality(df)
    trends_df = calculate_quality_trends(df)
    
    viz_config = config.get("visualization", {}) if config else {}
    interactive_config = viz_config.get("interactive", {})
    quality_config = viz_config.get("quality", {})
    
    fig = create_dashboard_layout(
        df, trends_df, stats, insights, 
        rejection_analysis, source_analysis,
        viz_config, interactive_config, quality_config
    )
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    html_file = output_path / f"dashboard_{user_id}.html"
    
    fig.write_html(
        str(html_file),
        include_plotlyjs='directory',
        config={
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
            'toImageButtonOptions': {
                'format': 'png',
                'filename': f'dashboard_{user_id}',
                'height': 1080,
                'width': 1920,
                'scale': 2
            }
        }
    )
    
    logger.success(f"✓ Standard: {html_file.name}")
    return str(html_file)

def create_dashboard_layout(df: pd.DataFrame,
                           trends_df: pd.DataFrame,
                           stats: Dict[str, Any],
                           insights: List[Dict[str, Any]],
                           rejection_analysis: Dict[str, Any],
                           source_analysis: Dict[str, Any],
                           viz_config: Dict[str, Any],
                           interactive_config: Dict[str, Any],
                           quality_config: Dict[str, Any]) -> go.Figure:
    
    theme = viz_config.get("theme", "plotly_white")
    
    fig = make_subplots(
        rows=3, cols=4,
        row_heights=[0.4, 0.3, 0.3],
        column_widths=[0.25, 0.25, 0.25, 0.25],
        subplot_titles=(
            "Weight Measurements with Quality Overlay", None, None, "Statistics",
            "Quality Score Timeline", "Component Breakdown", "Quality Distribution", "Insights",
            "Rejection Analysis", None, "Source Quality Comparison", None
        ),
        specs=[
            [{"colspan": 3}, None, None, {"type": "table", "rowspan": 3}],
            [{"type": "scatter"}, {"type": "bar"}, {"type": "histogram"}, None],
            [{"colspan": 2, "type": "bar"}, None, {"colspan": 2, "type": "sunburst"}, None]
        ],
        horizontal_spacing=0.08,
        vertical_spacing=0.12
    )
    
    add_main_time_series(fig, df, quality_config, row=1, col=1)
    
    add_quality_timeline(fig, trends_df, quality_config, row=2, col=1)
    
    add_component_breakdown(fig, df, row=2, col=2)
    
    add_quality_distribution(fig, df, quality_config, row=2, col=3)
    
    add_statistics_table(fig, stats, insights, row=1, col=4)
    
    add_rejection_analysis(fig, df, rejection_analysis, row=3, col=1)
    
    add_source_quality_comparison(fig, source_analysis, row=3, col=3)
    
    fig.update_layout(
        title={
            'text': f"Weight Stream Analytics Dashboard",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24}
        },
        template=theme,
        height=900,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5
        ),
        hovermode='x unified'
    )
    
    return fig

def add_main_time_series(fig: go.Figure, df: pd.DataFrame, quality_config: Dict[str, Any], row: int, col: int):
    
    accepted_df = df[df["accepted"] == True]
    rejected_df = df[df["accepted"] == False]
    
    has_quality_scores = "quality_score" in accepted_df.columns and not accepted_df["quality_score"].isna().all()
    
    if quality_config.get("show_quality_scores", True) and has_quality_scores:
        colorscale = quality_config.get("quality_color_scheme", "RdYlGn")
        
        fig.add_trace(
            go.Scatter(
                x=accepted_df["timestamp"],
                y=accepted_df["weight"],
                mode='markers+lines',
                name='Accepted',
                marker=dict(
                    size=8,
                    color=accepted_df["quality_score"],
                    colorscale=colorscale,
                    cmin=0,
                    cmax=1,
                    colorbar=dict(
                        title="Quality<br>Score",
                        thickness=15,
                        len=0.3,
                        y=0.85,
                        x=0.73
                    ),
                    line=dict(width=1, color='DarkSlateGrey'),
                    showscale=True
                ),
                line=dict(width=1, color='lightgray'),
                customdata=accepted_df[["quality_score", "safety", "plausibility", "consistency", "reliability", "source"]].values,
                hovertemplate=(
                    "<b>%{x|%Y-%m-%d %H:%M}</b><br>" +
                    "Weight: <b>%{y:.1f} kg</b><br>" +
                    "Quality: <b>%{customdata[0]:.2f}</b><br>" +
                    "Safety: %{customdata[1]:.2f}<br>" +
                    "Plausibility: %{customdata[2]:.2f}<br>" +
                    "Consistency: %{customdata[3]:.2f}<br>" +
                    "Reliability: %{customdata[4]:.2f}<br>" +
                    "Source: %{customdata[5]}<extra></extra>"
                )
            ),
            row=row, col=col
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=accepted_df["timestamp"],
                y=accepted_df["weight"],
                mode='markers+lines',
                name='Accepted',
                marker=dict(size=6, color='green'),
                line=dict(width=1, color='lightgreen')
            ),
            row=row, col=col
        )
    
    if not rejected_df.empty:
        fig.add_trace(
            go.Scatter(
                x=rejected_df["timestamp"],
                y=rejected_df["weight"],
                mode='markers',
                name='Rejected',
                marker=dict(
                    size=8,
                    color='red',
                    symbol='x',
                    line=dict(width=2, color='darkred')
                ),
                customdata=rejected_df[["rejection_reason", "quality_score"]].values,
                hovertemplate=(
                    "<b>%{x|%Y-%m-%d %H:%M}</b><br>" +
                    "Weight: <b>%{y:.1f} kg</b><br>" +
                    "Status: <b>Rejected</b><br>" +
                    "Reason: %{customdata[0]}<br>" +
                    "Quality: %{customdata[1]:.2f}<extra></extra>"
                )
            ),
            row=row, col=col
        )
    
    if quality_config.get("highlight_threshold_zone", True):
        threshold = 0.6
        buffer = quality_config.get("threshold_buffer", 0.05)
        
        y_min = df["weight"].min() if not df.empty else 0
        y_max = df["weight"].max() if not df.empty else 100
        
        fig.add_hrect(
            y0=y_min, y1=y_max,
            x0=df["timestamp"].min(), x1=df["timestamp"].max(),
            fillcolor="red", opacity=0.05,
            layer="below", line_width=0,
            annotation_text="Quality Threshold Zone",
            annotation_position="top right",
            row=row, col=col
        )
    
    fig.update_xaxes(title_text="Date", row=row, col=col)
    fig.update_yaxes(title_text="Weight (kg)", row=row, col=col)

def add_quality_timeline(fig: go.Figure, trends_df: pd.DataFrame, quality_config: Dict[str, Any], row: int, col: int):
    
    if trends_df.empty or "quality_ma" not in trends_df.columns:
        return
    
    fig.add_trace(
        go.Scatter(
            x=trends_df["timestamp"],
            y=trends_df["quality_score"],
            mode='markers',
            name='Quality Score',
            marker=dict(size=3, color='lightgray', opacity=0.5),
            showlegend=False
        ),
        row=row, col=col
    )
    
    fig.add_trace(
        go.Scatter(
            x=trends_df["timestamp"],
            y=trends_df["quality_ma"],
            mode='lines',
            name='7-day Average',
            line=dict(width=2, color='blue')
        ),
        row=row, col=col
    )
    
    fig.add_hline(
        y=0.6,
        line_dash="dash",
        line_color="red",
        annotation_text="Threshold",
        row=row, col=col
    )
    
    if quality_config.get("highlight_threshold_zone", True):
        fig.add_hrect(
            y0=0.55, y1=0.65,
            fillcolor="orange", opacity=0.1,
            layer="below", line_width=0,
            row=row, col=col
        )
    
    fig.update_xaxes(title_text="Date", row=row, col=col)
    fig.update_yaxes(title_text="Quality Score", range=[0, 1], row=row, col=col)

def add_component_breakdown(fig: go.Figure, df: pd.DataFrame, row: int, col: int):
    
    components = ["safety", "plausibility", "consistency", "reliability"]
    colors = ["green", "blue", "orange", "purple"]
    
    means = []
    for comp in components:
        if comp in df.columns and not df[comp].isna().all():
            means.append(df[comp].mean())
        else:
            means.append(0)
    
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
    
    fig.add_hline(
        y=0.6,
        line_dash="dash",
        line_color="red",
        row=row, col=col
    )
    
    fig.update_xaxes(title_text="Component", row=row, col=col)
    fig.update_yaxes(title_text="Average Score", range=[0, 1], row=row, col=col)

def add_quality_distribution(fig: go.Figure, df: pd.DataFrame, quality_config: Dict[str, Any], row: int, col: int):
    
    if "quality_score" not in df.columns or df["quality_score"].isna().all():
        return
    
    accepted_quality = df[df["accepted"] == True]["quality_score"].dropna()
    rejected_quality = df[df["accepted"] == False]["quality_score"].dropna()
    
    fig.add_trace(
        go.Histogram(
            x=accepted_quality,
            name='Accepted',
            marker_color='green',
            opacity=0.7,
            nbinsx=20,
            hovertemplate="Quality: %{x:.2f}<br>Count: %{y}<extra></extra>"
        ),
        row=row, col=col
    )
    
    if not rejected_quality.empty:
        fig.add_trace(
            go.Histogram(
                x=rejected_quality,
                name='Rejected',
                marker_color='red',
                opacity=0.7,
                nbinsx=20,
                hovertemplate="Quality: %{x:.2f}<br>Count: %{y}<extra></extra>"
            ),
            row=row, col=col
        )
    
    fig.add_vline(
        x=0.6,
        line_dash="dash",
        line_color="red",
        annotation_text="Threshold",
        row=row, col=col
    )
    
    fig.update_xaxes(title_text="Quality Score", range=[0, 1], row=row, col=col)
    fig.update_yaxes(title_text="Count", row=row, col=col)

def add_statistics_table(fig: go.Figure, stats: Dict[str, Any], insights: List[Dict[str, Any]], row: int, col: int):
    
    def format_value(value):
        if value is None:
            return "N/A"
        elif isinstance(value, float):
            return f"{value:.2f}"
        elif isinstance(value, int):
            return str(value)
        else:
            return str(value)
    
    table_data = []
    
    table_data.append(["<b>OVERALL METRICS</b>", ""])
    table_data.append(["Total Measurements", format_value(stats.get("total_measurements", 0))])
    table_data.append(["Accepted", format_value(stats.get("accepted_count", 0))])
    table_data.append(["Rejected", format_value(stats.get("rejected_count", 0))])
    table_data.append(["Acceptance Rate", f"{stats.get('acceptance_rate', 0):.1f}%"])
    
    table_data.append(["", ""])
    table_data.append(["<b>QUALITY SCORES</b>", ""])
    table_data.append(["Overall Quality", format_value(stats.get("overall_quality"))])
    table_data.append(["Quality Std Dev", format_value(stats.get("quality_std"))])
    
    if "quality_percentiles" in stats:
        table_data.append(["25th Percentile", format_value(stats["quality_percentiles"]["p25"])])
        table_data.append(["Median", format_value(stats["quality_percentiles"]["p50"])])
        table_data.append(["75th Percentile", format_value(stats["quality_percentiles"]["p75"])])
    
    table_data.append(["", ""])
    table_data.append(["<b>COMPONENTS</b>", ""])
    for component in ["safety", "plausibility", "consistency", "reliability"]:
        if f"{component}_mean" in stats:
            table_data.append([component.capitalize(), format_value(stats[f"{component}_mean"])])
    
    if insights:
        table_data.append(["", ""])
        table_data.append(["<b>TOP INSIGHTS</b>", ""])
        for i, insight in enumerate(insights[:3], 1):
            icon = {"alert": "⚠️", "warning": "⚡", "info": "ℹ️"}.get(insight["type"], "•")
            table_data.append([f"{icon} {insight['title']}", ""])
    
    labels = [row[0] for row in table_data]
    values = [row[1] for row in table_data]
    
    colors = []
    for label in labels:
        if "<b>" in label:
            colors.append("lightblue")
        elif any(icon in label for icon in ["⚠️", "⚡", "ℹ️"]):
            colors.append("lightyellow")
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

def add_rejection_analysis(fig: go.Figure, df: pd.DataFrame, rejection_analysis: Dict[str, Any], row: int, col: int):
    
    if not rejection_analysis.get("reasons"):
        fig.add_annotation(
            text="No rejections to analyze",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            row=row, col=col
        )
        return
    
    reasons = list(rejection_analysis["reasons"].keys())
    counts = [rejection_analysis["reasons"][r]["count"] for r in reasons]
    avg_qualities = [rejection_analysis["reasons"][r]["avg_quality"] for r in reasons]
    
    fig.add_trace(
        go.Bar(
            x=reasons,
            y=counts,
            name='Rejection Count',
            marker=dict(
                color=avg_qualities,
                colorscale='RdYlGn',
                cmin=0,
                cmax=1,
                colorbar=dict(
                    title="Avg<br>Quality",
                    thickness=10,
                    len=0.3,
                    x=0.45
                )
            ),
            text=[f"{c}<br>Q:{q:.2f}" for c, q in zip(counts, avg_qualities)],
            textposition='auto',
            hovertemplate=(
                "<b>%{x}</b><br>" +
                "Count: %{y}<br>" +
                "Avg Quality: %{marker.color:.2f}<extra></extra>"
            )
        ),
        row=row, col=col
    )
    
    fig.update_xaxes(title_text="Rejection Reason", row=row, col=col)
    fig.update_yaxes(title_text="Count", row=row, col=col)

def add_source_quality_comparison(fig: go.Figure, source_analysis: Dict[str, Any], row: int, col: int):
    
    if not source_analysis:
        return
    
    sources = []
    parents = []
    values = []
    colors = []
    
    for source, data in source_analysis.items():
        sources.append(source)
        parents.append("")
        values.append(data["count"])
        colors.append(data.get("avg_quality", 0.5))
        
        for component in ["safety", "plausibility", "consistency", "reliability"]:
            comp_key = f"{component}_mean"
            if comp_key in data:
                sources.append(f"{source}-{component}")
                parents.append(source)
                values.append(data["count"] / 4)
                colors.append(data[comp_key])
    
    fig.add_trace(
        go.Sunburst(
            labels=sources,
            parents=parents,
            values=values,
            branchvalues="total",
            marker=dict(
                colors=colors,
                colorscale='RdYlGn',
                cmin=0,
                cmax=1,
                colorbar=dict(
                    title="Quality",
                    thickness=10,
                    len=0.3
                )
            ),
            hovertemplate=(
                "<b>%{label}</b><br>" +
                "Count: %{value}<br>" +
                "Quality: %{color:.2f}<extra></extra>"
            ),
            textinfo="label+percent parent"
        ),
        row=row, col=col
    )