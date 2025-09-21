"""
Weight timeline visualization with quality insights and Kalman filter details.
"""

import csv
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from ..constants import (
        REJECTION_SEVERITY_COLORS,
        SOURCE_MARKER_SYMBOLS,
        get_rejection_severity,
    )
except ImportError:
    from src.constants import (
        REJECTION_SEVERITY_COLORS,
        SOURCE_MARKER_SYMBOLS,
        get_rejection_severity,
    )


def load_user_employer_start_date(
    user_id: str, employer_file: str = "data/2025-09-17-user-employers.csv"
) -> Optional[datetime]:
    """
    Load the start date for a specific user from the employer CSV file.

    Args:
        user_id: User identifier to look up
        employer_file: Path to the employer CSV file

    Returns:
        Start date as datetime if found, None otherwise
    """
    employer_path = Path(employer_file)
    if not employer_path.exists():
        return None

    try:
        with open(employer_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["user_id"] == user_id:
                    # Parse the start_date
                    start_date_str = row["start_date"]
                    # Remove quotes if present
                    start_date_str = start_date_str.strip('"')
                    # Parse the datetime (format: "2025-09-12 14:38:30.911")
                    return datetime.strptime(
                        start_date_str.split(".")[0], "%Y-%m-%d %H:%M:%S"
                    )
    except Exception as e:
        print(f"Warning: Could not load employer start date for user {user_id}: {e}")

    return None


def create_weight_timeline(
    results: List[Dict[str, Any]],
    user_id: str,
    output_dir: str = "output",
    config: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Create an interactive weight timeline with quality analysis and Kalman insights.

    Args:
        results: List of measurement results
        user_id: User identifier
        output_dir: Output directory for HTML file
        config: Optional visualization configuration

    Returns:
        Path to generated HTML file
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly is required for visualization")

    if not results:
        raise ValueError("No results to visualize")

    # Default configuration
    viz_config = config or {}
    colors = viz_config.get(
        "colors",
        {
            "raw_accepted": "#4CAF50",
            "raw_rejected": "#F44336",
            "kalman_line": "#2196F3",
            "confidence_band": "rgba(33, 150, 243, 0.2)",
            "quality_safety": "#4CAF50",
            "quality_plausibility": "#2196F3",
            "quality_consistency": "#FF9800",
            "quality_reliability": "#9C27B0",
        },
    )

    # Process data
    accepted_data = []
    rejected_data = []
    reset_events = []

    for r in results:
        timestamp = r.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)

        source = r.get("source", "unknown")
        marker_symbol = SOURCE_MARKER_SYMBOLS.get(
            source, SOURCE_MARKER_SYMBOLS["default"]
        )

        # Check for reset events
        if r.get("was_reset", False):
            reset_events.append(
                {
                    "timestamp": timestamp,
                    "reason": r.get("reset_reason", "Unknown"),
                    "type": r.get("reset_type", "unknown"),
                    "gap_days": r.get("gap_days"),
                }
            )

        # Extract data
        data_point = {
            "timestamp": timestamp,
            "raw_weight": r.get("raw_weight", r.get("weight")),
            "filtered_weight": r.get("filtered_weight", r.get("kalman_estimate")),
            "source": source,
            "marker_symbol": marker_symbol,
            "confidence": r.get("confidence", 0),
            "innovation": r.get("innovation", 0),
            "normalized_innovation": r.get("normalized_innovation", 0),
            "kalman_upper": r.get("kalman_confidence_upper"),
            "kalman_lower": r.get("kalman_confidence_lower"),
            "kalman_variance": r.get("kalman_variance"),
            "trend": r.get("trend", 0),
            "trend_weekly": r.get("trend_weekly", 0),
            "was_reset": r.get("was_reset", False),
            "reset_reason": r.get("reset_reason"),
            "reset_type": r.get("reset_type"),
            "gap_days": r.get("gap_days"),
        }

        # Extract quality components
        if "quality_components" in r:
            data_point["quality_components"] = r["quality_components"]
            data_point["quality_score"] = r.get("quality_score", 0)
        elif "quality_score" in r:
            # Handle legacy format
            if isinstance(r["quality_score"], dict):
                data_point["quality_components"] = r["quality_score"].get(
                    "components", {}
                )
                data_point["quality_score"] = r["quality_score"].get("overall", 0)
            else:
                data_point["quality_score"] = r["quality_score"]
                data_point["quality_components"] = {}
        else:
            data_point["quality_score"] = 0
            data_point["quality_components"] = {}

        if r.get("accepted", False):
            data_point["hover_text"] = _create_accepted_hover(r, data_point)
            accepted_data.append(data_point)
        else:
            reason = r.get("reason", "Unknown")
            weight_change = abs(
                r.get("raw_weight", 0)
                - r.get("filtered_weight", r.get("raw_weight", 0))
            )
            severity = get_rejection_severity(reason, weight_change)

            data_point["reason"] = reason
            data_point["severity"] = severity
            data_point["hover_text"] = _create_rejected_hover(r, data_point)
            rejected_data.append(data_point)

    # Create subplots with shared x-axes
    fig = make_subplots(
        rows=3,
        cols=1,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=(
            f"Weight Measurements - {user_id}",
            "Quality Score Components",
            "Innovation / Residuals",
        ),
        vertical_spacing=0.08,
        specs=[
            [{"secondary_y": False}],
            [{"secondary_y": False}],
            [{"secondary_y": False}],
        ],
        shared_xaxes=True,  # This links all x-axes together
    )

    # Main weight timeline (Row 1)
    _add_weight_traces(fig, accepted_data, rejected_data, colors, row=1)

    # Quality components (Row 2)
    _add_quality_traces(fig, accepted_data, rejected_data, colors, row=2)

    # Innovation/Residuals (Row 3)
    _add_innovation_traces(fig, accepted_data, rejected_data, colors, row=3)

    # Add reset event vertical lines across all subplots (subtle, behind data)
    if reset_events:
        for reset in reset_events:
            # Add vertical line shape that spans all subplots
            fig.add_shape(
                type="line",
                x0=reset["timestamp"],
                x1=reset["timestamp"],
                y0=0,
                y1=1,
                yref="paper",
                line=dict(
                    color="rgba(255, 102, 0, 0.2)",  # Very transparent orange
                    width=1,  # Thin line
                    dash="dot",  # Dotted line
                ),
                layer="below",  # Place behind the data
            )

            # Add subtle annotation at the top
            gap_days = reset.get("gap_days")
            if gap_days is not None:
                gap_text = f"{gap_days:.0f}d"
            else:
                gap_text = "Reset"

            fig.add_annotation(
                x=reset["timestamp"],
                y=1.01,
                yref="paper",
                text=gap_text,
                showarrow=False,
                font=dict(size=8, color="rgba(255, 102, 0, 0.5)"),
                bgcolor="rgba(255, 255, 255, 0.7)",
                bordercolor="rgba(255, 102, 0, 0.2)",
                borderwidth=0.5,
                align="center",
            )

    # Add employer start date vertical line if available
    employer_start_date = load_user_employer_start_date(user_id)
    if employer_start_date:
        # Add vertical line shape that spans all subplots
        fig.add_shape(
            type="line",
            x0=employer_start_date,
            x1=employer_start_date,
            y0=0,
            y1=1,
            yref="paper",
            line=dict(
                color="rgba(0, 128, 255, 0.5)",  # Semi-transparent blue
                width=2,  # Slightly thicker line
                dash="dashdot",  # Dash-dot line style
            ),
            layer="below",  # Place behind the data
        )

        # Add annotation for employer start date
        fig.add_annotation(
            x=employer_start_date,
            y=1.02,
            yref="paper",
            text="Start Date",
            showarrow=False,
            font=dict(size=10, color="rgba(0, 128, 255, 0.7)"),
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="rgba(0, 128, 255, 0.5)",
            borderwidth=1,
            borderpad=2,
            align="center",
        )

        # Add 90-day mark after start date
        from datetime import timedelta

        ninety_day_mark = employer_start_date + timedelta(days=90)

        # Add vertical line for 90-day mark
        fig.add_shape(
            type="line",
            x0=ninety_day_mark,
            x1=ninety_day_mark,
            y0=0,
            y1=1,
            yref="paper",
            line=dict(
                color="rgba(0, 200, 0, 0.5)",  # Semi-transparent green
                width=2,  # Slightly thicker line
                dash="dash",  # Dashed line style
            ),
            layer="below",  # Place behind the data
        )

        # Add annotation for 90-day mark
        fig.add_annotation(
            x=ninety_day_mark,
            y=1.02,
            yref="paper",
            text="90 Days",
            showarrow=False,
            font=dict(size=10, color="rgba(0, 200, 0, 0.7)"),
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="rgba(0, 200, 0, 0.5)",
            borderwidth=1,
            borderpad=2,
            align="center",
        )

    # Add invisible traces for source type legend
    source_types = {
        "care-team-upload": ("Care Team", "triangle-up"),
        "patient-upload": ("Patient Upload", "circle"),
        "internal-questionnaire": ("Questionnaire", "square"),
        "patient-device": ("Device", "diamond"),
        "https://connectivehealth.io": ("ConnectiveHealth", "hexagon"),
        "https://api.iglucose.com": ("iGlucose", "hexagon"),
    }

    # Add legend entries for source types (only to first subplot)
    for source_key, (display_name, symbol) in source_types.items():
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name=f"{display_name}",
                marker=dict(
                    size=10,
                    symbol=symbol,
                    color="gray",
                    line=dict(color="darkgray", width=1),
                ),
                showlegend=True,
                legendgroup="sources",
                legendgrouptitle_text="Source Types",
            ),
            row=1,
            col=1,
        )

    # Update layout
    stats = _calculate_stats(results, accepted_data, rejected_data)

    fig.update_layout(
        title={
            "text": f"Enhanced Weight Analysis - {user_id}<br><sub>{stats}</sub>",
            "x": 0.5,
            "xanchor": "center",
        },
        height=900,
        hovermode="closest",
        showlegend=True,
        legend=dict(
            x=1.02,
            y=1,
            bgcolor="rgba(255, 255, 255, 0.95)",
            bordercolor="rgba(0, 0, 0, 0.3)",
            borderwidth=2,
            font=dict(size=9),
            itemsizing="constant",
            tracegroupgap=15,
            groupclick="toggleitem",
        ),
        plot_bgcolor="#FAFAFA",
        paper_bgcolor="white",
        updatemenus=[_create_visibility_menu()],
    )

    # Update axes - x-axes are shared so zooming one affects all
    # Add range selector buttons to the top subplot for easy time navigation
    fig.update_xaxes(
        showgrid=True,
        gridcolor="#E0E0E0",
        rangeselector=dict(
            buttons=list(
                [
                    dict(count=7, label="1w", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all", label="All"),
                ]
            ),
            x=0.01,
            y=1.05,
        ),
        rangeslider=dict(visible=False),  # Hide on first subplot
        row=1,
        col=1,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#E0E0E0", row=2, col=1)
    # Add range slider to bottom subplot for fine control
    fig.update_xaxes(
        showgrid=True,
        gridcolor="#E0E0E0",
        title_text="Date",
        rangeslider=dict(visible=True, thickness=0.05),
        row=3,
        col=1,
    )

    fig.update_yaxes(
        title_text="Weight (kg)", showgrid=True, gridcolor="#E0E0E0", row=1, col=1
    )
    fig.update_yaxes(
        title_text="Quality Score",
        range=[0, 1.1],
        showgrid=True,
        gridcolor="#E0E0E0",
        row=2,
        col=1,
    )
    fig.update_yaxes(
        title_text="Innovation (kg)", showgrid=True, gridcolor="#E0E0E0", row=3, col=1
    )

    # Save HTML
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    html_file = output_path / f"{user_id}_timeline.html"

    config = {
        "displayModeBar": True,
        "displaylogo": False,
        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        "toImageButtonOptions": {
            "format": "png",
            "filename": f"weight_timeline_{user_id}",
            "height": 900,
            "width": 1400,
            "scale": 2,
        },
    }

    fig.write_html(str(html_file), config=config)

    return str(html_file)


def _add_weight_traces(
    fig, accepted_data: List[Dict], rejected_data: List[Dict], colors: Dict, row: int
):
    """Add weight measurement traces to the main plot."""

    # Add rejected measurements first (so they appear below accepted in z-order)
    if rejected_data:
        severity_groups = {}
        for d in rejected_data:
            severity = d.get("severity", "Low")
            if severity not in severity_groups:
                severity_groups[severity] = []
            severity_groups[severity].append(d)

        for severity, data_points in severity_groups.items():
            fig.add_trace(
                go.Scatter(
                    x=[d["timestamp"] for d in data_points],
                    y=[d["raw_weight"] for d in data_points],
                    mode="markers",
                    name=f"Rejected ({severity})",
                    marker=dict(
                        size=6,
                        color=REJECTION_SEVERITY_COLORS.get(severity, "#FF0000"),
                        symbol=[d["marker_symbol"] for d in data_points],
                        line=dict(color="#8B0000", width=1),
                    ),
                    text=[d["hover_text"] for d in data_points],
                    hovertemplate="%{text}<extra></extra>",
                    legendgroup="rejected",
                ),
                row=row,
                col=1,
            )

    # Add Kalman filter components (confidence bands and line)
    if accepted_data:
        # Confidence bands first (so they appear behind the Kalman line)
        timestamps = [
            d["timestamp"] for d in accepted_data if d.get("kalman_upper") is not None
        ]
        upper_bounds = [
            d["kalman_upper"]
            for d in accepted_data
            if d.get("kalman_upper") is not None
        ]
        lower_bounds = [
            d["kalman_lower"]
            for d in accepted_data
            if d.get("kalman_lower") is not None
        ]

        if timestamps:
            # Upper bound
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=upper_bounds,
                    mode="lines",
                    name="Upper Confidence",
                    line=dict(width=0),
                    showlegend=False,
                    legendgroup="confidence",
                ),
                row=row,
                col=1,
            )

            # Lower bound with fill
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=lower_bounds,
                    mode="lines",
                    name="Confidence Band (±2σ)",
                    line=dict(width=0),
                    fillcolor=colors["confidence_band"],
                    fill="tonexty",
                    legendgroup="confidence",
                ),
                row=row,
                col=1,
            )

        # Kalman filtered line
        fig.add_trace(
            go.Scatter(
                x=[d["timestamp"] for d in accepted_data],
                y=[
                    d["filtered_weight"]
                    for d in accepted_data
                    if d["filtered_weight"] is not None
                ],
                mode="lines",
                name="Kalman Filtered",
                line=dict(color=colors["kalman_line"], width=2),
                legendgroup="kalman",
            ),
            row=row,
            col=1,
        )

        # Raw accepted measurements last (so they appear on top)
        fig.add_trace(
            go.Scatter(
                x=[d["timestamp"] for d in accepted_data],
                y=[d["raw_weight"] for d in accepted_data],
                mode="markers",
                name="Raw (Accepted)",
                marker=dict(
                    size=10,
                    color=[
                        "#000000" if d["was_reset"] else d["quality_score"]
                        for d in accepted_data
                    ],
                    colorscale="Viridis",
                    cmin=0,
                    cmax=1,
                    showscale=True,
                    colorbar=dict(title="Quality<br>Score", x=1.15, y=0.85, len=0.3),
                    symbol=[d["marker_symbol"] for d in accepted_data],
                    line=dict(color="white", width=1),
                ),
                text=[d["hover_text"] for d in accepted_data],
                hovertemplate="%{text}<extra></extra>",
                legendgroup="raw",
            ),
            row=row,
            col=1,
        )


def _add_quality_traces(
    fig, accepted_data: List[Dict], rejected_data: List[Dict], colors: Dict, row: int
):
    """Add quality score component traces."""

    all_data = accepted_data + rejected_data
    all_data.sort(key=lambda x: x["timestamp"])

    if not all_data:
        return

    timestamps = [d["timestamp"] for d in all_data]

    # Extract component scores
    safety_scores = []
    plausibility_scores = []
    consistency_scores = []
    reliability_scores = []
    overall_scores = []

    for d in all_data:
        components = d.get("quality_components", {})
        safety_scores.append(components.get("safety", 0))
        plausibility_scores.append(components.get("plausibility", 0))
        consistency_scores.append(components.get("consistency", 0))
        reliability_scores.append(components.get("reliability", 0))
        overall_scores.append(d.get("quality_score", 0))

    # Add stacked area chart for components
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=safety_scores,
            mode="lines",
            name="Safety",
            line=dict(width=0.5, color=colors["quality_safety"]),
            stackgroup="quality",
            fillcolor=colors["quality_safety"],
            legendgroup="quality",
        ),
        row=row,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=plausibility_scores,
            mode="lines",
            name="Plausibility",
            line=dict(width=0.5, color=colors["quality_plausibility"]),
            stackgroup="quality",
            fillcolor=colors["quality_plausibility"],
            legendgroup="quality",
        ),
        row=row,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=consistency_scores,
            mode="lines",
            name="Consistency",
            line=dict(width=0.5, color=colors["quality_consistency"]),
            stackgroup="quality",
            fillcolor=colors["quality_consistency"],
            legendgroup="quality",
        ),
        row=row,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=reliability_scores,
            mode="lines",
            name="Reliability",
            line=dict(width=0.5, color=colors["quality_reliability"]),
            stackgroup="quality",
            fillcolor=colors["quality_reliability"],
            legendgroup="quality",
        ),
        row=row,
        col=1,
    )

    # Add overall score line
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=overall_scores,
            mode="lines+markers",
            name="Overall Score",
            line=dict(color="black", width=2, dash="dot"),
            marker=dict(size=4),
            legendgroup="quality",
        ),
        row=row,
        col=1,
    )

    # Add threshold line
    fig.add_trace(
        go.Scatter(
            x=[timestamps[0], timestamps[-1]],
            y=[0.6, 0.6],
            mode="lines",
            name="Acceptance Threshold",
            line=dict(color="red", width=1, dash="dash"),
            legendgroup="quality",
        ),
        row=row,
        col=1,
    )


def _add_innovation_traces(
    fig, accepted_data: List[Dict], rejected_data: List[Dict], colors: Dict, row: int
):
    """Add innovation/residual traces."""

    if accepted_data:
        innovations = [d["innovation"] for d in accepted_data]
        timestamps = [d["timestamp"] for d in accepted_data]
        normalized = [d["normalized_innovation"] for d in accepted_data]

        # Color based on normalized innovation
        bar_colors = []
        for n in normalized:
            if n <= 1:
                bar_colors.append("green")
            elif n <= 2:
                bar_colors.append("yellow")
            elif n <= 3:
                bar_colors.append("orange")
            else:
                bar_colors.append("red")

        fig.add_trace(
            go.Bar(
                x=timestamps,
                y=innovations,
                name="Innovation",
                marker=dict(color=bar_colors),
                text=[
                    f"{i:.2f}kg<br>({n:.1f}σ)" for i, n in zip(innovations, normalized)
                ],
                textposition="auto",
                legendgroup="innovation",
            ),
            row=row,
            col=1,
        )

        # Add reference lines
        if timestamps:
            # Find y-axis range
            max_innovation = (
                max(abs(min(innovations)), abs(max(innovations))) if innovations else 5
            )

            # Add standard deviation reference lines
            for sigma, label, color in [
                (1, "±1σ", "green"),
                (2, "±2σ", "orange"),
                (3, "±3σ", "red"),
            ]:
                # Estimate sigma from data (rough approximation)
                estimated_sigma = max_innovation / 3
                y_val = sigma * estimated_sigma

                fig.add_trace(
                    go.Scatter(
                        x=[timestamps[0], timestamps[-1]],
                        y=[y_val, y_val],
                        mode="lines",
                        name=f"{label} ({y_val:.1f}kg)",
                        line=dict(color=color, width=1, dash="dot"),
                        legendgroup="innovation_ref",
                        showlegend=True,
                    ),
                    row=row,
                    col=1,
                )

                fig.add_trace(
                    go.Scatter(
                        x=[timestamps[0], timestamps[-1]],
                        y=[-y_val, -y_val],
                        mode="lines",
                        line=dict(color=color, width=1, dash="dot"),
                        legendgroup="innovation_ref",
                        showlegend=False,
                    ),
                    row=row,
                    col=1,
                )


def _create_accepted_hover(result: Dict[str, Any], data_point: Dict[str, Any]) -> str:
    """Create hover text for accepted measurements."""
    timestamp = data_point["timestamp"]

    lines = [
        f"<b>✓ ACCEPTED</b>",
        f"<b>Date:</b> {timestamp.strftime('%Y-%m-%d %H:%M')}",
        f"<b>Source:</b> {_format_source(data_point['source'])}",
    ]

    # Add reset information if applicable
    if data_point.get("was_reset", False):
        lines.extend(
            [
                "",
                f"<b>⚠️ KALMAN FILTER RESET</b>",
            ]
        )
        if data_point.get("reset_type"):
            lines.append(f"<b>Type:</b> {data_point['reset_type'].upper()}")
        if data_point.get("reset_reason"):
            lines.append(
                f"<b>Reason:</b> {data_point['reset_reason'].replace('_', ' ').title()}"
            )
        if data_point.get("gap_days"):
            lines.append(f"<b>Data Gap:</b> {data_point['gap_days']:.1f} days")
        lines.append(f"<b>Filter reinitialized from this measurement</b>")

    lines.extend(
        [
            "",
            "<b>Weight Data:</b>",
            f"  Raw: {data_point['raw_weight']:.2f} kg",
            f"  Filtered: {data_point.get('filtered_weight', 0):.2f} kg",
            f"  Trend: {data_point.get('trend_weekly', 0):.2f} kg/week",
            "",
            "<b>Kalman Filter:</b>",
            f"  Confidence: {data_point.get('confidence', 0):.1%}",
            f"  Innovation: {data_point.get('innovation', 0):.2f} kg",
            f"  Normalized: {data_point.get('normalized_innovation', 0):.2f}σ",
        ]
    )

    if data_point.get("kalman_upper") is not None:
        lines.extend(
            [
                f"  CI Upper: {data_point['kalman_upper']:.2f} kg",
                f"  CI Lower: {data_point['kalman_lower']:.2f} kg",
            ]
        )

    if data_point.get("quality_score") is not None:
        lines.extend(
            [
                "",
                f"<b>Quality Score:</b> {data_point['quality_score']:.2f}",
            ]
        )

        if data_point.get("quality_components"):
            lines.append("<b>Components:</b>")
            for key, value in data_point["quality_components"].items():
                indicator = "✓" if value >= 0.7 else "⚠" if value >= 0.4 else "✗"
                lines.append(f"  {indicator} {key.capitalize()}: {value:.2f}")

    return "<br>".join(lines)


def _create_rejected_hover(result: Dict[str, Any], data_point: Dict[str, Any]) -> str:
    """Create hover text for rejected measurements."""
    timestamp = data_point["timestamp"]

    lines = [
        f"<b>✗ REJECTED</b>",
        f"<b>Date:</b> {timestamp.strftime('%Y-%m-%d %H:%M')}",
        f"<b>Source:</b> {_format_source(data_point['source'])}",
        f"<b>Weight:</b> {data_point['raw_weight']:.2f} kg",
        f"<b>Reason:</b> {data_point.get('reason', 'Unknown')}",
        f"<b>Severity:</b> {data_point.get('severity', 'Unknown')}",
    ]


    if data_point.get("quality_score") is not None:
        lines.extend(
            [
                "",
                f"<b>Quality Score:</b> {data_point['quality_score']:.2f}",
            ]
        )

        if data_point.get("quality_components"):
            lines.append("<b>Components:</b>")
            for key, value in data_point["quality_components"].items():
                indicator = "✓" if value >= 0.7 else "⚠" if value >= 0.4 else "✗"
                lines.append(f"  {indicator} {key.capitalize()}: {value:.2f}")

    return "<br>".join(lines)


def _format_source(source: str) -> str:
    """Format source string for display."""
    source_map = {
        "care-team-upload": "Care Team",
        "patient-upload": "Patient Upload",
        "patient-device": "Patient Device",
        "internal-questionnaire": "Internal Questionnaire",
        "initial-questionnaire": "Initial Questionnaire",
        "https://connectivehealth.io": "ConnectiveHealth",
        "https://api.iglucose.com": "iGlucose",
    }
    return source_map.get(source, source)


def _calculate_stats(
    results: List[Dict[str, Any]],
    accepted_data: List[Dict[str, Any]],
    rejected_data: List[Dict[str, Any]],
) -> str:
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
        f"Rejected: {rejected}",
    ]

    if accepted_data:
        weights = [d["raw_weight"] for d in accepted_data]
        avg_weight = sum(weights) / len(weights)

        # Get latest Kalman state
        latest = accepted_data[-1]
        if latest.get("filtered_weight"):
            stats_parts.append(f"Current: {latest['filtered_weight']:.1f} kg")
        if latest.get("trend_weekly"):
            trend_sign = "+" if latest["trend_weekly"] > 0 else ""
            stats_parts.append(
                f"Trend: {trend_sign}{latest['trend_weekly']:.2f} kg/week"
            )

        # Average quality score
        quality_scores = [
            d["quality_score"]
            for d in accepted_data
            if d.get("quality_score") is not None
        ]
        if quality_scores:
            avg_quality = sum(quality_scores) / len(quality_scores)
            stats_parts.append(f"Avg Quality: {avg_quality:.2f}")

    return " | ".join(stats_parts)


def _create_visibility_menu():
    """Create dropdown menu for toggling trace visibility."""
    return dict(
        type="dropdown",
        direction="down",
        x=0.02,
        y=1.15,
        buttons=[
            dict(
                label="All Layers",
                method="update",
                args=[{"visible": [True] * 50}],  # Show all traces
            ),
            dict(
                label="Raw Data Only",
                method="update",
                args=[{"visible": [True if i < 3 else False for i in range(50)]}],
            ),
            dict(
                label="Kalman Only",
                method="update",
                args=[{"visible": [False, True, True, True, True] + [False] * 45}],
            ),
            dict(
                label="Quality Only",
                method="update",
                args=[{"visible": [False] * 5 + [True] * 6 + [False] * 39}],
            ),
            dict(
                label="Innovation Only",
                method="update",
                args=[{"visible": [False] * 11 + [True] * 10 + [False] * 29}],
            ),
        ],
    )
