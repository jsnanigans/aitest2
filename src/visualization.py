"""
Unified Visualization Module for Weight Stream Processor
Consolidates all visualization functionality into a single module
"""

import os
import sys
import json
import base64
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import defaultdict, deque
from io import BytesIO

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import toml
except ImportError:
    toml = None


# ============================================================================
# SECTION 1: Constants and Configuration
# ============================================================================

# Chart colors and styles
CHART_COLORS = {
    'accepted': '#2E7D32',
    'rejected': '#C62828', 
    'filtered': '#1976D2',
    'raw': '#757575',
    'trend': '#FF6F00',
    'confidence': '#7B1FA2',
    'innovation': '#00796B',
    'grid': '#E0E0E0',
    'background': '#FAFAFA'
}

# Source type styles for visualization
SOURCE_TYPE_STYLES = {
    'care-team-upload': {'color': '#2E7D32', 'marker': 'o', 'size': 8, 'label': 'Care Team'},
    'patient-upload': {'color': '#1976D2', 'marker': 's', 'size': 7, 'label': 'Patient Upload'},
    'patient-device': {'color': '#7B1FA2', 'marker': '^', 'size': 7, 'label': 'Patient Device'},
    'internal-questionnaire': {'color': '#FF6F00', 'marker': 'D', 'size': 6, 'label': 'Internal Quest'},
    'initial-questionnaire': {'color': '#FF8F00', 'marker': 'd', 'size': 6, 'label': 'Initial Quest'},
    'https://connectivehealth.io': {'color': '#00796B', 'marker': 'v', 'size': 6, 'label': 'ConnectiveHealth'},
    'https://api.iglucose.com': {'color': '#C62828', 'marker': 'x', 'size': 7, 'label': 'iGlucose'},
    'default': {'color': '#757575', 'marker': 'o', 'size': 6, 'label': 'Other'}
}

# Rejection category colors
REJECTION_CATEGORY_COLORS = {
    'BMI Value': '#8B4513',
    'Unit Convert': '#FF4500',
    'Physio Limit': '#DC143C',
    'Out of Bounds': '#FF6B6B',
    'Extreme Dev': '#FFA500',
    'High Variance': '#FFD700',
    'Sustained': '#9370DB',
    'Daily Fluct': '#87CEEB',
    'Quality Score': '#FF69B4',
    'Other': '#A9A9A9'
}

# Plotly configuration
PLOTLY_CONFIG = {
    'displayModeBar': True,
    'displaylogo': False,
    'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
    'toImageButtonOptions': {
        'format': 'png',
        'filename': 'weight_dashboard',
        'height': 800,
        'width': 1200,
        'scale': 2
    }
}

# Default dashboard configuration
DEFAULT_DASHBOARD_CONFIG = {
    'height': 800,
    'width': 1200,
    'show_rejected': True,
    'show_confidence': True,
    'show_innovation': True,
    'show_quality': True,
    'show_sources': True,
    'interactive': True
}


# ============================================================================
# SECTION 2: Utility Functions
# ============================================================================

def get_source_style(source: str) -> Dict:
    """Get visualization style for a data source."""
    return SOURCE_TYPE_STYLES.get(source, SOURCE_TYPE_STYLES['default'])


def get_rejection_color(category: str) -> str:
    """Get color for rejection category."""
    return REJECTION_CATEGORY_COLORS.get(category, REJECTION_CATEGORY_COLORS['Other'])


def categorize_rejection(reason: str) -> str:
    """Categorize rejection reason into clear, high-level category."""
    reason_lower = reason.lower()
    
    if "bmi" in reason_lower:
        return "BMI Value"
    elif "unit" in reason_lower or "pound" in reason_lower or "conversion" in reason_lower:
        return "Unit Convert"
    elif "physiological" in reason_lower:
        return "Physio Limit"
    elif "outside bounds" in reason_lower:
        return "Out of Bounds"
    elif "extreme deviation" in reason_lower:
        return "Extreme Dev"
    elif "session variance" in reason_lower or "different user" in reason_lower:
        return "High Variance"
    elif "sustained" in reason_lower:
        return "Sustained"
    elif "daily fluctuation" in reason_lower:
        return "Daily Fluct"
    elif "quality score" in reason_lower:
        return "Quality Score"
    else:
        return "Other"


def format_timestamp(ts: Union[str, datetime]) -> str:
    """Format timestamp for display."""
    if isinstance(ts, str):
        ts = datetime.fromisoformat(ts)
    return ts.strftime("%Y-%m-%d %H:%M")


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers."""
    return numerator / denominator if denominator != 0 else default


def load_config() -> Dict[str, Any]:
    """Load configuration from config.toml if available."""
    if toml is None:
        return {}
    
    config_path = Path(__file__).parent.parent / "config.toml"
    if config_path.exists():
        return toml.load(config_path)
    return {}


def should_use_interactive(config: Optional[Dict[str, Any]] = None, 
                          output_format: Optional[str] = None) -> bool:
    """Determine if interactive visualization should be used."""
    if not PLOTLY_AVAILABLE:
        return False
    
    if config is None:
        config = load_config()
    
    viz_config = config.get("visualization", {})
    mode = viz_config.get("mode", "auto")
    
    if mode == "interactive":
        return True
    elif mode == "static":
        return False
    else:  # auto mode
        if output_format:
            return output_format.lower() in ["html", "interactive", "plotly"]
        
        # Check for interactive environment
        if "JUPYTER_RUNTIME_DIR" in os.environ:
            return True
        if hasattr(sys, 'ps1'):
            return True
        if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
            return True
        
        return False


# ============================================================================
# SECTION 3: Base Dashboard Class
# ============================================================================

class BaseDashboard(ABC):
    """Abstract base class for all dashboard implementations."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize dashboard with configuration."""
        self.config = config or DEFAULT_DASHBOARD_CONFIG
        self.results = []
        self.user_id = None
        
    @abstractmethod
    def create_dashboard(self, results: List[Dict[str, Any]], 
                        user_id: str, 
                        config: Optional[Dict[str, Any]] = None,
                        output_dir: str = "output") -> Any:
        """Create dashboard visualization."""
        pass
    
    def prepare_data(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        """Prepare results data for visualization."""
        if not results:
            return pd.DataFrame()
        
        df = pd.DataFrame(results)
        
        # Convert timestamp to datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        return df
    
    def calculate_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate summary statistics from results."""
        if not results:
            return {}
        
        accepted = [r for r in results if r.get('accepted', False)]
        rejected = [r for r in results if r.get('rejected', False)]
        
        stats = {
            'total_measurements': len(results),
            'accepted_count': len(accepted),
            'rejected_count': len(rejected),
            'acceptance_rate': len(accepted) / len(results) if results else 0,
            'unique_sources': len(set(r.get('source', 'unknown') for r in results))
        }
        
        if accepted:
            weights = [r['filtered_weight'] for r in accepted if 'filtered_weight' in r]
            if weights:
                stats['mean_weight'] = np.mean(weights)
                stats['std_weight'] = np.std(weights)
                stats['min_weight'] = np.min(weights)
                stats['max_weight'] = np.max(weights)
        
        return stats


# ============================================================================
# SECTION 4: Static Dashboard (Matplotlib)
# ============================================================================

class StaticDashboard(BaseDashboard):
    """Static dashboard using matplotlib."""
    
    def create_dashboard(self, results: List[Dict[str, Any]], 
                        user_id: str,
                        config: Optional[Dict[str, Any]] = None,
                        output_dir: str = "output") -> str:
        """Create static matplotlib dashboard."""
        self.results = results
        self.user_id = user_id
        if config:
            self.config.update(config)
        
        # Prepare data
        df = self.prepare_data(results)
        stats = self.calculate_statistics(results)
        
        # Create figure
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle(f'Weight Processing Dashboard - User {user_id}', fontsize=16, fontweight='bold')
        
        # Create subplots
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        # Weight timeline
        ax1 = fig.add_subplot(gs[0:2, :])
        self._plot_weight_timeline(ax1, df)
        
        # Rejection analysis
        ax2 = fig.add_subplot(gs[2, 0])
        self._plot_rejection_pie(ax2, results)
        
        # Source distribution
        ax3 = fig.add_subplot(gs[2, 1])
        self._plot_source_distribution(ax3, results)
        
        # Confidence over time
        ax4 = fig.add_subplot(gs[2, 2])
        self._plot_confidence_timeline(ax4, df)
        
        # Statistics text
        ax5 = fig.add_subplot(gs[3, :])
        self._plot_statistics(ax5, stats)
        
        # Save figure
        output_path = Path(output_dir) / f"{user_id}_dashboard.png"
        output_path.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def _plot_weight_timeline(self, ax, df):
        """Plot weight measurements over time."""
        if df.empty:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
            return
        
        # Accepted measurements
        accepted = df[df['accepted'] == True]
        if not accepted.empty:
            ax.plot(accepted['timestamp'], accepted['filtered_weight'], 
                   'o-', color=CHART_COLORS['accepted'], label='Accepted', markersize=6)
        
        # Rejected measurements
        rejected = df[df['accepted'] == False] if 'accepted' in df.columns else pd.DataFrame()
        if not rejected.empty:
            ax.scatter(rejected['timestamp'], rejected['raw_weight'],
                      color=CHART_COLORS['rejected'], marker='x', s=50, label='Rejected')
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Weight (kg)')
        ax.set_title('Weight Measurements Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    def _plot_rejection_pie(self, ax, results):
        """Plot rejection reason distribution."""
        rejected = [r for r in results if r.get('rejected', False)]
        
        if not rejected:
            ax.text(0.5, 0.5, 'No rejections', ha='center', va='center')
            ax.set_title('Rejection Reasons')
            return
        
        # Categorize rejections
        categories = defaultdict(int)
        for r in rejected:
            reason = r.get('reason', 'Unknown')
            category = categorize_rejection(reason)
            categories[category] += 1
        
        # Create pie chart
        labels = list(categories.keys())
        sizes = list(categories.values())
        colors = [get_rejection_color(cat) for cat in labels]
        
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title('Rejection Reasons')
    
    def _plot_source_distribution(self, ax, results):
        """Plot data source distribution."""
        sources = defaultdict(int)
        for r in results:
            source = r.get('source', 'unknown')
            sources[source] += 1
        
        if not sources:
            ax.text(0.5, 0.5, 'No source data', ha='center', va='center')
            ax.set_title('Data Sources')
            return
        
        # Create bar chart
        labels = list(sources.keys())
        values = list(sources.values())
        colors = [get_source_style(s)['color'] for s in labels]
        
        bars = ax.bar(range(len(labels)), values, color=colors)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels([get_source_style(s)['label'] for s in labels], rotation=45, ha='right')
        ax.set_ylabel('Count')
        ax.set_title('Data Sources')
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val}', ha='center', va='bottom')
    
    def _plot_confidence_timeline(self, ax, df):
        """Plot confidence scores over time."""
        accepted = df[df['accepted'] == True]
        
        if accepted.empty or 'confidence' not in accepted.columns:
            ax.text(0.5, 0.5, 'No confidence data', ha='center', va='center')
            ax.set_title('Confidence Scores')
            return
        
        ax.plot(accepted['timestamp'], accepted['confidence'],
               color=CHART_COLORS['confidence'], marker='o', markersize=4)
        ax.set_xlabel('Date')
        ax.set_ylabel('Confidence')
        ax.set_title('Confidence Scores Over Time')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    def _plot_statistics(self, ax, stats):
        """Display summary statistics."""
        ax.axis('off')
        
        text = f"""
        Summary Statistics:
        • Total Measurements: {stats.get('total_measurements', 0)}
        • Accepted: {stats.get('accepted_count', 0)} ({stats.get('acceptance_rate', 0):.1%})
        • Rejected: {stats.get('rejected_count', 0)}
        • Unique Sources: {stats.get('unique_sources', 0)}
        """
        
        if 'mean_weight' in stats:
            text += f"""
        • Mean Weight: {stats['mean_weight']:.1f} kg
        • Std Dev: {stats['std_weight']:.2f} kg
        • Range: {stats['min_weight']:.1f} - {stats['max_weight']:.1f} kg
        """
        
        ax.text(0.1, 0.5, text, fontsize=10, va='center', family='monospace')


# ============================================================================
# SECTION 5: Interactive Dashboard (Plotly)
# ============================================================================

class InteractiveDashboard(BaseDashboard):
    """Interactive dashboard using Plotly."""
    
    def create_dashboard(self, results: List[Dict[str, Any]], 
                        user_id: str,
                        config: Optional[Dict[str, Any]] = None,
                        output_dir: str = "output") -> str:
        """Create interactive Plotly dashboard."""
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is not installed. Please install it to use interactive dashboards.")
        
        self.results = results
        self.user_id = user_id
        if config:
            self.config.update(config)
        
        # Prepare data
        df = self.prepare_data(results)
        stats = self.calculate_statistics(results)
        
        # Create figure with subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Weight Measurements Over Time',
                'Kalman Filter Performance',
                'Rejection Analysis',
                'Data Source Distribution',
                'Confidence & Innovation',
                'Quality Scores'
            ),
            specs=[
                [{'colspan': 2}, None],
                [{}, {}],
                [{}, {}]
            ],
            vertical_spacing=0.1,
            horizontal_spacing=0.15
        )
        
        # Add traces
        self._add_weight_timeline(fig, df, row=1, col=1)
        self._add_kalman_performance(fig, df, row=2, col=1)
        self._add_rejection_analysis(fig, results, row=2, col=2)
        self._add_source_distribution(fig, results, row=3, col=1)
        self._add_confidence_innovation(fig, df, row=3, col=2)
        
        # Update layout
        fig.update_layout(
            title=f'Weight Processing Dashboard - User {user_id}',
            height=900,
            showlegend=True,
            hovermode='x unified',
            template='plotly_white'
        )
        
        # Save to HTML
        output_path = Path(output_dir) / f"{user_id}_interactive.html"
        output_path.parent.mkdir(exist_ok=True, parents=True)
        fig.write_html(str(output_path), config=PLOTLY_CONFIG)
        
        return str(output_path)
    
    def _add_weight_timeline(self, fig, df, row, col):
        """Add weight timeline to figure."""
        if df.empty:
            return
        
        # Accepted measurements
        accepted = df[df['accepted'] == True]
        if not accepted.empty:
            fig.add_trace(
                go.Scatter(
                    x=accepted['timestamp'],
                    y=accepted['filtered_weight'],
                    mode='lines+markers',
                    name='Accepted',
                    marker=dict(color=CHART_COLORS['accepted'], size=8),
                    line=dict(color=CHART_COLORS['accepted'], width=2)
                ),
                row=row, col=col
            )
        
        # Rejected measurements
        rejected = df[df['accepted'] == False] if 'accepted' in df.columns else pd.DataFrame()
        if not rejected.empty:
            fig.add_trace(
                go.Scatter(
                    x=rejected['timestamp'],
                    y=rejected['raw_weight'],
                    mode='markers',
                    name='Rejected',
                    marker=dict(
                        color=CHART_COLORS['rejected'],
                        size=10,
                        symbol='x'
                    )
                ),
                row=row, col=col
            )
        
        # Add trend line if available
        if not accepted.empty and 'trend_weekly' in accepted.columns:
            fig.add_trace(
                go.Scatter(
                    x=accepted['timestamp'],
                    y=accepted['filtered_weight'] + accepted['trend_weekly'],
                    mode='lines',
                    name='Trend',
                    line=dict(color=CHART_COLORS['trend'], width=1, dash='dash')
                ),
                row=row, col=col
            )
        
        fig.update_xaxes(title_text="Date", row=row, col=col)
        fig.update_yaxes(title_text="Weight (kg)", row=row, col=col)
    
    def _add_kalman_performance(self, fig, df, row, col):
        """Add Kalman filter performance metrics."""
        accepted = df[df['accepted'] == True]
        
        if accepted.empty:
            return
        
        # Innovation (residuals)
        if 'innovation' in accepted.columns:
            fig.add_trace(
                go.Scatter(
                    x=accepted['timestamp'],
                    y=accepted['innovation'],
                    mode='lines+markers',
                    name='Innovation',
                    marker=dict(color=CHART_COLORS['innovation'], size=6),
                    line=dict(color=CHART_COLORS['innovation'], width=1)
                ),
                row=row, col=col
            )
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=row, col=col)
        
        fig.update_xaxes(title_text="Date", row=row, col=col)
        fig.update_yaxes(title_text="Innovation (kg)", row=row, col=col)
    
    def _add_rejection_analysis(self, fig, results, row, col):
        """Add rejection analysis pie chart."""
        rejected = [r for r in results if r.get('rejected', False)]
        
        if not rejected:
            return
        
        # Categorize rejections
        categories = defaultdict(int)
        for r in rejected:
            reason = r.get('reason', 'Unknown')
            category = categorize_rejection(reason)
            categories[category] += 1
        
        # Create pie chart
        labels = list(categories.keys())
        values = list(categories.values())
        colors = [get_rejection_color(cat) for cat in labels]
        
        fig.add_trace(
            go.Pie(
                labels=labels,
                values=values,
                marker=dict(colors=colors),
                textposition='inside',
                textinfo='percent+label'
            ),
            row=row, col=col
        )
    
    def _add_source_distribution(self, fig, results, row, col):
        """Add data source distribution bar chart."""
        sources = defaultdict(int)
        for r in results:
            source = r.get('source', 'unknown')
            sources[source] += 1
        
        if not sources:
            return
        
        labels = list(sources.keys())
        values = list(sources.values())
        colors = [get_source_style(s)['color'] for s in labels]
        display_labels = [get_source_style(s)['label'] for s in labels]
        
        fig.add_trace(
            go.Bar(
                x=display_labels,
                y=values,
                marker=dict(color=colors),
                text=values,
                textposition='auto',
                name='Sources'
            ),
            row=row, col=col
        )
        
        fig.update_xaxes(title_text="Source", row=row, col=col)
        fig.update_yaxes(title_text="Count", row=row, col=col)
    
    def _add_confidence_innovation(self, fig, df, row, col):
        """Add confidence and normalized innovation plot."""
        accepted = df[df['accepted'] == True]
        
        if accepted.empty:
            return
        
        # Confidence scores
        if 'confidence' in accepted.columns:
            fig.add_trace(
                go.Scatter(
                    x=accepted['timestamp'],
                    y=accepted['confidence'],
                    mode='lines+markers',
                    name='Confidence',
                    marker=dict(color=CHART_COLORS['confidence'], size=6),
                    line=dict(color=CHART_COLORS['confidence'], width=2),
                    yaxis='y'
                ),
                row=row, col=col
            )
        
        # Normalized innovation
        if 'normalized_innovation' in accepted.columns:
            fig.add_trace(
                go.Scatter(
                    x=accepted['timestamp'],
                    y=accepted['normalized_innovation'],
                    mode='lines+markers',
                    name='Norm. Innovation',
                    marker=dict(color=CHART_COLORS['innovation'], size=6),
                    line=dict(color=CHART_COLORS['innovation'], width=1, dash='dot'),
                    yaxis='y2'
                ),
                row=row, col=col
            )
        
        fig.update_xaxes(title_text="Date", row=row, col=col)
        fig.update_yaxes(title_text="Confidence", row=row, col=col)


# ============================================================================
# SECTION 6: Diagnostic Dashboard
# ============================================================================

class DiagnosticDashboard(InteractiveDashboard):
    """Enhanced diagnostic dashboard with detailed analysis."""
    
    def create_dashboard(self, results: List[Dict[str, Any]], 
                        user_id: str,
                        config: Optional[Dict[str, Any]] = None,
                        output_dir: str = "output") -> str:
        """Create comprehensive diagnostic dashboard."""
        if not PLOTLY_AVAILABLE:
            # Fall back to static dashboard
            static_dash = StaticDashboard(config)
            return static_dash.create_dashboard(results, user_id, config, output_dir)
        
        self.results = results
        self.user_id = user_id
        if config:
            self.config.update(config)
        
        # Prepare data
        df = self.prepare_data(results)
        
        # Create comprehensive figure
        fig = self._create_diagnostic_figure(df, results)
        
        # Add diagnostic report
        report = self._create_diagnostic_report(results)
        
        # Save outputs
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # Save HTML dashboard
        dashboard_path = output_path / f"{user_id}_diagnostic.html"
        fig.write_html(str(dashboard_path), config=PLOTLY_CONFIG)
        
        # Save text report
        report_path = output_path / f"{user_id}_diagnostic_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        
        return str(dashboard_path)
    
    def _create_diagnostic_figure(self, df, results):
        """Create comprehensive diagnostic figure."""
        # Create figure with many subplots for detailed analysis
        fig = make_subplots(
            rows=5, cols=2,
            subplot_titles=(
                'Weight Timeline with All Data',
                'Kalman Filter State Evolution',
                'Rejection Patterns Over Time',
                'Source Reliability Analysis',
                'Innovation Distribution',
                'Confidence vs Innovation',
                'Quality Score Components',
                'Processing Latency',
                'Data Gaps Analysis',
                'Summary Statistics'
            ),
            specs=[
                [{'colspan': 2}, None],
                [{}, {}],
                [{}, {}],
                [{}, {}],
                [{}, {'type': 'table'}]
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.12
        )
        
        # Add all diagnostic plots
        self._add_comprehensive_timeline(fig, df, row=1, col=1)
        self._add_kalman_evolution(fig, df, row=2, col=1)
        self._add_rejection_patterns(fig, results, row=2, col=2)
        self._add_source_reliability(fig, results, row=3, col=1)
        self._add_innovation_distribution(fig, df, row=3, col=2)
        self._add_confidence_vs_innovation(fig, df, row=4, col=1)
        self._add_quality_components(fig, results, row=4, col=2)
        self._add_data_gaps(fig, df, row=5, col=1)
        self._add_summary_stats(fig, results, row=5, col=2)
        
        # Update layout
        fig.update_layout(
            title=f'Diagnostic Dashboard - User {self.user_id}',
            height=1400,
            showlegend=True,
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    def _add_comprehensive_timeline(self, fig, df, row, col):
        """Add comprehensive weight timeline with all data points."""
        if df.empty:
            return
        
        # Raw measurements
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['raw_weight'],
                mode='markers',
                name='Raw',
                marker=dict(color=CHART_COLORS['raw'], size=4, opacity=0.5)
            ),
            row=row, col=col
        )
        
        # Filtered weights
        accepted = df[df['accepted'] == True]
        if not accepted.empty:
            fig.add_trace(
                go.Scatter(
                    x=accepted['timestamp'],
                    y=accepted['filtered_weight'],
                    mode='lines+markers',
                    name='Filtered',
                    marker=dict(color=CHART_COLORS['filtered'], size=6),
                    line=dict(color=CHART_COLORS['filtered'], width=2)
                ),
                row=row, col=col
            )
        
        # Rejected with reasons
        if 'rejected' in df.columns:
            rejected = df[df['rejected'] == True]
        else:
            rejected = df[df['accepted'] == False] if 'accepted' in df.columns else pd.DataFrame()
        if not rejected.empty:
            # Group by rejection category
            for category in rejected['reason'].apply(categorize_rejection).unique():
                cat_data = rejected[rejected['reason'].apply(categorize_rejection) == category]
                fig.add_trace(
                    go.Scatter(
                        x=cat_data['timestamp'],
                        y=cat_data['raw_weight'],
                        mode='markers',
                        name=f'Rejected: {category}',
                        marker=dict(
                            color=get_rejection_color(category),
                            size=8,
                            symbol='x'
                        )
                    ),
                    row=row, col=col
                )
        
        fig.update_xaxes(title_text="Date", row=row, col=col)
        fig.update_yaxes(title_text="Weight (kg)", row=row, col=col)
    
    def _add_kalman_evolution(self, fig, df, row, col):
        """Show Kalman filter state evolution."""
        accepted = df[df['accepted'] == True]
        
        if accepted.empty:
            return
        
        # State estimate vs raw
        fig.add_trace(
            go.Scatter(
                x=accepted['timestamp'],
                y=accepted['raw_weight'],
                mode='markers',
                name='Raw Input',
                marker=dict(color=CHART_COLORS['raw'], size=4, opacity=0.5)
            ),
            row=row, col=col
        )
        
        fig.add_trace(
            go.Scatter(
                x=accepted['timestamp'],
                y=accepted['filtered_weight'],
                mode='lines',
                name='Kalman Estimate',
                line=dict(color=CHART_COLORS['filtered'], width=2)
            ),
            row=row, col=col
        )
        
        # Add confidence bands if available
        if 'confidence' in accepted.columns:
            # Approximate confidence bands
            std_dev = accepted['filtered_weight'].std()
            upper = accepted['filtered_weight'] + std_dev * (1 - accepted['confidence'])
            lower = accepted['filtered_weight'] - std_dev * (1 - accepted['confidence'])
            
            fig.add_trace(
                go.Scatter(
                    x=accepted['timestamp'],
                    y=upper,
                    mode='lines',
                    name='Upper Bound',
                    line=dict(color=CHART_COLORS['confidence'], width=1, dash='dot'),
                    showlegend=False
                ),
                row=row, col=col
            )
            
            fig.add_trace(
                go.Scatter(
                    x=accepted['timestamp'],
                    y=lower,
                    mode='lines',
                    name='Lower Bound',
                    line=dict(color=CHART_COLORS['confidence'], width=1, dash='dot'),
                    fill='tonexty',
                    fillcolor='rgba(123, 31, 162, 0.1)',
                    showlegend=False
                ),
                row=row, col=col
            )
        
        fig.update_xaxes(title_text="Date", row=row, col=col)
        fig.update_yaxes(title_text="Weight (kg)", row=row, col=col)
    
    def _add_rejection_patterns(self, fig, results, row, col):
        """Analyze rejection patterns over time."""
        rejected = [r for r in results if r.get('rejected', False)]
        
        if not rejected:
            return
        
        # Create time series of rejections by category
        df_rejected = pd.DataFrame(rejected)
        df_rejected['timestamp'] = pd.to_datetime(df_rejected['timestamp'])
        df_rejected['category'] = df_rejected['reason'].apply(categorize_rejection)
        df_rejected['date'] = df_rejected['timestamp'].dt.date
        
        # Count rejections by date and category
        rejection_counts = df_rejected.groupby(['date', 'category']).size().reset_index(name='count')
        
        # Create stacked bar chart
        for category in rejection_counts['category'].unique():
            cat_data = rejection_counts[rejection_counts['category'] == category]
            fig.add_trace(
                go.Bar(
                    x=cat_data['date'],
                    y=cat_data['count'],
                    name=category,
                    marker=dict(color=get_rejection_color(category))
                ),
                row=row, col=col
            )
        
        fig.update_xaxes(title_text="Date", row=row, col=col)
        fig.update_yaxes(title_text="Rejection Count", row=row, col=col)
        fig.update_layout(barmode='stack')
    
    def _add_source_reliability(self, fig, results, row, col):
        """Analyze reliability by data source."""
        source_stats = defaultdict(lambda: {'total': 0, 'accepted': 0, 'rejected': 0})
        
        for r in results:
            source = r.get('source', 'unknown')
            source_stats[source]['total'] += 1
            if r.get('accepted', False):
                source_stats[source]['accepted'] += 1
            if r.get('rejected', False):
                source_stats[source]['rejected'] += 1
        
        # Calculate acceptance rates
        sources = []
        acceptance_rates = []
        counts = []
        
        for source, stats in source_stats.items():
            sources.append(get_source_style(source)['label'])
            acceptance_rates.append(stats['accepted'] / stats['total'] if stats['total'] > 0 else 0)
            counts.append(stats['total'])
        
        # Create bar chart with acceptance rates
        fig.add_trace(
            go.Bar(
                x=sources,
                y=acceptance_rates,
                text=[f'{rate:.1%}<br>n={count}' for rate, count in zip(acceptance_rates, counts)],
                textposition='auto',
                marker=dict(
                    color=acceptance_rates,
                    colorscale='RdYlGn',
                    cmin=0,
                    cmax=1,
                    showscale=True,
                    colorbar=dict(title="Accept Rate")
                ),
                name='Acceptance Rate'
            ),
            row=row, col=col
        )
        
        fig.update_xaxes(title_text="Source", row=row, col=col)
        fig.update_yaxes(title_text="Acceptance Rate", range=[0, 1], row=row, col=col)
    
    def _add_innovation_distribution(self, fig, df, row, col):
        """Show distribution of innovations."""
        accepted = df[df['accepted'] == True]
        
        if accepted.empty or 'innovation' not in accepted.columns:
            return
        
        # Create histogram
        fig.add_trace(
            go.Histogram(
                x=accepted['innovation'],
                nbinsx=30,
                marker=dict(color=CHART_COLORS['innovation']),
                name='Innovation'
            ),
            row=row, col=col
        )
        
        # Add normal distribution overlay
        mean = accepted['innovation'].mean()
        std = accepted['innovation'].std()
        x_range = np.linspace(mean - 3*std, mean + 3*std, 100)
        y_normal = np.exp(-0.5 * ((x_range - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))
        y_normal = y_normal * len(accepted) * (accepted['innovation'].max() - accepted['innovation'].min()) / 30
        
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=y_normal,
                mode='lines',
                name='Normal Dist',
                line=dict(color='red', width=2, dash='dash')
            ),
            row=row, col=col
        )
        
        fig.update_xaxes(title_text="Innovation (kg)", row=row, col=col)
        fig.update_yaxes(title_text="Frequency", row=row, col=col)
    
    def _add_confidence_vs_innovation(self, fig, df, row, col):
        """Plot confidence vs normalized innovation."""
        accepted = df[df['accepted'] == True]
        
        if accepted.empty or 'confidence' not in accepted.columns or 'normalized_innovation' not in accepted.columns:
            return
        
        # Scatter plot
        fig.add_trace(
            go.Scatter(
                x=accepted['normalized_innovation'],
                y=accepted['confidence'],
                mode='markers',
                marker=dict(
                    color=accepted.index,
                    colorscale='Viridis',
                    size=6,
                    showscale=True,
                    colorbar=dict(title="Time Index")
                ),
                text=[f"Date: {t}" for t in accepted['timestamp']],
                name='Measurements'
            ),
            row=row, col=col
        )
        
        # Add theoretical curve
        x_theory = np.linspace(0, accepted['normalized_innovation'].max(), 100)
        y_theory = np.exp(-0.5 * x_theory ** 2)  # Theoretical confidence
        
        fig.add_trace(
            go.Scatter(
                x=x_theory,
                y=y_theory,
                mode='lines',
                name='Theoretical',
                line=dict(color='red', width=2, dash='dash')
            ),
            row=row, col=col
        )
        
        fig.update_xaxes(title_text="Normalized Innovation", row=row, col=col)
        fig.update_yaxes(title_text="Confidence", range=[0, 1], row=row, col=col)
    
    def _add_quality_components(self, fig, results, row, col):
        """Show quality score components if available."""
        quality_data = []
        
        for r in results:
            if 'quality_score' in r and isinstance(r['quality_score'], dict):
                components = r['quality_score'].get('components', {})
                if components:
                    quality_data.append({
                        'timestamp': r['timestamp'],
                        **components
                    })
        
        if not quality_data:
            # No quality data, show placeholder
            fig.add_annotation(
                text="No quality score data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                row=row, col=col
            )
            return
        
        df_quality = pd.DataFrame(quality_data)
        df_quality['timestamp'] = pd.to_datetime(df_quality['timestamp'])
        
        # Plot each component
        for component in df_quality.columns:
            if component != 'timestamp':
                fig.add_trace(
                    go.Scatter(
                        x=df_quality['timestamp'],
                        y=df_quality[component],
                        mode='lines+markers',
                        name=component.capitalize(),
                        marker=dict(size=4)
                    ),
                    row=row, col=col
                )
        
        fig.update_xaxes(title_text="Date", row=row, col=col)
        fig.update_yaxes(title_text="Score", range=[0, 1], row=row, col=col)
    
    def _add_data_gaps(self, fig, df, row, col):
        """Analyze data gaps and patterns."""
        if df.empty:
            return
        
        # Calculate time differences
        df_sorted = df.sort_values('timestamp')
        time_diffs = df_sorted['timestamp'].diff()
        
        # Convert to hours
        gap_hours = time_diffs.dt.total_seconds() / 3600
        
        # Create histogram of gaps
        fig.add_trace(
            go.Histogram(
                x=gap_hours.dropna(),
                nbinsx=50,
                marker=dict(color=CHART_COLORS['innovation']),
                name='Gap Distribution'
            ),
            row=row, col=col
        )
        
        # Add vertical lines for notable thresholds
        fig.add_vline(x=24, line_dash="dash", line_color="red", 
                     annotation_text="1 day", row=row, col=col)
        fig.add_vline(x=168, line_dash="dash", line_color="orange",
                     annotation_text="1 week", row=row, col=col)
        
        fig.update_xaxes(title_text="Gap Duration (hours)", type="log", row=row, col=col)
        fig.update_yaxes(title_text="Frequency", row=row, col=col)
    
    def _add_summary_stats(self, fig, results, row, col):
        """Add summary statistics table."""
        stats = self.calculate_statistics(results)
        
        # Additional statistics
        if results:
            # Time span
            timestamps = [datetime.fromisoformat(r['timestamp']) if isinstance(r['timestamp'], str) 
                         else r['timestamp'] for r in results]
            time_span = (max(timestamps) - min(timestamps)).days
            stats['time_span_days'] = time_span
            
            # Average measurements per day
            stats['avg_per_day'] = len(results) / max(time_span, 1)
        
        # Create table
        stat_names = []
        stat_values = []
        
        stat_names.extend([
            'Total Measurements',
            'Accepted',
            'Rejected',
            'Acceptance Rate',
            'Time Span (days)',
            'Avg per Day',
            'Unique Sources'
        ])
        
        stat_values.extend([
            str(stats.get('total_measurements', 0)),
            str(stats.get('accepted_count', 0)),
            str(stats.get('rejected_count', 0)),
            f"{stats.get('acceptance_rate', 0):.1%}",
            str(stats.get('time_span_days', 0)),
            f"{stats.get('avg_per_day', 0):.1f}",
            str(stats.get('unique_sources', 0))
        ])
        
        if 'mean_weight' in stats:
            stat_names.extend(['Mean Weight (kg)', 'Std Dev (kg)', 'Range (kg)'])
            stat_values.extend([
                f"{stats['mean_weight']:.1f}",
                f"{stats['std_weight']:.2f}",
                f"{stats['min_weight']:.1f} - {stats['max_weight']:.1f}"
            ])
        
        fig.add_trace(
            go.Table(
                header=dict(
                    values=['Metric', 'Value'],
                    fill_color='lightgray',
                    align='left'
                ),
                cells=dict(
                    values=[stat_names, stat_values],
                    fill_color='white',
                    align='left'
                )
            ),
            row=row, col=col
        )
    
    def _create_diagnostic_report(self, results):
        """Create detailed text diagnostic report."""
        report = []
        report.append("=" * 80)
        report.append(f"DIAGNOSTIC REPORT - User {self.user_id}")
        report.append("=" * 80)
        report.append("")
        
        # Summary statistics
        stats = self.calculate_statistics(results)
        report.append("SUMMARY STATISTICS")
        report.append("-" * 40)
        report.append(f"Total Measurements: {stats.get('total_measurements', 0)}")
        report.append(f"Accepted: {stats.get('accepted_count', 0)} ({stats.get('acceptance_rate', 0):.1%})")
        report.append(f"Rejected: {stats.get('rejected_count', 0)}")
        report.append(f"Unique Sources: {stats.get('unique_sources', 0)}")
        
        if 'mean_weight' in stats:
            report.append(f"\nWeight Statistics:")
            report.append(f"  Mean: {stats['mean_weight']:.1f} kg")
            report.append(f"  Std Dev: {stats['std_weight']:.2f} kg")
            report.append(f"  Range: {stats['min_weight']:.1f} - {stats['max_weight']:.1f} kg")
        
        # Rejection analysis
        report.append("\n" + "=" * 40)
        report.append("REJECTION ANALYSIS")
        report.append("-" * 40)
        
        rejected = [r for r in results if r.get('rejected', False)]
        if rejected:
            categories = defaultdict(list)
            for r in rejected:
                reason = r.get('reason', 'Unknown')
                category = categorize_rejection(reason)
                categories[category].append(reason)
            
            for category, reasons in sorted(categories.items(), key=lambda x: -len(x[1])):
                report.append(f"\n{category}: {len(reasons)} occurrences")
                # Show first few unique reasons
                unique_reasons = list(set(reasons))[:3]
                for reason in unique_reasons:
                    report.append(f"  - {reason[:80]}")
        else:
            report.append("No rejections found")
        
        # Source analysis
        report.append("\n" + "=" * 40)
        report.append("SOURCE ANALYSIS")
        report.append("-" * 40)
        
        source_stats = defaultdict(lambda: {'total': 0, 'accepted': 0, 'rejected': 0})
        for r in results:
            source = r.get('source', 'unknown')
            source_stats[source]['total'] += 1
            if r.get('accepted', False):
                source_stats[source]['accepted'] += 1
            if r.get('rejected', False):
                source_stats[source]['rejected'] += 1
        
        for source, stats in sorted(source_stats.items(), key=lambda x: -x[1]['total']):
            accept_rate = stats['accepted'] / stats['total'] if stats['total'] > 0 else 0
            report.append(f"\n{source}:")
            report.append(f"  Total: {stats['total']}")
            report.append(f"  Accepted: {stats['accepted']} ({accept_rate:.1%})")
            report.append(f"  Rejected: {stats['rejected']}")
        
        # Data quality indicators
        report.append("\n" + "=" * 40)
        report.append("DATA QUALITY INDICATORS")
        report.append("-" * 40)
        
        # Check for data gaps
        if results:
            timestamps = sorted([datetime.fromisoformat(r['timestamp']) if isinstance(r['timestamp'], str) 
                               else r['timestamp'] for r in results])
            
            gaps = []
            for i in range(1, len(timestamps)):
                gap = (timestamps[i] - timestamps[i-1]).total_seconds() / 3600
                if gap > 24:  # Gaps longer than 1 day
                    gaps.append((timestamps[i-1], timestamps[i], gap))
            
            if gaps:
                report.append(f"\nSignificant gaps found: {len(gaps)}")
                for start, end, hours in gaps[:5]:  # Show first 5 gaps
                    report.append(f"  {start.date()} to {end.date()}: {hours/24:.1f} days")
            else:
                report.append("\nNo significant gaps found")
        
        # Kalman filter performance
        accepted = [r for r in results if r.get('accepted', False)]
        if accepted and any('confidence' in r for r in accepted):
            confidences = [r['confidence'] for r in accepted if 'confidence' in r]
            report.append(f"\nKalman Filter Performance:")
            report.append(f"  Mean Confidence: {np.mean(confidences):.3f}")
            report.append(f"  Min Confidence: {np.min(confidences):.3f}")
            report.append(f"  Max Confidence: {np.max(confidences):.3f}")
        
        report.append("\n" + "=" * 80)
        report.append(f"Report generated: {datetime.now().isoformat()}")
        report.append("=" * 80)
        
        return "\n".join(report)


# ============================================================================
# SECTION 7: Specialized Visualizers
# ============================================================================

class KalmanVisualizer:
    """Specialized visualizations for Kalman filter analysis."""
    
    @staticmethod
    def extract_kalman_data(results: List[Dict[str, Any]]) -> pd.DataFrame:
        """Extract Kalman-specific data from results."""
        kalman_data = []
        
        for r in results:
            if r.get('accepted', False):
                kalman_data.append({
                    'timestamp': r['timestamp'],
                    'raw_weight': r.get('raw_weight'),
                    'filtered_weight': r.get('filtered_weight'),
                    'innovation': r.get('innovation'),
                    'normalized_innovation': r.get('normalized_innovation'),
                    'confidence': r.get('confidence'),
                    'trend': r.get('trend'),
                    'trend_weekly': r.get('trend_weekly')
                })
        
        if not kalman_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(kalman_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df.sort_values('timestamp')
    
    @staticmethod
    def create_kalman_analysis_plot(results: List[Dict[str, Any]]) -> go.Figure:
        """Create detailed Kalman filter analysis plot."""
        df = KalmanVisualizer.extract_kalman_data(results)
        
        if df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No Kalman data available", x=0.5, y=0.5)
            return fig
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=(
                'Kalman Filter State Estimation',
                'Innovation Analysis',
                'Confidence Evolution'
            ),
            vertical_spacing=0.1
        )
        
        # State estimation
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['raw_weight'],
                mode='markers',
                name='Raw',
                marker=dict(color=CHART_COLORS['raw'], size=4, opacity=0.5)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['filtered_weight'],
                mode='lines',
                name='Filtered',
                line=dict(color=CHART_COLORS['filtered'], width=2)
            ),
            row=1, col=1
        )
        
        # Innovation
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['innovation'],
                mode='lines+markers',
                name='Innovation',
                marker=dict(color=CHART_COLORS['innovation'], size=4)
            ),
            row=2, col=1
        )
        
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
        
        # Confidence
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['confidence'],
                mode='lines+markers',
                name='Confidence',
                marker=dict(color=CHART_COLORS['confidence'], size=4),
                fill='tozeroy',
                fillcolor='rgba(123, 31, 162, 0.2)'
            ),
            row=3, col=1
        )
        
        # Update layout
        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_yaxes(title_text="Weight (kg)", row=1, col=1)
        fig.update_yaxes(title_text="Innovation (kg)", row=2, col=1)
        fig.update_yaxes(title_text="Confidence", range=[0, 1], row=3, col=1)
        
        fig.update_layout(
            title="Kalman Filter Analysis",
            height=800,
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig


class QualityVisualizer:
    """Specialized visualizations for quality score analysis."""
    
    @staticmethod
    def extract_quality_data(results: List[Dict[str, Any]]) -> pd.DataFrame:
        """Extract quality score data from results."""
        quality_data = []
        
        for r in results:
            if 'quality_score' in r and isinstance(r['quality_score'], dict):
                quality_data.append({
                    'timestamp': r['timestamp'],
                    'overall': r['quality_score'].get('overall'),
                    'accepted': r['quality_score'].get('accepted', r.get('accepted', False)),
                    **r['quality_score'].get('components', {})
                })
        
        if not quality_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(quality_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df.sort_values('timestamp')
    
    @staticmethod
    def create_quality_analysis_plot(results: List[Dict[str, Any]]) -> go.Figure:
        """Create quality score analysis plot."""
        df = QualityVisualizer.extract_quality_data(results)
        
        if df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No quality score data available", x=0.5, y=0.5)
            return fig
        
        # Create figure
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(
                'Overall Quality Score',
                'Quality Components'
            ),
            vertical_spacing=0.15
        )
        
        # Overall score with acceptance
        accepted = df[df['accepted'] == True]
        rejected = df[df['accepted'] == False]
        
        if not accepted.empty:
            fig.add_trace(
                go.Scatter(
                    x=accepted['timestamp'],
                    y=accepted['overall'],
                    mode='markers',
                    name='Accepted',
                    marker=dict(color=CHART_COLORS['accepted'], size=8)
                ),
                row=1, col=1
            )
        
        if not rejected.empty:
            fig.add_trace(
                go.Scatter(
                    x=rejected['timestamp'],
                    y=rejected['overall'],
                    mode='markers',
                    name='Rejected',
                    marker=dict(color=CHART_COLORS['rejected'], size=8, symbol='x')
                ),
                row=1, col=1
            )
        
        # Add threshold line
        fig.add_hline(y=0.6, line_dash="dash", line_color="orange",
                     annotation_text="Threshold", row=1, col=1)
        
        # Components
        component_cols = [col for col in df.columns 
                         if col not in ['timestamp', 'overall', 'accepted']]
        
        for col in component_cols:
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df[col],
                    mode='lines+markers',
                    name=col.capitalize(),
                    marker=dict(size=4)
                ),
                row=2, col=1
            )
        
        # Update layout
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Score", range=[0, 1], row=1, col=1)
        fig.update_yaxes(title_text="Component Score", range=[0, 1], row=2, col=1)
        
        fig.update_layout(
            title="Quality Score Analysis",
            height=600,
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig


class IndexVisualizer:
    """Create index/overview visualizations with user list and iframe viewer."""
    
    @staticmethod
    def extract_user_stats(all_results: Dict[str, List[Dict]]) -> List[Dict[str, Any]]:
        """Extract statistics for each user from results."""
        user_stats = []
        
        for user_id, measurements in all_results.items():
            if not measurements:
                continue
            
            total = len(measurements)
            accepted = sum(1 for m in measurements if m.get("accepted", False))
            rejected = total - accepted
            
            dates = [m.get("timestamp") for m in measurements if m.get("timestamp")]
            first_date = min(dates) if dates else None
            last_date = max(dates) if dates else None
            
            user_stats.append({
                "id": user_id,
                "stats": {
                    "total": total,
                    "accepted": accepted,
                    "rejected": rejected,
                    "first_date": str(first_date) if first_date else None,
                    "last_date": str(last_date) if last_date else None,
                    "acceptance_rate": accepted / total if total > 0 else 0
                }
            })
        
        return user_stats
    
    @staticmethod
    def find_dashboard_files(output_dir: str, user_stats: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find dashboard HTML files for each user."""
        output_path = Path(output_dir)
        
        for user in user_stats:
            user_id = user["id"]
            
            possible_files = [
                f"{user_id}_diagnostic.html",
                f"{user_id}_interactive.html",
                f"{user_id}.html",
                f"dashboard_enhanced_{user_id}.html",
                f"dashboard_{user_id}.html",
                f"viz_{user_id}.html"
            ]
            
            dashboard_file = None
            for filename in possible_files:
                if (output_path / filename).exists():
                    dashboard_file = filename
                    break
            
            user["dashboard_file"] = dashboard_file
        
        return user_stats
    
    @staticmethod
    def generate_summary_stats(user_stats: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate overall summary statistics."""
        total_users = len(user_stats)
        total_measurements = sum(u["stats"]["total"] for u in user_stats)
        total_accepted = sum(u["stats"]["accepted"] for u in user_stats)
        
        return {
            "total_users": total_users,
            "total_measurements": total_measurements,
            "total_accepted": total_accepted,
            "total_rejected": total_measurements - total_accepted,
            "overall_acceptance_rate": total_accepted / total_measurements if total_measurements > 0 else 0
        }
    
    @staticmethod
    def generate_css() -> str:
        """Generate CSS for the index viewer."""
        return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: #f5f5f5;
            height: 100vh;
            overflow: hidden;
        }
        
        .container {
            display: flex;
            flex-direction: column;
            height: 100vh;
        }
        
        .header {
            background: white;
            border-bottom: 2px solid #e0e0e0;
            padding: 1rem 1.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        
        .header h1 {
            font-size: 1.5rem;
            color: #333;
            margin-bottom: 0.5rem;
        }
        
        .summary-stats {
            display: flex;
            gap: 2rem;
            font-size: 0.9rem;
            color: #666;
        }
        
        .stat-item {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .stat-value {
            font-weight: 600;
            color: #333;
        }
        
        .main-content {
            display: flex;
            flex: 1;
            overflow: hidden;
        }
        
        .sidebar {
            width: 320px;
            background: white;
            border-right: 1px solid #e0e0e0;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        
        .sidebar-header {
            padding: 1rem;
            border-bottom: 1px solid #e0e0e0;
        }
        
        .sort-control {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 0.5rem;
        }
        
        .sort-label {
            font-size: 0.85rem;
            color: #666;
            font-weight: 500;
            margin-right: 0.25rem;
        }
        
        .sort-select {
            flex: 1;
            padding: 0.5rem;
            border: 2px solid #e0e0e0;
            border-radius: 6px;
            font-size: 0.9rem;
            background: white;
            cursor: pointer;
            transition: border-color 0.2s;
            font-weight: 500;
        }
        
        .sort-select:hover {
            border-color: #2196f3;
        }
        
        .sort-select:focus {
            outline: none;
            border-color: #2196f3;
            box-shadow: 0 0 0 3px rgba(33, 150, 243, 0.1);
        }
        
        .sort-direction {
            padding: 0.5rem 0.8rem;
            border: 2px solid #e0e0e0;
            border-radius: 6px;
            background: white;
            cursor: pointer;
            font-size: 1.1rem;
            font-weight: bold;
            transition: all 0.2s;
            color: #2196f3;
        }
        
        .sort-direction:hover {
            background: #f5f5f5;
            border-color: #2196f3;
        }
        
        .sort-direction:active {
            transform: scale(0.95);
        }
        
        .search-box {
            width: 100%;
            padding: 0.5rem;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 0.9rem;
            margin-bottom: 0.75rem;
        }
        
        .quick-sort-buttons {
            display: flex;
            gap: 0.4rem;
            flex-wrap: wrap;
        }
        
        .quick-sort-btn {
            flex: 1;
            min-width: 80px;
            padding: 0.4rem 0.6rem;
            border: 1px solid #e0e0e0;
            border-radius: 4px;
            background: white;
            color: #555;
            font-size: 0.75rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s;
            text-align: center;
        }
        
        .quick-sort-btn:hover {
            background: #e3f2fd;
            border-color: #2196f3;
            color: #1976d2;
        }
        
        .quick-sort-btn:active {
            transform: scale(0.95);
        }
        
        .quick-sort-btn.active {
            background: #2196f3;
            color: white;
            border-color: #1976d2;
        }
        
        .user-list {
            flex: 1;
            overflow-y: auto;
            padding: 0.5rem;
            will-change: scroll-position;
        }
        
        .user-item {
            padding: 0.75rem;
            margin-bottom: 0.5rem;
            border: 1px solid #e0e0e0;
            border-radius: 6px;
            cursor: pointer;
            transition: background-color 0.2s, border-color 0.2s, box-shadow 0.2s;
            background: white;
            will-change: auto;
        }
        
        .user-item:hover {
            background: #f8f9fa;
            border-color: #007bff;
        }
        
        .user-item.selected {
            background: #e3f2fd;
            border-color: #2196f3;
            box-shadow: 0 2px 4px rgba(33, 150, 243, 0.2);
        }
        
        .user-id {
            font-size: 0.85rem;
            font-weight: 500;
            color: #333;
            margin-bottom: 0.5rem;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        
        .user-stats {
            display: flex;
            gap: 1rem;
            font-size: 0.8rem;
        }
        
        .user-stat {
            display: flex;
            align-items: center;
            gap: 0.25rem;
        }
        
        .stat-label {
            color: #666;
        }
        
        .stat-badge {
            padding: 0.15rem 0.4rem;
            border-radius: 12px;
            font-weight: 600;
        }
        
        .badge-total {
            background: #e0e0e0;
            color: #333;
        }
        
        .badge-accepted {
            background: #d4edda;
            color: #155724;
        }
        
        .badge-rejected {
            background: #f8d7da;
            color: #721c24;
        }
        
        .stat-sorted {
            position: relative;
            font-weight: 600;
        }
        
        .stat-sorted .stat-badge {
            box-shadow: 0 0 0 2px rgba(33, 150, 243, 0.3);
            font-weight: 700;
        }
        
        .user-id.stat-sorted {
            color: #2196f3;
            font-weight: 600;
        }
        
        .dashboard-container {
            flex: 1;
            padding: 1rem;
            background: #f5f5f5;
            position: relative;
        }
        
        .dashboard-frame {
            width: 100%;
            height: 100%;
            border: none;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            display: block;
        }
        
        .no-selection {
            display: flex;
            align-items: center;
            justify-content: center;
            height: 100%;
            color: #999;
            font-size: 1.1rem;
        }
        
        .loading {
            display: flex;
            align-items: center;
            justify-content: center;
            height: 100%;
            color: #666;
        }
        
        .error-message {
            display: flex;
            align-items: center;
            justify-content: center;
            height: 100%;
            color: #d32f2f;
            font-size: 1rem;
        }
        
        .keyboard-hint {
            position: absolute;
            bottom: 1rem;
            right: 1rem;
            background: rgba(0, 0, 0, 0.7);
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 4px;
            font-size: 0.85rem;
            opacity: 0;
            transition: opacity 0.3s;
        }
        
        .keyboard-hint.show {
            opacity: 1;
        }
        
        .spinner {
            border: 3px solid #f3f3f3;
            border-top: 3px solid #3498db;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        """
    
    @staticmethod
    def generate_javascript() -> str:
        """Generate JavaScript for the dashboard viewer."""
        return """
        class DashboardViewer {
            constructor(data) {
                this.data = data;
                this.currentUser = null;
                this.sortField = 'id';
                this.sortDirection = 'asc';
                this.filteredUsers = [...data.users];
                this.searchTimer = null;
                this.loadingIframe = false;
                this.init();
            }
            
            init() {
                this.setupEventListeners();
                this.renderUserList();
                this.showKeyboardHint();
                
                if (this.filteredUsers.length > 0) {
                    this.selectUser(this.filteredUsers[0]);
                }
            }
            
            setupEventListeners() {
                document.getElementById('sortSelect').addEventListener('change', (e) => {
                    this.sortField = e.target.value;
                    this.sortUsers();
                    document.querySelectorAll('.quick-sort-btn').forEach(b => {
                        b.classList.remove('active');
                    });
                });
                
                document.getElementById('sortDirection').addEventListener('click', () => {
                    this.sortDirection = this.sortDirection === 'asc' ? 'desc' : 'asc';
                    document.getElementById('sortDirection').textContent =
                        this.sortDirection === 'asc' ? '↑' : '↓';
                    this.sortUsers();
                });
                
                document.getElementById('searchBox').addEventListener('input', (e) => {
                    clearTimeout(this.searchTimer);
                    this.searchTimer = setTimeout(() => {
                        this.filterUsers(e.target.value);
                    }, 150);
                });
                
                document.addEventListener('keydown', (e) => {
                    this.handleKeyboard(e);
                });
                
                document.querySelectorAll('.quick-sort-btn').forEach(btn => {
                    btn.addEventListener('click', (e) => {
                        const sortField = e.target.dataset.sort;
                        const sortDirection = e.target.dataset.direction;
                        
                        document.getElementById('sortSelect').value = sortField;
                        this.sortField = sortField;
                        this.sortDirection = sortDirection;
                        document.getElementById('sortDirection').textContent = 
                            sortDirection === 'asc' ? '↑' : '↓';
                        
                        document.querySelectorAll('.quick-sort-btn').forEach(b => {
                            b.classList.remove('active');
                        });
                        e.target.classList.add('active');
                        
                        this.sortUsers();
                    });
                });
            }
            
            filterUsers(searchTerm) {
                if (!searchTerm) {
                    this.filteredUsers = [...this.data.users];
                } else {
                    const term = searchTerm.toLowerCase();
                    this.filteredUsers = this.data.users.filter(user =>
                        user.id.toLowerCase().includes(term)
                    );
                }
                this.sortUsers();
            }
            
            sortUsers() {
                this.filteredUsers.sort((a, b) => {
                    let aVal, bVal;
                    
                    if (this.sortField === 'id') {
                        aVal = a.id;
                        bVal = b.id;
                    } else if (this.sortField === 'total') {
                        aVal = a.stats.total;
                        bVal = b.stats.total;
                    } else if (this.sortField === 'accepted') {
                        aVal = a.stats.accepted;
                        bVal = b.stats.accepted;
                    } else if (this.sortField === 'rejected') {
                        aVal = a.stats.rejected;
                        bVal = b.stats.rejected;
                    } else if (this.sortField === 'rate') {
                        aVal = a.stats.acceptance_rate;
                        bVal = b.stats.acceptance_rate;
                    } else if (this.sortField === 'first_date') {
                        aVal = a.stats.first_date || '';
                        bVal = b.stats.first_date || '';
                    } else if (this.sortField === 'last_date') {
                        aVal = a.stats.last_date || '';
                        bVal = b.stats.last_date || '';
                    }
                    
                    if (aVal === null || aVal === undefined) aVal = '';
                    if (bVal === null || bVal === undefined) bVal = '';
                    
                    if (typeof aVal === 'string') {
                        return this.sortDirection === 'asc'
                            ? aVal.localeCompare(bVal)
                            : bVal.localeCompare(aVal);
                    } else {
                        return this.sortDirection === 'asc'
                            ? aVal - bVal
                            : bVal - aVal;
                    }
                });
                
                this.renderUserList();
                
                if (this.currentUser && !this.filteredUsers.includes(this.currentUser)) {
                    if (this.filteredUsers.length > 0) {
                        this.selectUser(this.filteredUsers[0]);
                    } else {
                        this.currentUser = null;
                        this.showNoSelection();
                    }
                }
            }
            
            renderUserList() {
                const container = document.getElementById('userList');
                
                if (this.filteredUsers.length === 0) {
                    container.innerHTML = '<div style="padding: 1rem; text-align: center; color: #999;">No users found</div>';
                    return;
                }
                
                container.innerHTML = '';
                
                this.filteredUsers.forEach(user => {
                    const item = document.createElement('div');
                    item.className = 'user-item';
                    item.dataset.userId = user.id;
                    
                    const acceptanceRate = (user.stats.acceptance_rate * 100).toFixed(1);
                    
                    const highlightClass = (field) => {
                        return this.sortField === field ? 'stat-sorted' : '';
                    };
                    
                    item.innerHTML = `
                        <div class="user-id ${highlightClass('id')}" title="${user.id}">${user.id}</div>
                        <div class="user-stats">
                            <div class="user-stat ${highlightClass('total')}">
                                <span class="stat-label">Total:</span>
                                <span class="stat-badge badge-total">${user.stats.total}</span>
                            </div>
                            <div class="user-stat ${highlightClass('accepted')}">
                                <span class="stat-label">✓</span>
                                <span class="stat-badge badge-accepted">${user.stats.accepted}</span>
                            </div>
                            <div class="user-stat ${highlightClass('rejected')}">
                                <span class="stat-label">✗</span>
                                <span class="stat-badge badge-rejected">${user.stats.rejected}</span>
                            </div>
                            <div class="user-stat ${highlightClass('rate')}">
                                <span class="stat-label">${acceptanceRate}%</span>
                            </div>
                        </div>
                    `;
                    
                    item.addEventListener('click', () => this.selectUser(user));
                    
                    if (this.currentUser && this.currentUser.id === user.id) {
                        item.classList.add('selected');
                    } else {
                        item.classList.remove('selected');
                    }
                    
                    container.appendChild(item);
                });
                
                if (this.currentUser) {
                    requestAnimationFrame(() => {
                        this.scrollToUser(this.currentUser);
                    });
                }
            }
            
            selectUser(user) {
                if (this.currentUser === user) return;
                
                this.currentUser = user;
                
                document.querySelectorAll('.user-item').forEach(item => {
                    if (item.dataset.userId === user.id) {
                        item.classList.add('selected');
                    } else {
                        item.classList.remove('selected');
                    }
                });
                
                this.loadDashboard(user);
                requestAnimationFrame(() => {
                    this.scrollToUser(user);
                });
            }
            
            loadDashboard(user) {
                if (this.loadingIframe) return;
                
                const container = document.getElementById('dashboardContainer');
                
                if (!user.dashboard_file) {
                    container.innerHTML = `
                        <div class="error-message">
                            Dashboard not found for user ${user.id}
                        </div>
                    `;
                    return;
                }
                
                const existingIframe = container.querySelector('iframe');
                if (existingIframe && existingIframe.src.endsWith(user.dashboard_file)) {
                    return;
                }
                
                this.loadingIframe = true;
                
                const iframe = document.createElement('iframe');
                iframe.className = 'dashboard-frame';
                iframe.style.visibility = 'hidden';
                iframe.src = user.dashboard_file;
                
                let loadHandled = false;
                let spinnerTimer = null;
                
                spinnerTimer = setTimeout(() => {
                    if (!loadHandled) {
                        container.innerHTML = '<div class="loading"><div class="spinner"></div></div>';
                    }
                }, 100);
                
                const finishLoading = () => {
                    if (!loadHandled) {
                        loadHandled = true;
                        clearTimeout(spinnerTimer);
                        iframe.style.visibility = 'visible';
                        container.innerHTML = '';
                        container.appendChild(iframe);
                        this.loadingIframe = false;
                        this.preloadAdjacent();
                    }
                };
                
                iframe.onload = finishLoading;
                
                iframe.onerror = () => {
                    if (!loadHandled) {
                        loadHandled = true;
                        clearTimeout(spinnerTimer);
                        container.innerHTML = `
                            <div class="error-message">
                                Failed to load dashboard for user ${user.id}
                            </div>
                        `;
                        this.loadingIframe = false;
                    }
                };
                
                setTimeout(() => {
                    if (!loadHandled) {
                        finishLoading();
                    }
                }, 10);
                
                container.appendChild(iframe);
            }
            
            preloadAdjacent() {
                if (!this.currentUser) return;
                
                const currentIndex = this.filteredUsers.findIndex(u => u.id === this.currentUser.id);
                if (currentIndex === -1) return;
                
                const preloadUser = (user) => {
                    if (!user || !user.dashboard_file) return;
                    const link = document.createElement('link');
                    link.rel = 'prefetch';
                    link.href = user.dashboard_file;
                    link.as = 'document';
                    document.head.appendChild(link);
                };
                
                if (currentIndex > 0) {
                    preloadUser(this.filteredUsers[currentIndex - 1]);
                }
                if (currentIndex < this.filteredUsers.length - 1) {
                    preloadUser(this.filteredUsers[currentIndex + 1]);
                }
            }
            
            showNoSelection() {
                const container = document.getElementById('dashboardContainer');
                container.innerHTML = `
                    <div class="no-selection">
                        Select a user from the list to view their dashboard
                    </div>
                `;
            }
            
            scrollToUser(user) {
                const item = document.querySelector(`[data-user-id="${user.id}"]`);
                if (item) {
                    const container = item.parentElement;
                    const itemRect = item.getBoundingClientRect();
                    const containerRect = container.getBoundingClientRect();
                    
                    if (itemRect.top < containerRect.top || itemRect.bottom > containerRect.bottom) {
                        item.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                    }
                }
            }
            
            handleKeyboard(e) {
                if (e.target.tagName === 'INPUT') return;
                
                const currentIndex = this.filteredUsers.findIndex(u => u.id === this.currentUser?.id);
                
                if (e.key === 'ArrowDown') {
                    e.preventDefault();
                    if (currentIndex < this.filteredUsers.length - 1) {
                        this.selectUser(this.filteredUsers[currentIndex + 1]);
                    }
                } else if (e.key === 'ArrowUp') {
                    e.preventDefault();
                    if (currentIndex > 0) {
                        this.selectUser(this.filteredUsers[currentIndex - 1]);
                    }
                } else if (e.key === 'Home') {
                    e.preventDefault();
                    if (this.filteredUsers.length > 0) {
                        this.selectUser(this.filteredUsers[0]);
                    }
                } else if (e.key === 'End') {
                    e.preventDefault();
                    if (this.filteredUsers.length > 0) {
                        this.selectUser(this.filteredUsers[this.filteredUsers.length - 1]);
                    }
                }
            }
            
            showKeyboardHint() {
                const hint = document.getElementById('keyboardHint');
                if (hint) {
                    hint.classList.add('show');
                    setTimeout(() => {
                        hint.classList.remove('show');
                    }, 3000);
                }
            }
        }
        
        document.addEventListener('DOMContentLoaded', () => {
            if (typeof DASHBOARD_DATA !== 'undefined') {
                window.viewer = new DashboardViewer(DASHBOARD_DATA);
            }
        });
        """
    
    @staticmethod
    def generate_index_html(
        user_stats: List[Dict[str, Any]],
        summary: Dict[str, Any],
        output_dir: str,
        generated_time: Optional[str] = None
    ) -> str:
        """Generate the complete index.html content."""
        
        if not generated_time:
            generated_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        data = {
            "generated": generated_time,
            "output_dir": os.path.basename(output_dir),
            "users": user_stats,
            "summary": summary
        }
        
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Weight Stream Processor - Dashboard Viewer</title>
    <style>
{IndexVisualizer.generate_css()}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Weight Stream Processor Dashboard Viewer</h1>
            <div class="summary-stats">
                <div class="stat-item">
                    <span>Users:</span>
                    <span class="stat-value">{summary['total_users']:,}</span>
                </div>
                <div class="stat-item">
                    <span>Measurements:</span>
                    <span class="stat-value">{summary['total_measurements']:,}</span>
                </div>
                <div class="stat-item">
                    <span>Accepted:</span>
                    <span class="stat-value" style="color: #28a745;">{summary['total_accepted']:,}</span>
                </div>
                <div class="stat-item">
                    <span>Rejected:</span>
                    <span class="stat-value" style="color: #dc3545;">{summary['total_rejected']:,}</span>
                </div>
                <div class="stat-item">
                    <span>Acceptance Rate:</span>
                    <span class="stat-value">{summary['overall_acceptance_rate']:.1%}</span>
                </div>
            </div>
        </div>
        
        <div class="main-content">
            <div class="sidebar" id="sidebar">
                <div class="sidebar-header">
                    <div class="sort-control">
                        <span class="sort-label">Sort by:</span>
                        <select id="sortSelect" class="sort-select">
                            <option value="id">User ID</option>
                            <option value="total">Total Measurements</option>
                            <option value="accepted">Accepted Count</option>
                            <option value="rejected">Rejected Count</option>
                            <option value="rate">Acceptance Rate (%)</option>
                            <option value="first_date">First Date</option>
                            <option value="last_date">Last Date</option>
                        </select>
                        <button id="sortDirection" class="sort-direction" title="Click to reverse sort">↑</button>
                    </div>
                    <input type="text" id="searchBox" class="search-box" placeholder="Search users...">
                    <div class="quick-sort-buttons">
                        <button class="quick-sort-btn" data-sort="rejected" data-direction="desc" title="Most rejected first">
                            Most Rejected
                        </button>
                        <button class="quick-sort-btn" data-sort="rate" data-direction="asc" title="Lowest acceptance rate first">
                            Lowest Rate
                        </button>
                        <button class="quick-sort-btn" data-sort="total" data-direction="desc" title="Most measurements first">
                            Most Data
                        </button>
                    </div>
                </div>
                <div class="user-list" id="userList">
                </div>
            </div>
            
            <div class="dashboard-container" id="dashboardContainer">
                <div class="no-selection">
                    Select a user from the list to view their dashboard
                </div>
            </div>
            
            <div class="keyboard-hint" id="keyboardHint">
                Use ↑↓ arrow keys to navigate • Home/End to jump
            </div>
        </div>
    </div>
    
    <script>
    const DASHBOARD_DATA = {json.dumps(data)};
    </script>
    <script>
{IndexVisualizer.generate_javascript()}
    </script>
</body>
</html>"""
        
        return html_content
    
    @staticmethod
    def create_index_from_results(all_results: Dict[str, List[Dict]], output_dir: str = "output") -> str:
        """Create an index.html with user list sidebar and iframe viewer."""
        
        # Extract user statistics
        user_stats = IndexVisualizer.extract_user_stats(all_results)
        
        # Find dashboard files for each user
        user_stats = IndexVisualizer.find_dashboard_files(output_dir, user_stats)
        
        # Generate summary statistics
        summary = IndexVisualizer.generate_summary_stats(user_stats)
        
        # Generate HTML content
        html_content = IndexVisualizer.generate_index_html(
            user_stats,
            summary,
            output_dir
        )
        
        # Save to file
        output_path = Path(output_dir) / "index.html"
        output_path.parent.mkdir(exist_ok=True, parents=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return str(output_path)


# ============================================================================
# SECTION 8: Export Functions
# ============================================================================

def export_dashboard(dashboard_path: str, 
                    export_format: str,
                    output_path: Optional[str] = None) -> str:
    """Export dashboard to different formats."""
    
    if not Path(dashboard_path).exists():
        raise FileNotFoundError(f"Dashboard not found: {dashboard_path}")
    
    export_format = export_format.lower()
    
    if export_format == 'pdf':
        return export_to_pdf(dashboard_path, output_path)
    elif export_format == 'png':
        return export_to_png(dashboard_path, output_path)
    elif export_format == 'html':
        # Already HTML, just copy if needed
        if output_path and output_path != dashboard_path:
            import shutil
            shutil.copy(dashboard_path, output_path)
            return output_path
        return dashboard_path
    else:
        raise ValueError(f"Unsupported export format: {export_format}")


def export_to_pdf(html_path: str, output_path: Optional[str] = None) -> str:
    """Export HTML dashboard to PDF (requires additional libraries)."""
    # This would require pdfkit or similar
    # For now, return a message
    return f"PDF export not implemented. Please use browser print function on {html_path}"


def export_to_png(html_path: str, output_path: Optional[str] = None) -> str:
    """Export HTML dashboard to PNG (requires additional libraries)."""
    # This would require selenium or similar
    # For now, return a message
    return f"PNG export not implemented. Please use browser screenshot function on {html_path}"


# ============================================================================
# SECTION 9: Main Factory Function
# ============================================================================

def create_dashboard(results: List[Dict[str, Any]], 
                    user_id: str,
                    output_dir: str = "output",
                    config: Optional[Dict[str, Any]] = None,
                    output_format: Optional[str] = None,
                    dashboard_type: Optional[str] = None) -> str:
    """
    Main factory function to create appropriate dashboard.
    
    Args:
        results: Processing results
        user_id: User identifier
        output_dir: Output directory
        config: Configuration dictionary
        output_format: Desired output format (html, png, pdf)
        dashboard_type: Type of dashboard (diagnostic, interactive, static)
    
    Returns:
        Path to generated dashboard
    """
    
    if config is None:
        config = load_config()
    
    # Determine dashboard type
    if dashboard_type is None:
        viz_config = config.get("visualization", {})
        
        # Check config preferences
        if viz_config.get("use_diagnostic", False):
            dashboard_type = "diagnostic"
        elif should_use_interactive(config, output_format):
            dashboard_type = "interactive"
        else:
            dashboard_type = "static"
    
    # Create appropriate dashboard
    if dashboard_type == "diagnostic":
        dashboard = DiagnosticDashboard(config)
    elif dashboard_type == "interactive" and PLOTLY_AVAILABLE:
        dashboard = InteractiveDashboard(config)
    else:
        dashboard = StaticDashboard(config)
    
    # Generate dashboard
    dashboard_path = dashboard.create_dashboard(results, user_id, config, output_dir)
    
    # Export if different format requested
    if output_format and output_format != "html":
        dashboard_path = export_dashboard(dashboard_path, output_format)
    
    return dashboard_path


# ============================================================================
# SECTION 10: Backward Compatibility
# ============================================================================

# Import functions that might be used elsewhere
def create_diagnostic_report(results: List[Dict[str, Any]], user_id: str) -> str:
    """Backward compatibility wrapper."""
    dashboard = DiagnosticDashboard()
    return dashboard._create_diagnostic_report(results)


def create_index_from_results(all_results: Dict[str, List[Dict]], output_dir: str = "output") -> str:
    """Backward compatibility wrapper."""
    return IndexVisualizer.create_index_from_results(all_results, output_dir)


# For direct module execution
if __name__ == "__main__":
    print("Unified Visualization Module")
    print("Available dashboards: static, interactive, diagnostic")
    print("Use create_dashboard() function to generate visualizations")