"""
Enhanced user dashboard with comprehensive visualizations.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import matplotlib.patches as mpatches
import logging
import warnings

# Suppress matplotlib warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

logger = logging.getLogger(__name__)

# Set matplotlib style directly instead of using seaborn
plt.style.use('seaborn-v0_8-whitegrid') if 'seaborn-v0_8-whitegrid' in plt.style.available else plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 10


class UserDashboard:
    """Creates comprehensive visualization dashboard for a single user."""
    
    def __init__(self, user_id: str, results: Dict[str, Any], output_dir: str = "output/visualizations"):
        self.user_id = user_id
        self.results = results
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract time series data
        self.time_series = results.get('time_series', [])
        if not self.time_series:
            logger.warning(f"No time series data for user {user_id}")
            
    def create_dashboard(self) -> Optional[Path]:
        """
        Create a comprehensive dashboard with multiple plots.
        
        Returns:
            Path to saved dashboard image, or None if insufficient data
        """
        if len(self.time_series) < 3:
            logger.warning(f"Insufficient data for visualization: {len(self.time_series)} points")
            return None
            
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle(f'Weight Analysis Dashboard - User: {self.user_id}', fontsize=16, fontweight='bold')
        
        # Create grid spec for better layout control
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Weight trajectory with confidence bands
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_weight_trajectory(ax1)
        
        # 2. Trend analysis
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_trend_analysis(ax2)
        
        # 3. Innovation/residual analysis
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_innovation_analysis(ax3)
        
        # 4. Confidence distribution
        ax4 = fig.add_subplot(gs[1, 2])
        self._plot_confidence_distribution(ax4)
        
        # 5. Outlier analysis
        ax5 = fig.add_subplot(gs[2, 0])
        self._plot_outlier_analysis(ax5)
        
        # 6. Weekly pattern analysis
        ax6 = fig.add_subplot(gs[2, 1])
        self._plot_weekly_pattern(ax6)
        
        # 7. Statistics summary
        ax7 = fig.add_subplot(gs[2, 2])
        self._plot_statistics_summary(ax7)
        
        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"dashboard_{self.user_id}_{timestamp}.png"
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Dashboard created for user {self.user_id}: {output_path}")
        return output_path
        
    def _plot_weight_trajectory(self, ax):
        """Plot weight over time with filtered values and confidence bands."""
        dates = []
        for d in self.time_series:
            date_str = d['date']
            # Handle both formats: with Z suffix and without
            if 'Z' in date_str:
                dates.append(datetime.fromisoformat(date_str.replace('Z', '+00:00')))
            else:
                dates.append(datetime.fromisoformat(date_str))
        weights = [d['weight'] for d in self.time_series]
        filtered = [d.get('filtered_weight', d['weight']) for d in self.time_series]
        confidence = [d.get('confidence', 0.5) for d in self.time_series]
        sources = [d.get('source', 'unknown') for d in self.time_series]
        
        # Color points by confidence score
        colors = []
        for conf in confidence:
            if conf >= 0.9:
                colors.append('#4CAF50')  # Green - high confidence
            elif conf >= 0.75:
                colors.append('#8BC34A')  # Light green
            elif conf >= 0.6:
                colors.append('#FFC107')  # Yellow
            else:
                colors.append('#FF9800')  # Orange - low confidence
        
        # Plot measurements with color coding
        for i, (date, weight, color, source) in enumerate(zip(dates, weights, colors, sources)):
            marker = 'o'
            # Handle None or empty source
            if source:
                source_lower = source.lower()
                if 'questionnaire' in source_lower:
                    marker = 'D'  # Diamond for questionnaire
                elif 'care' in source_lower:
                    marker = '^'  # Triangle for care team
                elif 'device' in source_lower:
                    marker = 's'  # Square for device
                
            ax.scatter(date, weight, c=color, marker=marker, s=30, alpha=0.8, edgecolors='black', linewidth=0.5)
        
        # Only plot filtered if available
        if any(f is not None for f in filtered):
            # Replace None with raw values for continuity
            filtered_clean = [f if f is not None else w for f, w in zip(filtered, weights)]
            ax.plot(dates, filtered_clean, '-', linewidth=2, label='Filtered weight', color='blue')
        else:
            ax.plot(dates, weights, '-', linewidth=1, label='Weight', color='blue')
        
        # Add confidence shading if we have filtered values
        if any(f is not None for f in filtered):
            filtered_clean = np.array([f if f is not None else w for f, w in zip(filtered, weights)])
            confidence_array = np.array(confidence)
            ax.fill_between(dates, 
                            filtered_clean - (1 - confidence_array) * 2,
                            filtered_clean + (1 - confidence_array) * 2,
                            alpha=0.2, color='blue', label='Uncertainty')
        
        # Add baseline if available
        if 'baseline' in self.results:
            baseline_weight = self.results['baseline'].get('weight')
            if baseline_weight:
                ax.axhline(y=baseline_weight, color='green', linestyle='--', 
                          alpha=0.5, label=f'Baseline: {baseline_weight:.1f}kg')
        
        # Highlight rejected measurements
        rejected_dates = [dates[i] for i, d in enumerate(self.time_series) if not d.get('is_valid', True)]
        rejected_weights = [weights[i] for i, d in enumerate(self.time_series) if not d.get('is_valid', True)]
        if rejected_dates:
            ax.scatter(rejected_dates, rejected_weights, color='red', s=50, 
                      marker='x', label='Rejected', zorder=5)
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Weight (kg)')
        ax.set_title('Weight Trajectory with Kalman Filtering')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Format x-axis - fixed to avoid too many ticks
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        # Use appropriate locator based on date range
        date_range = (dates[-1] - dates[0]).days if dates else 0
        if date_range > 365:
            ax.xaxis.set_major_locator(mdates.MonthLocator())
        elif date_range > 60:
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        elif date_range > 14:
            ax.xaxis.set_major_locator(mdates.WeekdayLocator())
        else:
            ax.xaxis.set_major_locator(mdates.DayLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
    def _plot_trend_analysis(self, ax):
        """Plot trend over time."""
        dates = []
        for d in self.time_series:
            date_str = d['date']
            if 'Z' in date_str:
                dates.append(datetime.fromisoformat(date_str.replace('Z', '+00:00')))
            else:
                dates.append(datetime.fromisoformat(date_str))
        trends = [(d.get('trend_kg_per_day', 0) or 0) * 7 for d in self.time_series]  # Convert to weekly, handle None
        
        # Plot trend
        ax.plot(dates, trends, '-', linewidth=2, color='purple')
        ax.fill_between(dates, 0, trends, alpha=0.3, color='purple')
        
        # Add zero line
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add average trend
        if trends:
            avg_trend = np.mean(trends)
            ax.axhline(y=avg_trend, color='red', linestyle='--', 
                      label=f'Avg: {avg_trend:.2f}kg/week')
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Trend (kg/week)')
        ax.set_title('Weight Change Trend')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format x-axis - fixed to avoid too many ticks
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        # Use appropriate locator based on date range
        if dates:
            date_range = (dates[-1] - dates[0]).days
            if date_range > 365:
                ax.xaxis.set_major_locator(mdates.MonthLocator())
            elif date_range > 60:
                ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
            elif date_range > 14:
                ax.xaxis.set_major_locator(mdates.WeekdayLocator())
            else:
                ax.xaxis.set_major_locator(mdates.DayLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
    def _plot_innovation_analysis(self, ax):
        """Plot prediction errors (innovations)."""
        innovations = []
        for d in self.time_series:
            if 'predicted_weight' in d and d['predicted_weight'] is not None:
                innovation = d['weight'] - d['predicted_weight']
                innovations.append(innovation)
        
        if not innovations:
            ax.text(0.5, 0.5, 'No innovation data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Innovation Analysis')
            return
            
        # Histogram of innovations
        ax.hist(innovations, bins=20, edgecolor='black', alpha=0.7, color='orange')
        
        # Add normal distribution overlay
        if len(innovations) > 5:
            mu, std = np.mean(innovations), np.std(innovations)
            x = np.linspace(min(innovations), max(innovations), 100)
            ax2 = ax.twinx()
            ax2.plot(x, np.exp(-(x - mu)**2 / (2 * std**2)) / (std * np.sqrt(2 * np.pi)), 
                    'r-', linewidth=2, label='Normal fit')
            ax2.set_ylabel('Density')
        
        ax.set_xlabel('Innovation (kg)')
        ax.set_ylabel('Frequency')
        ax.set_title('Prediction Error Distribution')
        ax.grid(True, alpha=0.3)
        
    def _plot_confidence_distribution(self, ax):
        """Plot distribution of confidence scores."""
        confidences = [d.get('confidence', 0.5) for d in self.time_series]
        
        # Create bins for confidence levels
        bins = [0, 0.3, 0.5, 0.8, 0.95, 1.0]
        labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
        
        # Count confidences in each bin
        hist, _ = np.histogram(confidences, bins=bins)
        
        # Create bar chart
        colors = ['red', 'orange', 'yellow', 'lightgreen', 'green']
        bars = ax.bar(labels, hist, color=colors, edgecolor='black')
        
        # Add value labels on bars
        for bar, count in zip(bars, hist):
            if count > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       str(count), ha='center', va='bottom')
        
        ax.set_xlabel('Confidence Level')
        ax.set_ylabel('Count')
        ax.set_title('Measurement Confidence Distribution')
        ax.grid(True, alpha=0.3, axis='y')
        
    def _plot_outlier_analysis(self, ax):
        """Plot outlier types and frequencies."""
        outlier_types = {}
        for d in self.time_series:
            if not d.get('is_valid', True) and d.get('outlier_type'):
                outlier_type = d['outlier_type']
                outlier_types[outlier_type] = outlier_types.get(outlier_type, 0) + 1
        
        if not outlier_types:
            # No outliers - show acceptance rate
            valid_count = sum(1 for d in self.time_series if d.get('is_valid', True))
            total = len(self.time_series)
            acceptance_rate = valid_count / total if total > 0 else 1.0
            
            ax.pie([valid_count, total - valid_count], 
                  labels=['Accepted', 'Rejected'],
                  colors=['green', 'red'],
                  autopct='%1.1f%%',
                  startangle=90)
            ax.set_title(f'Data Quality\n({acceptance_rate:.1%} Acceptance Rate)')
        else:
            # Show outlier breakdown
            ax.pie(outlier_types.values(), 
                  labels=outlier_types.keys(),
                  autopct='%1.0f',
                  startangle=90)
            ax.set_title('Outlier Type Distribution')
            
    def _plot_weekly_pattern(self, ax):
        """Analyze weekly patterns in weight."""
        # Group by day of week
        weekday_weights = {i: [] for i in range(7)}
        weekday_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        
        for d in self.time_series:
            if d.get('is_valid', True):
                date_str = d['date']
                if 'Z' in date_str:
                    date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                else:
                    date = datetime.fromisoformat(date_str)
                weekday = date.weekday()
                weight = d.get('filtered_weight', d['weight'])
                weekday_weights[weekday].append(weight)
        
        # Calculate averages and std
        avg_weights = []
        std_weights = []
        for i in range(7):
            if weekday_weights[i]:
                avg_weights.append(np.mean(weekday_weights[i]))
                std_weights.append(np.std(weekday_weights[i]))
            else:
                avg_weights.append(None)
                std_weights.append(None)
        
        # Filter out None values for plotting
        plot_days = [i for i in range(7) if avg_weights[i] is not None]
        plot_names = [weekday_names[i] for i in plot_days]
        plot_avgs = [avg_weights[i] for i in plot_days]
        plot_stds = [std_weights[i] for i in plot_days]
        
        if plot_avgs:
            # Normalize to show variation from mean
            overall_mean = np.mean([w for weights in weekday_weights.values() 
                                   for w in weights if weights])
            plot_avgs_normalized = [a - overall_mean for a in plot_avgs]
            
            bars = ax.bar(plot_names, plot_avgs_normalized, yerr=plot_stds,
                         capsize=5, color='skyblue', edgecolor='black')
            
            # Color bars based on deviation
            for bar, val in zip(bars, plot_avgs_normalized):
                if val > 0:
                    bar.set_color('salmon')
                else:
                    bar.set_color('lightgreen')
            
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax.set_xlabel('Day of Week')
            ax.set_ylabel('Deviation from Mean (kg)')
            ax.set_title('Weekly Weight Pattern')
        else:
            ax.text(0.5, 0.5, 'Insufficient data for weekly pattern', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Weekly Weight Pattern')
            
        ax.grid(True, alpha=0.3, axis='y')
        
    def _plot_statistics_summary(self, ax):
        """Display key statistics as text."""
        ax.axis('off')
        
        # Gather statistics
        stats_text = "📊 **KEY STATISTICS**\n\n"
        
        # Basic stats
        stats = self.results.get('stats', {})
        stats_text += f"Total Readings: {stats.get('total_readings', 0)}\n"
        stats_text += f"Accepted: {stats.get('accepted_readings', 0)}\n"
        stats_text += f"Rejected: {stats.get('rejected_readings', 0)}\n"
        
        acceptance_rate = stats.get('acceptance_rate', 0)
        if acceptance_rate:
            stats_text += f"Acceptance Rate: {acceptance_rate:.1%}\n"
        
        stats_text += "\n"
        
        # Weight stats
        stats_text += f"Min Weight: {stats.get('min_weight', 0):.1f} kg\n"
        stats_text += f"Max Weight: {stats.get('max_weight', 0):.1f} kg\n"
        stats_text += f"Range: {stats.get('weight_range', 0):.1f} kg\n"
        
        if 'average_weight' in stats:
            stats_text += f"Average: {stats['average_weight']:.1f} kg\n"
        
        stats_text += "\n"
        
        # Baseline info
        if 'baseline' in self.results:
            baseline = self.results['baseline']
            stats_text += f"Baseline: {baseline.get('weight', 0):.1f} kg\n"
            stats_text += f"Confidence: {baseline.get('confidence', 'unknown')}\n"
        
        # Current state
        if 'current_state' in self.results:
            current = self.results['current_state']
            stats_text += f"\nCurrent Weight: {current.get('weight', 0):.1f} kg\n"
            trend_week = current.get('trend_kg_per_week', 0)
            if trend_week != 0:
                direction = "📈" if trend_week > 0 else "📉"
                stats_text += f"Trend: {direction} {abs(trend_week):.2f} kg/week\n"
        
        # Display text
        ax.text(0.1, 0.9, stats_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               fontfamily='monospace')


def create_user_visualizations(
    results: Dict[str, Dict[str, Any]], 
    output_dir: str = "output/visualizations",
    max_users: Optional[int] = None
) -> Dict[str, Path]:
    """
    Create visualizations for multiple users.
    
    Args:
        results: Dictionary of user results
        output_dir: Output directory for visualizations
        max_users: Maximum number of users to visualize (None for all)
        
    Returns:
        Dictionary mapping user_id to dashboard path
    """
    dashboard_paths = {}
    users_to_process = list(results.keys())
    
    if max_users:
        users_to_process = users_to_process[:max_users]
    
    logger.info(f"Creating visualizations for {len(users_to_process)} users")
    
    for i, user_id in enumerate(users_to_process, 1):
        try:
            dashboard = UserDashboard(user_id, results[user_id], output_dir)
            path = dashboard.create_dashboard()
            if path:
                dashboard_paths[user_id] = path
                logger.info(f"Created dashboard {i}/{len(users_to_process)} for user {user_id}")
        except Exception as e:
            logger.error(f"Failed to create dashboard for user {user_id}: {e}")
            
    logger.info(f"Successfully created {len(dashboard_paths)} dashboards")
    return dashboard_paths