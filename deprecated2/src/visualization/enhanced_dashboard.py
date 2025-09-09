"""
Enhanced dashboard for pure Kalman filter pipeline visualization.
Comprehensive insights into: state estimation, prediction errors, innovation analysis,
measurement validation, confidence evolution, and filter performance metrics.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import matplotlib.patches as mpatches
import logging
import warnings
from scipy import stats

warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

logger = logging.getLogger(__name__)

plt.rcParams['figure.figsize'] = (24, 20)
plt.rcParams['font.size'] = 9


class EnhancedDashboard:
    """Creates comprehensive Kalman filter visualization dashboard."""

    SOURCE_STYLES = {
        'care-team-upload': {'marker': '^', 'color': '#2E7D32', 'label': 'Care Team', 'size': 100},
        'patient-device': {'marker': 's', 'color': '#1565C0', 'label': 'Patient Device', 'size': 80},
        'internal-questionnaire': {'marker': 'D', 'color': '#7B1FA2', 'label': 'Questionnaire', 'size': 120},
        'patient-upload': {'marker': 'v', 'color': '#E65100', 'label': 'Patient Upload', 'size': 80},
        'https://connectivehealth.io': {'marker': 'p', 'color': '#00796B', 'label': 'Connective Health', 'size': 90},
        'https://api.iglucose.com': {'marker': 'h', 'color': '#FF6F00', 'label': 'iGlucose', 'size': 90},
        'unknown': {'marker': 'o', 'color': '#616161', 'label': 'Other', 'size': 60}
    }

    def __init__(self, user_id: str, results: Dict[str, Any], output_dir: str = "output/visualizations"):
        self.user_id = user_id
        self.results = results
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.time_series = results.get('time_series', [])
        self.baseline = results.get('baseline', {})
        self.current_state = results.get('current_state', {})
        self.stats = results.get('stats', {})

    def create_dashboard(self) -> Optional[Path]:
        """Create comprehensive Kalman filter insights dashboard."""
        if len(self.time_series) < 2:
            logger.warning(f"Insufficient data for visualization")
            return None

        fig = plt.figure(figsize=(24, 24), constrained_layout=False)

        # Enhanced Layout: 6 rows x 3 columns
        # Row 0: Full timeline overview (compressed height)
        ax_overview = plt.subplot2grid((6, 3), (0, 0), colspan=3, rowspan=1)

        # Row 1: Main timeline with Kalman state (cropped to 2025+)
        ax1 = plt.subplot2grid((6, 3), (1, 0), colspan=3, rowspan=1)

        # Row 2: Innovation and prediction errors (cropped)
        ax2 = plt.subplot2grid((6, 3), (2, 0), colspan=2, rowspan=1)
        ax3 = plt.subplot2grid((6, 3), (2, 2), colspan=1, rowspan=1)

        # Row 3: Confidence evolution (cropped) and validation gate
        ax4 = plt.subplot2grid((6, 3), (3, 0), colspan=2, rowspan=1)
        ax5 = plt.subplot2grid((6, 3), (3, 2), colspan=1, rowspan=1)

        # Row 4: Trend analysis (cropped) and state uncertainty
        ax6 = plt.subplot2grid((6, 3), (4, 0), colspan=2, rowspan=1)
        ax7 = plt.subplot2grid((6, 3), (4, 2), colspan=1, rowspan=1)

        # Row 5: Statistics panels
        ax8 = plt.subplot2grid((6, 3), (5, 0), colspan=1, rowspan=1)
        ax9 = plt.subplot2grid((6, 3), (5, 1), colspan=1, rowspan=1)
        ax10 = plt.subplot2grid((6, 3), (5, 2), colspan=1, rowspan=1)

        # Plot all panels
        self._plot_overview_timeline(ax_overview)  # Full overview plot
        self._plot_main_timeline_with_kalman(ax1, focus_start_date=datetime(2025, 1, 1))
        self._plot_innovation_sequence(ax2, focus_start_date=datetime(2025, 1, 1))
        self._plot_innovation_histogram(ax3)
        self._plot_confidence_evolution(ax4, focus_start_date=datetime(2025, 1, 1))
        self._plot_validation_gate_analysis(ax5)
        self._plot_trend_analysis(ax6, focus_start_date=datetime(2025, 1, 1))
        self._plot_state_uncertainty(ax7, focus_start_date=datetime(2025, 1, 1))
        self._plot_kalman_statistics(ax8)
        self._plot_weight_distribution(ax9)
        self._plot_acceptance_metrics(ax10)

        # Title and metadata
        fig.suptitle(f'Pure Kalman Filter Analysis - User: {self.user_id}',
                    fontsize=16, fontweight='bold', y=0.98)

        fig.text(0.99, 0.01, f'Generated: {datetime.now():%Y-%m-%d %H:%M}',
                ha='right', va='bottom', fontsize=8, alpha=0.5)

        plt.tight_layout(rect=[0, 0.02, 1, 0.96])

        # Save
        output_file = self.output_dir / f"kalman_dashboard-{self.user_id}.png"
        plt.savefig(output_file, dpi=100, bbox_inches='tight', facecolor='white')
        plt.close()

        logger.info(f"Created Kalman dashboard for user {self.user_id[:8]}...")
        return output_file

    def _plot_overview_timeline(self, ax):
        """Plot full timeline overview showing all data points."""
        dates = []
        weights = []
        filtered_weights = []
        is_valid = []

        for ts in self.time_series:
            dates.append(self._parse_date(ts['date']))
            weights.append(ts['weight'])
            filtered_weights.append(ts.get('filtered_weight'))
            is_valid.append(ts.get('is_valid', True))

        if not dates:
            return

        # Plot all measurements with smaller markers
        valid_dates = [d for d, v in zip(dates, is_valid) if v]
        valid_weights = [w for w, v in zip(weights, is_valid) if v]
        invalid_dates = [d for d, v in zip(dates, is_valid) if not v]
        invalid_weights = [w for w, v in zip(weights, is_valid) if not v]

        if valid_dates:
            ax.scatter(valid_dates, valid_weights, c='#4CAF50', s=8, alpha=0.5, label=f'Valid ({len(valid_dates)})')
        if invalid_dates:
            ax.scatter(invalid_dates, invalid_weights, c='#FF4444', s=12, alpha=0.6, marker='x', label=f'Rejected ({len(invalid_dates)})')

        # Plot Kalman filtered trajectory
        valid_filtered = [(d, f) for d, f, v in zip(dates, filtered_weights, is_valid)
                         if f is not None and v]
        if valid_filtered:
            f_dates, f_weights = zip(*valid_filtered)
            ax.plot(f_dates, f_weights, 'g-', linewidth=1.5, alpha=0.8, label='Kalman Filtered')

        # Baseline
        if self.baseline and self.baseline.get('weight'):
            ax.axhline(y=self.baseline['weight'], color='#9C27B0', linestyle='--',
                      linewidth=1, alpha=0.6, label=f'Baseline: {self.baseline["weight"]:.1f}kg')

        # Highlight focus region (2025 onwards)
        focus_start = datetime(2025, 1, 1)
        if dates and max(dates) > focus_start:
            ax.axvspan(focus_start, max(dates), alpha=0.15, color='orange')
            ax.axvline(x=focus_start, color='orange', linestyle='--', linewidth=2,
                      alpha=0.7, label='2025+ Focus →')

            # Add stats for pre/post 2025
            pre_2025 = sum(1 for d in dates if d < focus_start)
            post_2025 = sum(1 for d in dates if d >= focus_start)
            ax.text(0.98, 0.95, f'Pre-2025: {pre_2025} readings\n2025+: {post_2025} readings',
                   transform=ax.transAxes, fontsize=8, ha='right', va='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_title('Complete Timeline Overview (All Historical Data)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Weight (kg)', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=7, ncol=5)

        # Format dates for full range
        if dates:
            date_range = (max(dates) - min(dates)).days
            if date_range > 365:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            else:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                ax.xaxis.set_major_locator(mdates.MonthLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)

    def _plot_main_timeline_with_kalman(self, ax, focus_start_date=None):
        """Plot main weight timeline with Kalman filter state estimation."""
        dates = []
        weights = []
        filtered_weights = []
        predicted_weights = []
        confidences = []
        sources = []
        is_valid = []

        for ts in self.time_series:
            date = self._parse_date(ts['date'])
            if focus_start_date and date < focus_start_date:
                continue
            dates.append(date)
            weights.append(ts['weight'])
            filtered_weights.append(ts.get('filtered_weight'))
            predicted_weights.append(ts.get('predicted_weight'))
            confidences.append(ts.get('confidence', 0.5))
            sources.append(ts.get('source', 'unknown'))
            is_valid.append(ts.get('is_valid', True))

        # Plot baseline period
        if self.baseline and dates:
            baseline_weight = self.baseline.get('weight', 0)
            if baseline_weight:
                ax.axhline(y=baseline_weight, color='#9C27B0', linestyle='--',
                          linewidth=1, alpha=0.5, label=f'Baseline: {baseline_weight:.1f}kg')

        # Plot measurements by source and validation status
        for i, (date, weight, valid, source, conf) in enumerate(zip(dates, weights, is_valid, sources, confidences)):
            style = self.SOURCE_STYLES.get(source, self.SOURCE_STYLES['unknown'])

            if valid:
                # Color by confidence
                if conf >= 0.9:
                    color = '#4CAF50'
                elif conf >= 0.75:
                    color = '#8BC34A'
                elif conf >= 0.5:
                    color = '#FFC107'
                else:
                    color = '#FF9800'
            else:
                color = '#FF4444'

            ax.scatter(date, weight, marker=style['marker'], c=color,
                      s=style['size'], alpha=0.8 if valid else 0.4,
                      edgecolors='black' if valid else 'red',
                      linewidth=1 if valid else 2, zorder=10)

        # Plot Kalman filtered trajectory
        valid_filtered = [(d, f) for d, f, v in zip(dates, filtered_weights, is_valid)
                         if f is not None and v]
        if valid_filtered:
            f_dates, f_weights = zip(*valid_filtered)
            ax.plot(f_dates, f_weights, 'g-', linewidth=2, alpha=0.9,
                   label='Kalman Filtered', zorder=20)

        # Plot predictions
        valid_predicted = [(d, p) for d, p in zip(dates, predicted_weights) if p is not None]
        if valid_predicted:
            p_dates, p_weights = zip(*valid_predicted)
            ax.plot(p_dates, p_weights, 'b--', linewidth=1, alpha=0.5,
                   label='Predictions', zorder=15)

        # Legend for sources
        for source in set(sources):
            style = self.SOURCE_STYLES.get(source, self.SOURCE_STYLES['unknown'])
            count = sources.count(source)
            ax.scatter([], [], marker=style['marker'], c=style['color'],
                      s=style['size'], label=f"{style['label']} (n={count})")

        title = 'Weight Measurements and Kalman Filter State Estimation'
        if focus_start_date and dates and min(dates) >= focus_start_date:
            title += ' [2025+ Focus]'
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Weight (kg)')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=8, ncol=2)

        # Format dates - adjust based on date range
        if dates:
            date_range = (max(dates) - min(dates)).days
            if date_range <= 30:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
                ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
            elif date_range <= 90:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
                ax.xaxis.set_major_locator(mdates.WeekdayLocator())
            elif date_range <= 365:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m/%d'))
                ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=1, interval=2))
            else:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                ax.xaxis.set_major_locator(mdates.MonthLocator())
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    def _plot_innovation_sequence(self, ax, focus_start_date=None):
        """Plot innovation (prediction error) sequence."""
        dates = []
        innovations = []
        is_valid = []

        for ts in self.time_series:
            if ts.get('predicted_weight') is not None:
                date = self._parse_date(ts['date'])
                if focus_start_date and date < focus_start_date:
                    continue
                dates.append(date)
                innovation = ts['weight'] - ts['predicted_weight']
                innovations.append(innovation)
                is_valid.append(ts.get('is_valid', True))

        if innovations:
            # Color by validation status
            colors = ['green' if v else 'red' for v in is_valid]
            ax.scatter(dates, innovations, c=colors, alpha=0.6, s=50)

            # Add zero line
            ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

            # Add confidence bounds (assuming 3-sigma from validation gate)
            std_innovation = np.std([i for i, v in zip(innovations, is_valid) if v]) if any(is_valid) else 1.0
            ax.axhline(y=3*std_innovation, color='red', linestyle='--', alpha=0.3, label='±3σ gate')
            ax.axhline(y=-3*std_innovation, color='red', linestyle='--', alpha=0.3)
            ax.fill_between(dates, [-3*std_innovation]*len(dates), [3*std_innovation]*len(dates),
                           alpha=0.1, color='red')

            title = 'Innovation Sequence (Prediction Errors)'
            if focus_start_date and dates and min(dates) >= focus_start_date:
                title += ' [2025+ Focus]'
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.set_xlabel('Date')
            ax.set_ylabel('Innovation (kg)')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=8)

            # Format dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    def _plot_innovation_histogram(self, ax):
        """Plot histogram of normalized innovations."""
        innovations = []

        for ts in self.time_series:
            if ts.get('predicted_weight') is not None and ts.get('is_valid', True):
                innovation = ts['weight'] - ts['predicted_weight']
                innovations.append(innovation)

        if len(innovations) > 5:
            # Normalize innovations
            std_innovation = np.std(innovations)
            if std_innovation > 0:
                normalized = [i/std_innovation for i in innovations]

                # Plot histogram
                n, bins, patches = ax.hist(normalized, bins=20, density=True,
                                          alpha=0.7, color='blue', edgecolor='black')

                # Overlay standard normal
                x = np.linspace(-4, 4, 100)
                ax.plot(x, stats.norm.pdf(x, 0, 1), 'r-', linewidth=2,
                       label='Standard Normal')

                # Add statistics
                mean_norm = np.mean(normalized)
                std_norm = np.std(normalized)
                ax.axvline(x=mean_norm, color='green', linestyle='--',
                          label=f'Mean: {mean_norm:.2f}')

                ax.set_title('Normalized Innovation Distribution', fontsize=11, fontweight='bold')
                ax.set_xlabel('Normalized Innovation (σ)')
                ax.set_ylabel('Density')
                ax.legend(loc='upper right', fontsize=8)
                ax.grid(True, alpha=0.3, axis='y')

                # Add text statistics
                ax.text(0.05, 0.95, f'Std: {std_norm:.2f}\nKurtosis: {stats.kurtosis(normalized):.2f}',
                       transform=ax.transAxes, fontsize=8, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=10)
            ax.set_title('Normalized Innovation Distribution', fontsize=11, fontweight='bold')

    def _plot_confidence_evolution(self, ax, focus_start_date=None):
        """Plot evolution of confidence scores over time."""
        dates = []
        confidences = []
        is_valid = []

        for ts in self.time_series:
            date = self._parse_date(ts['date'])
            if focus_start_date and date < focus_start_date:
                continue
            dates.append(date)
            confidences.append(ts.get('confidence', 0.5))
            is_valid.append(ts.get('is_valid', True))

        if dates:
            # Create color map for confidence
            colors = []
            for conf, valid in zip(confidences, is_valid):
                if not valid:
                    colors.append('red')
                elif conf >= 0.9:
                    colors.append('#4CAF50')
                elif conf >= 0.75:
                    colors.append('#8BC34A')
                elif conf >= 0.5:
                    colors.append('#FFC107')
                else:
                    colors.append('#FF9800')

            ax.scatter(dates, confidences, c=colors, alpha=0.7, s=50)

            # Add confidence thresholds
            ax.axhline(y=0.95, color='green', linestyle='--', alpha=0.3, label='High (0.95)')
            ax.axhline(y=0.8, color='orange', linestyle='--', alpha=0.3, label='Medium (0.8)')
            ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.3, label='Low (0.5)')

            # Add rolling mean
            if len(confidences) > 10:
                window = min(10, len(confidences)//3)
                rolling_mean = np.convolve(confidences, np.ones(window)/window, mode='valid')
                rolling_dates = dates[window-1:]  # Adjust dates to match rolling_mean length
                if len(rolling_dates) == len(rolling_mean):
                    ax.plot(rolling_dates, rolling_mean, 'b-', linewidth=2, alpha=0.7,
                           label=f'{window}-pt Moving Avg')

            title = 'Confidence Score Evolution'
            if focus_start_date and dates and min(dates) >= focus_start_date:
                title += ' [2025+ Focus]'
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.set_xlabel('Date')
            ax.set_ylabel('Confidence Score')
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='lower right', fontsize=8)

            # Format dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    def _plot_validation_gate_analysis(self, ax):
        """Analyze validation gate performance."""
        # Count accepted vs rejected by normalized innovation
        accepted_innovations = []
        rejected_innovations = []

        for ts in self.time_series:
            if ts.get('predicted_weight') is not None:
                innovation = abs(ts['weight'] - ts['predicted_weight'])
                if ts.get('is_valid', True):
                    accepted_innovations.append(innovation)
                else:
                    rejected_innovations.append(innovation)

        # Create box plot
        data_to_plot = []
        labels = []

        if accepted_innovations:
            data_to_plot.append(accepted_innovations)
            labels.append(f'Accepted\n(n={len(accepted_innovations)})')

        if rejected_innovations:
            data_to_plot.append(rejected_innovations)
            labels.append(f'Rejected\n(n={len(rejected_innovations)})')

        if data_to_plot:
            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)

            # Color the boxes
            colors = ['lightgreen', 'lightcoral']
            for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                patch.set_facecolor(color)

            ax.set_title('Validation Gate Analysis', fontsize=11, fontweight='bold')
            ax.set_ylabel('Absolute Innovation (kg)')
            ax.grid(True, alpha=0.3, axis='y')

            # Add statistics
            total = len(accepted_innovations) + len(rejected_innovations)
            if total > 0:
                acceptance_rate = len(accepted_innovations) / total
                ax.text(0.5, 0.95, f'Acceptance Rate: {acceptance_rate:.1%}',
                       transform=ax.transAxes, fontsize=9, ha='center',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax.text(0.5, 0.5, 'No validation data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=10)
            ax.set_title('Validation Gate Analysis', fontsize=11, fontweight='bold')

    def _plot_trend_analysis(self, ax, focus_start_date=None):
        """Plot trend component from Kalman filter."""
        dates = []
        trends = []

        for ts in self.time_series:
            if ts.get('trend_kg_per_day') is not None:
                date = self._parse_date(ts['date'])
                if focus_start_date and date < focus_start_date:
                    continue
                dates.append(date)
                trends.append(ts['trend_kg_per_day'] * 7)  # Convert to weekly

        if dates:
            # Plot trend
            ax.plot(dates, trends, 'b-', linewidth=2, alpha=0.7)
            ax.fill_between(dates, trends, 0, alpha=0.3)

            # Add zero line
            ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

            # Add meaningful thresholds
            ax.axhline(y=0.5, color='green', linestyle='--', alpha=0.3, label='+0.5 kg/week')
            ax.axhline(y=-0.5, color='red', linestyle='--', alpha=0.3, label='-0.5 kg/week')

            # Calculate and display statistics
            if trends:
                avg_trend = np.mean(trends)
                ax.axhline(y=avg_trend, color='purple', linestyle='-', alpha=0.5,
                          label=f'Avg: {avg_trend:.3f} kg/week')

            title = 'Weight Trend Analysis (Kalman State)'
            if focus_start_date and dates and min(dates) >= focus_start_date:
                title += ' [2025+ Focus]'
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.set_xlabel('Date')
            ax.set_ylabel('Trend (kg/week)')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=8)

            # Format dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    def _plot_state_uncertainty(self, ax, focus_start_date=None):
        """Plot evolution of state uncertainty (from covariance matrix)."""
        dates = []
        uncertainties = []

        for ts in self.time_series:
            # Extract uncertainty from metadata if available
            metadata = ts.get('processing_metadata', {})
            kalman_data = metadata.get('kalman', {})
            if 'prediction_variance' in kalman_data:
                date = self._parse_date(ts['date'])
                if focus_start_date and date < focus_start_date:
                    continue
                dates.append(date)
                uncertainties.append(np.sqrt(kalman_data['prediction_variance']))

        if dates and uncertainties:
            ax.plot(dates, uncertainties, 'r-', linewidth=2, alpha=0.7)
            ax.fill_between(dates, uncertainties, 0, alpha=0.3, color='red')

            ax.set_title('Kalman Filter State Uncertainty', fontsize=11, fontweight='bold')
            ax.set_xlabel('Date')
            ax.set_ylabel('Uncertainty (kg, 1σ)')
            ax.grid(True, alpha=0.3)

            # Add statistics
            if uncertainties:
                avg_uncertainty = np.mean(uncertainties)
                ax.axhline(y=avg_uncertainty, color='blue', linestyle='--', alpha=0.5,
                          label=f'Avg: {avg_uncertainty:.3f} kg')
                ax.legend(loc='upper right', fontsize=8)

            # Format dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        else:
            ax.text(0.5, 0.5, 'Uncertainty data not available', ha='center', va='center',
                   transform=ax.transAxes, fontsize=10)
            ax.set_title('Kalman Filter State Uncertainty', fontsize=11, fontweight='bold')

    def _plot_kalman_statistics(self, ax):
        """Display Kalman filter statistics."""
        ax.axis('off')

        stats_lines = []
        stats_lines.append("KALMAN FILTER STATISTICS")
        stats_lines.append("=" * 25)
        stats_lines.append("")

        # Baseline info
        if self.baseline:
            stats_lines.append(f"Baseline Weight: {self.baseline.get('weight', 0):.2f} kg")
            stats_lines.append(f"Baseline Variance: {self.baseline.get('variance', 0):.4f}")
            stats_lines.append(f"Baseline Confidence: {self.baseline.get('confidence', 'N/A')}")
            stats_lines.append("")

        # Current state
        if self.current_state:
            stats_lines.append(f"Current Weight: {self.current_state.get('weight', 0):.2f} kg")
            stats_lines.append(f"Current Trend: {self.current_state.get('trend_kg_per_week', 0):.3f} kg/week")
            stats_lines.append(f"Measurements: {self.current_state.get('measurement_count', 0)}")
            stats_lines.append("")

        # Processing stats
        if self.stats:
            total = self.stats.get('total_readings', 0)
            accepted = self.stats.get('accepted_readings', 0)
            rejected = self.stats.get('rejected_readings', 0)

            stats_lines.append(f"Total Readings: {total}")
            stats_lines.append(f"Accepted: {accepted}")
            stats_lines.append(f"Rejected: {rejected}")
            if total > 0:
                stats_lines.append(f"Acceptance Rate: {accepted/total*100:.1f}%")
            stats_lines.append("")

        # Weight range
        weights = [ts['weight'] for ts in self.time_series]
        if weights:
            stats_lines.append(f"Min Weight: {min(weights):.1f} kg")
            stats_lines.append(f"Max Weight: {max(weights):.1f} kg")
            stats_lines.append(f"Range: {max(weights) - min(weights):.1f} kg")

        text = '\n'.join(stats_lines)
        ax.text(0.05, 0.95, text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top', fontfamily='monospace')
        ax.set_title('Filter Statistics', fontsize=11, fontweight='bold', loc='left')

    def _plot_weight_distribution(self, ax):
        """Plot weight distribution with Kalman estimates."""
        raw_weights = []
        filtered_weights = []

        for ts in self.time_series:
            raw_weights.append(ts['weight'])
            if ts.get('filtered_weight') is not None and ts.get('is_valid', True):
                filtered_weights.append(ts['filtered_weight'])

        if len(raw_weights) > 5:
            # Plot histograms
            bins = np.linspace(min(raw_weights), max(raw_weights), 20)

            ax.hist(raw_weights, bins=bins, alpha=0.5, color='blue',
                   label=f'Raw (n={len(raw_weights)})', edgecolor='black')

            if filtered_weights:
                ax.hist(filtered_weights, bins=bins, alpha=0.5, color='green',
                       label=f'Filtered (n={len(filtered_weights)})', edgecolor='black')

            # Add baseline and current markers
            if self.baseline.get('weight'):
                ax.axvline(x=self.baseline['weight'], color='purple', linestyle='--',
                          linewidth=2, label='Baseline')

            if self.current_state.get('weight'):
                ax.axvline(x=self.current_state['weight'], color='red', linestyle='-',
                          linewidth=2, label='Current')

            ax.set_title('Weight Distribution', fontsize=11, fontweight='bold')
            ax.set_xlabel('Weight (kg)')
            ax.set_ylabel('Frequency')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3, axis='y')
        else:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=10)
            ax.set_title('Weight Distribution', fontsize=11, fontweight='bold')

    def _plot_acceptance_metrics(self, ax):
        """Plot acceptance metrics by source."""
        source_stats = {}

        for ts in self.time_series:
            source = ts.get('source', 'unknown')
            if source not in source_stats:
                source_stats[source] = {'total': 0, 'accepted': 0}
            source_stats[source]['total'] += 1
            if ts.get('is_valid', True):
                source_stats[source]['accepted'] += 1

        if source_stats:
            sources = list(source_stats.keys())
            acceptance_rates = [source_stats[s]['accepted']/source_stats[s]['total']*100
                              for s in sources]
            totals = [source_stats[s]['total'] for s in sources]

            # Create bar chart
            x = np.arange(len(sources))
            bars = ax.bar(x, acceptance_rates, color=['#4CAF50' if ar >= 80 else '#FFC107' if ar >= 60 else '#FF5722'
                                                      for ar in acceptance_rates])

            # Add value labels
            for i, (bar, total, rate) in enumerate(zip(bars, totals, acceptance_rates)):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{rate:.0f}%\n(n={total})', ha='center', fontsize=8)

            ax.set_title('Acceptance Rate by Source', fontsize=11, fontweight='bold')
            ax.set_xlabel('Source')
            ax.set_ylabel('Acceptance Rate (%)')
            ax.set_xticks(x)
            ax.set_xticklabels([self.SOURCE_STYLES.get(s, {'label': s})['label']
                               for s in sources], rotation=45, ha='right')
            ax.set_ylim(0, 110)
            ax.grid(True, alpha=0.3, axis='y')

            # Add overall rate
            total_accepted = sum(source_stats[s]['accepted'] for s in sources)
            total_readings = sum(source_stats[s]['total'] for s in sources)
            overall_rate = total_accepted / total_readings * 100 if total_readings > 0 else 0

            ax.axhline(y=overall_rate, color='red', linestyle='--', alpha=0.5,
                      label=f'Overall: {overall_rate:.1f}%')
            ax.legend(loc='lower right', fontsize=8)
        else:
            ax.text(0.5, 0.5, 'No source data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=10)
            ax.set_title('Acceptance Rate by Source', fontsize=11, fontweight='bold')

    def _parse_date(self, date_str: str) -> datetime:
        """Parse date string to datetime object."""
        if isinstance(date_str, datetime):
            return date_str
        if 'Z' in date_str:
            return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        if 'T' in date_str:
            return datetime.fromisoformat(date_str)
        return datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')


def create_enhanced_dashboards(
    results: Dict[str, Dict[str, Any]],
    output_dir: str = "output/visualizations",
    max_users: Optional[int] = None
) -> Dict[str, Path]:
    """
    Create enhanced Kalman filter visualizations for multiple users.

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

    logger.info(f"Creating Kalman filter visualizations for {len(users_to_process)} users")

    for i, user_id in enumerate(users_to_process, 1):
        try:
            dashboard = EnhancedDashboard(user_id, results[user_id], output_dir)
            path = dashboard.create_dashboard()
            if path:
                dashboard_paths[user_id] = path
                logger.info(f"Created dashboard {i}/{len(users_to_process)} for user {user_id}")
        except Exception as e:
            logger.error(f"Failed to create dashboard for user {user_id}: {e}")

    logger.info(f"Successfully created {len(dashboard_paths)} Kalman dashboards")
    return dashboard_paths
