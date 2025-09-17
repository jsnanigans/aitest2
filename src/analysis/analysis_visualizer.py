"""
Analysis Visualizer Module

Generates comprehensive visualizations for weight loss interval analysis.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import logging
from datetime import datetime


class AnalysisVisualizer:
    """Generates visualizations for analysis results"""

    def __init__(self, output_dir: Path):
        """
        Initialize analysis visualizer

        Args:
            output_dir: Directory for visualization outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectory for top user charts
        self.top_users_dir = self.output_dir / 'top_200_users'
        self.top_users_dir.mkdir(exist_ok=True)

        self.logger = logging.getLogger(__name__)

        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")

    def generate_all_visualizations(self,
                                   user_intervals: Dict,
                                   statistics: Dict,
                                   top_users: List[Dict]) -> Dict[str, Path]:
        """
        Generate all visualization files

        Args:
            user_intervals: User interval analysis results
            statistics: Calculated statistics
            top_users: Top divergent users

        Returns:
            Dictionary mapping visualization type to file path
        """
        viz_files = {}

        try:
            # Generate summary visualizations
            viz_files['weight_loss_distribution'] = self._generate_weight_loss_distribution(
                statistics['interval_statistics']
            )

            viz_files['outlier_impact_heatmap'] = self._generate_outlier_impact_heatmap(
                user_intervals, top_users[:50]  # Top 50 for heatmap
            )

            viz_files['quality_correlation'] = self._generate_quality_correlation_plot(
                statistics['user_statistics']
            )

            viz_files['population_summary'] = self._generate_population_summary(
                statistics['population_statistics']
            )

            # Generate top 200 user visualizations
            user_viz_count = 0
            for user_info in top_users[:200]:
                user_id = user_info['user_id']
                if user_id in user_intervals:
                    user_file = self._generate_user_comparison_chart(
                        user_id, user_intervals[user_id], statistics['user_statistics'].get(user_id, {})
                    )
                    if user_file:
                        viz_files[f'user_{user_id}'] = user_file
                        user_viz_count += 1

            self.logger.info(f"Generated {user_viz_count} user visualizations")

            # Generate top 200 summary dashboard
            viz_files['top_200_summary'] = self._generate_top_users_dashboard(
                top_users, statistics
            )

        except Exception as e:
            self.logger.error(f"Error generating visualizations: {str(e)}", exc_info=True)

        self.logger.info(f"Generated {len(viz_files)} visualization files")
        return viz_files

    def _generate_weight_loss_distribution(self, interval_stats: List[Dict]) -> Path:
        """Generate box plots comparing raw vs filtered weight loss at each interval"""
        output_path = self.output_dir / 'weight_loss_distribution.png'

        # Prepare data for plotting
        intervals = []
        raw_changes = []
        filtered_changes = []

        for stat in interval_stats:
            if stat['interval_days'] > 0:  # Skip baseline
                intervals.append(stat['interval_days'])

                # Collect change data
                if stat.get('raw') and 'change' in stat['raw']:
                    raw_changes.append({
                        'interval': stat['interval_days'],
                        'mean': stat['raw']['change']['mean'],
                        'median': stat['raw']['change']['median'],
                        'std': stat['raw']['change']['std'],
                        'q1': stat['raw'].get('q1', 0),
                        'q3': stat['raw'].get('q3', 0)
                    })

                if stat.get('filtered') and 'change' in stat['filtered']:
                    filtered_changes.append({
                        'interval': stat['interval_days'],
                        'mean': stat['filtered']['change']['mean'],
                        'median': stat['filtered']['change']['median'],
                        'std': stat['filtered']['change']['std'],
                        'q1': stat['filtered'].get('q1', 0),
                        'q3': stat['filtered'].get('q3', 0)
                    })

        if not intervals:
            self.logger.warning("No interval data for weight loss distribution")
            return None

        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # Plot raw data
        if raw_changes:
            raw_df = pd.DataFrame(raw_changes)
            axes[0].errorbar(raw_df['interval'], raw_df['mean'],
                           yerr=raw_df['std'], fmt='o-', capsize=5, label='Mean ± SD')
            axes[0].plot(raw_df['interval'], raw_df['median'], 's--', label='Median')
            axes[0].fill_between(raw_df['interval'], raw_df['q1'], raw_df['q3'],
                               alpha=0.3, label='IQR')
            axes[0].set_title('Raw Data Weight Changes')
            axes[0].set_xlabel('Days from Baseline')
            axes[0].set_ylabel('Weight Change (lbs)')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            axes[0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        # Plot filtered data
        if filtered_changes:
            filtered_df = pd.DataFrame(filtered_changes)
            axes[1].errorbar(filtered_df['interval'], filtered_df['mean'],
                           yerr=filtered_df['std'], fmt='o-', capsize=5, label='Mean ± SD')
            axes[1].plot(filtered_df['interval'], filtered_df['median'], 's--', label='Median')
            axes[1].fill_between(filtered_df['interval'], filtered_df['q1'], filtered_df['q3'],
                               alpha=0.3, label='IQR')
            axes[1].set_title('Filtered Data Weight Changes')
            axes[1].set_xlabel('Days from Baseline')
            axes[1].set_ylabel('Weight Change (lbs)')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        plt.suptitle('Weight Loss Distribution: Raw vs Filtered Data', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return output_path

    def _generate_outlier_impact_heatmap(self,
                                        user_intervals: Dict,
                                        top_users: List[Dict]) -> Path:
        """Generate heatmap showing outlier impact across users and intervals"""
        output_path = self.output_dir / 'outlier_impact_heatmap.png'

        # Prepare data matrix
        user_ids = [u['user_id'] for u in top_users if u['user_id'] in user_intervals]
        if not user_ids:
            return None

        # Get maximum number of intervals
        max_intervals = max(
            len(user_intervals[uid]['intervals'])
            for uid in user_ids
        )

        # Create difference matrix
        diff_matrix = np.full((len(user_ids), max_intervals), np.nan)

        for i, user_id in enumerate(user_ids):
            intervals = user_intervals[user_id]['intervals']
            for j, interval in enumerate(intervals):
                if interval.get('raw_weight') and interval.get('filtered_weight'):
                    diff = abs(interval['raw_weight'] - interval['filtered_weight'])
                    diff_matrix[i, j] = diff

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 10))

        # Create heatmap
        sns.heatmap(diff_matrix,
                   xticklabels=[f"Day {i*30}" for i in range(max_intervals)],
                   yticklabels=[f"User {i+1}" for i in range(len(user_ids))],
                   cmap='YlOrRd',
                   cbar_kws={'label': 'Difference (lbs)'},
                   ax=ax)

        ax.set_title('Outlier Impact Heatmap: Top 50 Users', fontsize=14, fontweight='bold')
        ax.set_xlabel('Days from Baseline')
        ax.set_ylabel('Users (Ranked by Divergence)')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return output_path

    def _generate_quality_correlation_plot(self, user_stats: Dict) -> Path:
        """Generate scatter plot of data quality vs weight loss success"""
        output_path = self.output_dir / 'quality_correlation.png'

        # Collect data
        quality_scores = []
        weight_losses = []
        outlier_counts = []

        for user_id, stats in user_stats.items():
            if stats.get('data_completeness', 0) > 50:  # Only users with >50% data
                # Calculate average quality (if available)
                quality = stats.get('data_completeness', 50) / 100

                # Get weight loss percentage
                weight_change = stats.get('weight_change', {}).get('filtered', {}).get('percentage')

                if weight_change is not None:
                    quality_scores.append(quality)
                    weight_losses.append(-weight_change)  # Negative for loss
                    outlier_impact = stats.get('outlier_impact', {})
                    outlier_counts.append(outlier_impact.get('avg_deviation', 0))

        if not quality_scores:
            return None

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # Create scatter plot
        scatter = ax.scatter(quality_scores, weight_losses, c=outlier_counts,
                           cmap='viridis', s=100, alpha=0.6, edgecolors='black', linewidth=0.5)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Avg Outlier Deviation (lbs)', rotation=270, labelpad=20)

        # Add trend line
        z = np.polyfit(quality_scores, weight_losses, 1)
        p = np.poly1d(z)
        ax.plot(quality_scores, p(quality_scores), "r--", alpha=0.8, label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')

        ax.set_xlabel('Data Completeness Score')
        ax.set_ylabel('Weight Loss (%)')
        ax.set_title('Data Quality vs Weight Loss Success', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.axhline(y=5, color='green', linestyle='--', alpha=0.5, label='5% Loss Target')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return output_path

    def _generate_population_summary(self, pop_stats: Dict) -> Path:
        """Generate population-level summary visualization"""
        output_path = self.output_dir / 'population_summary.png'

        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

        # 1. Success rates comparison
        ax1 = fig.add_subplot(gs[0, 0])
        success_raw = pop_stats.get('success_rate_5pct', {}).get('raw', 0)
        success_filtered = pop_stats.get('success_rate_5pct', {}).get('filtered', 0)
        ax1.bar(['Raw Data', 'Filtered Data'], [success_raw, success_filtered],
               color=['coral', 'lightblue'])
        ax1.set_ylabel('Success Rate (%)')
        ax1.set_title('5% Weight Loss Success Rate')
        ax1.set_ylim([0, max(success_raw, success_filtered) * 1.2])

        # Add value labels on bars
        for i, v in enumerate([success_raw, success_filtered]):
            ax1.text(i, v + 1, f'{v:.1f}%', ha='center')

        # 2. Average weight loss comparison
        ax2 = fig.add_subplot(gs[0, 1])
        avg_raw = pop_stats.get('average_weight_loss', {}).get('raw', 0)
        avg_filtered = pop_stats.get('average_weight_loss', {}).get('filtered', 0)
        ax2.bar(['Raw Data', 'Filtered Data'], [avg_raw, avg_filtered],
               color=['coral', 'lightblue'])
        ax2.set_ylabel('Weight Change (lbs)')
        ax2.set_title('Average Weight Change')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        # 3. Data coverage
        ax3 = fig.add_subplot(gs[0, 2])
        total = pop_stats.get('total_users', 0)
        with_data = pop_stats.get('users_with_data', 0)
        ax3.pie([with_data, total - with_data],
               labels=['With Data', 'Missing Data'],
               autopct='%1.1f%%',
               colors=['lightgreen', 'lightgray'])
        ax3.set_title(f'Data Coverage (n={total})')

        # 4. Weight loss distribution (raw)
        ax4 = fig.add_subplot(gs[1, 0:2])
        if pop_stats.get('weight_loss_distribution', {}).get('raw'):
            dist = pop_stats['weight_loss_distribution']['raw']
            x = ['Min', 'Q1', 'Median', 'Q3', 'Max']
            y = [dist.get('min', 0), dist.get('q1', 0), dist.get('median', 0),
                dist.get('q3', 0), dist.get('max', 0)]
            ax4.plot(x, y, 'o-', label='Raw', color='coral', linewidth=2)

        if pop_stats.get('weight_loss_distribution', {}).get('filtered'):
            dist = pop_stats['weight_loss_distribution']['filtered']
            y = [dist.get('min', 0), dist.get('q1', 0), dist.get('median', 0),
                dist.get('q3', 0), dist.get('max', 0)]
            ax4.plot(x, y, 's-', label='Filtered', color='lightblue', linewidth=2)

        ax4.set_ylabel('Weight Change (lbs)')
        ax4.set_title('Weight Loss Distribution Comparison')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        # 5. Summary text
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.axis('off')
        summary_text = f"""Population Summary:

Total Users: {total:,}
Users with Data: {with_data:,}

Average Weight Loss:
  Raw: {avg_raw:.2f} lbs
  Filtered: {avg_filtered:.2f} lbs

Success Rate (>5% loss):
  Raw: {success_raw:.1f}%
  Filtered: {success_filtered:.1f}%
"""
        ax5.text(0.1, 0.9, summary_text, transform=ax5.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace')

        plt.suptitle('Population Analysis Summary', fontsize=16, fontweight='bold')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return output_path

    def _generate_user_comparison_chart(self,
                                       user_id: str,
                                       user_data: Dict,
                                       user_stats: Dict) -> Optional[Path]:
        """Generate detailed comparison chart for a single user"""
        output_path = self.top_users_dir / f'user_{user_id}_comparison.png'

        intervals = user_data['intervals']
        if not intervals:
            return None

        # Prepare data
        dates = []
        raw_weights = []
        filtered_weights = []
        quality_scores = []

        for interval in intervals:
            dates.append(interval['interval_days'])
            raw_weights.append(interval.get('raw_weight'))
            filtered_weights.append(interval.get('filtered_weight'))

        # Filter out None values for plotting
        valid_raw = [(d, w) for d, w in zip(dates, raw_weights) if w is not None]
        valid_filtered = [(d, w) for d, w in zip(dates, filtered_weights) if w is not None]

        if not valid_raw and not valid_filtered:
            return None

        # Create figure with 2x2 subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Timeline comparison
        if valid_raw:
            raw_dates, raw_vals = zip(*valid_raw)
            ax1.plot(raw_dates, raw_vals, 'o-', label='Raw', alpha=0.7, linewidth=2)
        if valid_filtered:
            filt_dates, filt_vals = zip(*valid_filtered)
            ax1.plot(filt_dates, filt_vals, 's-', label='Filtered', alpha=0.7, linewidth=2)

        ax1.set_xlabel('Days from Baseline')
        ax1.set_ylabel('Weight (lbs)')
        ax1.set_title('Weight Timeline Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Difference plot
        differences = []
        diff_dates = []
        for interval in intervals:
            if interval.get('raw_weight') and interval.get('filtered_weight'):
                diff = interval['raw_weight'] - interval['filtered_weight']
                differences.append(diff)
                diff_dates.append(interval['interval_days'])

        if differences:
            colors = ['red' if d > 0 else 'blue' for d in differences]
            ax2.bar(diff_dates, differences, color=colors, alpha=0.6)
            ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax2.set_xlabel('Days from Baseline')
            ax2.set_ylabel('Difference (Raw - Filtered) lbs')
            ax2.set_title('Measurement Differences')
            ax2.grid(True, alpha=0.3)

        # 3. Source distribution
        sources = [interval.get('raw_source') for interval in intervals if interval.get('raw_source')]
        if sources:
            source_counts = pd.Series(sources).value_counts()
            ax3.pie(source_counts.values, labels=source_counts.index, autopct='%1.0f%%')
            ax3.set_title('Data Source Distribution')

        # 4. Weight change summary
        interval_labels = []
        raw_changes = []
        filtered_changes = []

        baseline_raw = intervals[0].get('raw_weight')
        baseline_filtered = intervals[0].get('filtered_weight')

        for i, interval in enumerate(intervals[1:], 1):  # Skip baseline
            if interval['interval_days'] % 30 == 0:  # Only show 30-day intervals
                interval_labels.append(f"Day {interval['interval_days']}")

                if baseline_raw and interval.get('raw_weight'):
                    raw_changes.append(interval['raw_weight'] - baseline_raw)
                else:
                    raw_changes.append(None)

                if baseline_filtered and interval.get('filtered_weight'):
                    filtered_changes.append(interval['filtered_weight'] - baseline_filtered)
                else:
                    filtered_changes.append(None)

        if interval_labels:
            x = np.arange(len(interval_labels))
            width = 0.35

            # Filter out None values
            raw_x = [i for i, v in enumerate(raw_changes) if v is not None]
            raw_y = [v for v in raw_changes if v is not None]
            filtered_x = [i for i, v in enumerate(filtered_changes) if v is not None]
            filtered_y = [v for v in filtered_changes if v is not None]

            if raw_y:
                ax4.bar([i - width/2 for i in raw_x], raw_y, width, label='Raw', alpha=0.7, color='coral')
            if filtered_y:
                ax4.bar([i + width/2 for i in filtered_x], filtered_y, width, label='Filtered', alpha=0.7, color='lightblue')

            ax4.set_xlabel('Interval')
            ax4.set_ylabel('Weight Change (lbs)')
            ax4.set_title('Weight Change at Intervals')
            ax4.set_xticks(x)
            ax4.set_xticklabels(interval_labels, rotation=45)
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        # Add overall title with user info
        weight_change = user_stats.get('weight_change', {})
        raw_pct = weight_change.get('raw', {}).get('percentage', 'N/A')
        filtered_pct = weight_change.get('filtered', {}).get('percentage', 'N/A')

        fig.suptitle(f'User {user_id} - Raw Loss: {raw_pct}% | Filtered Loss: {filtered_pct}%',
                    fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')  # Lower DPI for individual files
        plt.close()

        return output_path

    def _generate_top_users_dashboard(self,
                                     top_users: List[Dict],
                                     statistics: Dict) -> Path:
        """Generate summary dashboard for top 200 users"""
        output_path = self.output_dir / 'top_200_summary.png'

        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

        # 1. Distribution of divergence scores
        ax1 = fig.add_subplot(gs[0, 0])
        divergence_scores = [u['divergence_score'] for u in top_users]
        ax1.hist(divergence_scores, bins=30, edgecolor='black', alpha=0.7)
        ax1.set_xlabel('Divergence Score (lbs)')
        ax1.set_ylabel('Number of Users')
        ax1.set_title('Distribution of Divergence Scores')
        ax1.axvline(np.mean(divergence_scores), color='red', linestyle='--', label=f'Mean: {np.mean(divergence_scores):.2f}')
        ax1.legend()

        # 2. Average impact per interval
        ax2 = fig.add_subplot(gs[0, 1])
        interval_impacts = {}
        for stat in statistics['interval_statistics']:
            if stat.get('difference'):
                interval_impacts[stat['interval_days']] = abs(stat['difference'].get('mean_diff', 0))

        if interval_impacts:
            ax2.bar(interval_impacts.keys(), interval_impacts.values(), color='skyblue')
            ax2.set_xlabel('Days from Baseline')
            ax2.set_ylabel('Avg Difference (lbs)')
            ax2.set_title('Average Impact by Interval')

        # 3. Source reliability breakdown
        ax3 = fig.add_subplot(gs[0, 2])
        source_impacts = statistics.get('divergence_statistics', {})
        if source_impacts:
            impact_dist = source_impacts.get('impact_distribution', {})
            if impact_dist:
                bp = ax3.boxplot([divergence_scores], labels=['Top 200'])
                ax3.set_ylabel('Divergence Score (lbs)')
                ax3.set_title('Divergence Score Distribution')

        # 4. Success rate comparison
        ax4 = fig.add_subplot(gs[1, 0])
        success_categories = {'<2%': 0, '2-5%': 0, '5-10%': 0, '>10%': 0}
        for user_info in top_users:
            user_id = user_info['user_id']
            if user_id in statistics['user_statistics']:
                pct = statistics['user_statistics'][user_id].get('weight_change', {}).get('filtered', {}).get('percentage')
                if pct:
                    pct = -pct  # Convert to positive for loss
                    if pct < 2:
                        success_categories['<2%'] += 1
                    elif pct < 5:
                        success_categories['2-5%'] += 1
                    elif pct < 10:
                        success_categories['5-10%'] += 1
                    else:
                        success_categories['>10%'] += 1

        ax4.bar(success_categories.keys(), success_categories.values(), color='lightgreen')
        ax4.set_xlabel('Weight Loss Category')
        ax4.set_ylabel('Number of Users')
        ax4.set_title('Weight Loss Success Distribution')

        # 5. Outlier frequency histogram
        ax5 = fig.add_subplot(gs[1, 1])
        outlier_rates = []
        for user_info in top_users:
            user_id = user_info['user_id']
            if user_id in statistics['user_statistics']:
                impact = statistics['user_statistics'][user_id].get('outlier_impact', {})
                if impact.get('has_impact'):
                    outlier_rates.append(impact.get('avg_deviation', 0))

        if outlier_rates:
            ax5.hist(outlier_rates, bins=20, edgecolor='black', alpha=0.7, color='orange')
            ax5.set_xlabel('Average Deviation (lbs)')
            ax5.set_ylabel('Number of Users')
            ax5.set_title('Outlier Impact Distribution')

        # 6. Data completeness
        ax6 = fig.add_subplot(gs[1, 2])
        completeness = []
        for user_info in top_users:
            user_id = user_info['user_id']
            if user_id in statistics['user_statistics']:
                comp = statistics['user_statistics'][user_id].get('data_completeness', 0)
                completeness.append(comp)

        if completeness:
            ax6.hist(completeness, bins=20, edgecolor='black', alpha=0.7, color='purple')
            ax6.set_xlabel('Data Completeness (%)')
            ax6.set_ylabel('Number of Users')
            ax6.set_title('Data Completeness Distribution')

        # 7. Consistency scores
        ax7 = fig.add_subplot(gs[2, 0:2])
        consistency_scores = []
        user_labels = []
        for i, user_info in enumerate(top_users[:20]):  # Top 20 for visibility
            user_id = user_info['user_id']
            if user_id in statistics['user_statistics']:
                trajectory = statistics['user_statistics'][user_id].get('trajectory', {}).get('filtered', {})
                if trajectory and trajectory.get('consistency'):
                    consistency_scores.append(trajectory['consistency'])
                    user_labels.append(f"U{i+1}")

        if consistency_scores:
            ax7.barh(user_labels, consistency_scores, color='teal')
            ax7.set_xlabel('Consistency Score')
            ax7.set_ylabel('User (Top 20)')
            ax7.set_title('Trajectory Consistency - Top 20 Users')

        # 8. Summary statistics text
        ax8 = fig.add_subplot(gs[2, 2])
        ax8.axis('off')

        avg_divergence = np.mean(divergence_scores) if divergence_scores else 0
        max_divergence = max(divergence_scores) if divergence_scores else 0
        total_impact = sum(divergence_scores) if divergence_scores else 0

        summary_text = f"""Top 200 Users Summary:

Average Divergence: {avg_divergence:.2f} lbs
Maximum Divergence: {max_divergence:.2f} lbs
Total Impact: {total_impact:.1f} lbs

Users with >5% loss: {sum(1 for cat, count in success_categories.items() if cat in ['5-10%', '>10%'])}
Users with >10% loss: {success_categories.get('>10%', 0)}

Avg Data Completeness: {np.mean(completeness):.1f}%
Avg Outlier Impact: {np.mean(outlier_rates):.2f} lbs
"""
        ax8.text(0.1, 0.9, summary_text, transform=ax8.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace')

        plt.suptitle('Top 200 Users Analysis Dashboard', fontsize=16, fontweight='bold')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return output_path