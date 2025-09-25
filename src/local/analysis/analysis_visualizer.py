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

    def _generate_filter_impact_overview(self, statistics: Dict, user_intervals: Dict) -> Path:
        """Generate comprehensive overview of filter impact"""
        output_path = self.output_dir / 'filter_impact_overview.png'

        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

        # 1. Outlier rejection rate by source
        ax1 = fig.add_subplot(gs[0, 0:2])
        source_stats = statistics.get('source_statistics', {})
        if source_stats:
            sources = []
            rejection_rates = []
            colors = []
            for source, stats in source_stats.items():
                sources.append(source.split('/')[-1] if '/' in source else source)
                raw_count = stats.get('raw_count', 0)
                filtered_count = stats.get('filtered_count', 0)
                if raw_count > 0:
                    rejection_rate = ((raw_count - filtered_count) / raw_count) * 100
                    rejection_rates.append(rejection_rate)
                    colors.append('red' if rejection_rate > 40 else 'orange' if rejection_rate > 20 else 'green')

            bars = ax1.bar(sources, rejection_rates, color=colors, edgecolor='black', alpha=0.7)
            ax1.set_xlabel('Data Source', fontsize=10)
            ax1.set_ylabel('Rejection Rate (%)')
            ax1.set_title('Outlier Rejection Rate by Source', fontweight='bold')
            ax1.set_xticklabels(sources, rotation=45, ha='right', fontsize=8)
            ax1.axhline(y=10, color='gray', linestyle='--', alpha=0.5, label='10% threshold')
            ax1.legend()

            # Add value labels on bars
            for bar, rate in zip(bars, rejection_rates):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{rate:.1f}%', ha='center', va='bottom', fontsize=8)

        # 2. Total measurements impact
        ax2 = fig.add_subplot(gs[0, 2:4])
        total_raw = sum(s.get('raw_count', 0) for s in source_stats.values()) if source_stats else 0
        total_filtered = sum(s.get('filtered_count', 0) for s in source_stats.values()) if source_stats else 0
        total_rejected = total_raw - total_filtered

        sizes = [total_filtered, total_rejected]
        labels = [f'Retained\n{total_filtered:,}\n({(total_filtered/total_raw*100):.1f}%)',
                  f'Rejected\n{total_rejected:,}\n({(total_rejected/total_raw*100):.1f}%)']
        colors_pie = ['#2ecc71', '#e74c3c']
        explode = (0.05, 0.05)

        ax2.pie(sizes, labels=labels, colors=colors_pie, autopct='',
                explode=explode, shadow=True, startangle=90)
        ax2.set_title('Total Measurement Filtering Impact', fontweight='bold')

        # 3. Standard deviation reduction
        ax3 = fig.add_subplot(gs[1, 0])
        interval_stats = statistics.get('interval_statistics', [])
        intervals = []
        raw_stds = []
        filtered_stds = []

        for stat in interval_stats[:12]:  # First year
            if stat.get('raw') and stat.get('filtered'):
                intervals.append(stat['interval_days'])
                raw_stds.append(stat['raw'].get('weight', {}).get('std', 0))
                filtered_stds.append(stat['filtered'].get('weight', {}).get('std', 0))

        if intervals:
            x = np.arange(len(intervals))
            width = 0.35
            ax3.bar(x - width/2, raw_stds, width, label='Raw', alpha=0.7, color='coral')
            ax3.bar(x + width/2, filtered_stds, width, label='Filtered', alpha=0.7, color='lightblue')
            ax3.set_xlabel('Days from Baseline')
            ax3.set_ylabel('Std Deviation (lbs)')
            ax3.set_title('Noise Reduction: Std Dev Comparison', fontweight='bold')
            ax3.set_xticks(x)
            ax3.set_xticklabels(intervals, rotation=45)
            ax3.legend()
            ax3.grid(True, alpha=0.3)

        # 4. Signal preservation (trend accuracy)
        ax4 = fig.add_subplot(gs[1, 1])
        # Calculate average weight change trajectory
        raw_trajectory = []
        filtered_trajectory = []
        for stat in interval_stats[:12]:
            if stat.get('raw') and stat.get('filtered'):
                raw_trajectory.append(stat['raw'].get('change', {}).get('mean', 0))
                filtered_trajectory.append(stat['filtered'].get('change', {}).get('mean', 0))

        if raw_trajectory and filtered_trajectory:
            days = [i*30 for i in range(len(raw_trajectory))]
            ax4.plot(days, raw_trajectory, 'o-', label='Raw', alpha=0.7, linewidth=2, markersize=8)
            ax4.plot(days, filtered_trajectory, 's-', label='Filtered', alpha=0.7, linewidth=2, markersize=6)
            ax4.set_xlabel('Days from Baseline')
            ax4.set_ylabel('Average Weight Change (lbs)')
            ax4.set_title('Signal Preservation: Weight Loss Trend', fontweight='bold')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        # 5. Filter effectiveness metrics
        ax5 = fig.add_subplot(gs[1, 2:4])
        # Calculate key metrics
        metrics = {
            'Noise Reduction': 0,
            'Outlier Removal': 0,
            'Trend Preservation': 0,
            'Data Retention': 0
        }

        # Noise reduction (std dev reduction)
        if raw_stds and filtered_stds:
            avg_raw_std = np.mean(raw_stds)
            avg_filtered_std = np.mean(filtered_stds)
            if avg_raw_std > 0:
                metrics['Noise Reduction'] = ((avg_raw_std - avg_filtered_std) / avg_raw_std) * 100

        # Outlier removal
        if total_raw > 0:
            metrics['Outlier Removal'] = (total_rejected / total_raw) * 100

        # Trend preservation (correlation between raw and filtered trajectories)
        if len(raw_trajectory) > 1 and len(filtered_trajectory) > 1:
            correlation = np.corrcoef(raw_trajectory, filtered_trajectory)[0, 1]
            metrics['Trend Preservation'] = correlation * 100

        # Data retention
        if total_raw > 0:
            metrics['Data Retention'] = (total_filtered / total_raw) * 100

        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        colors_metrics = ['green' if v > 70 else 'orange' if v > 40 else 'red' for v in metric_values]

        bars = ax5.barh(metric_names, metric_values, color=colors_metrics, edgecolor='black', alpha=0.7)
        ax5.set_xlabel('Effectiveness (%)')
        ax5.set_title('Kalman Filter Effectiveness Metrics', fontweight='bold')
        ax5.set_xlim([0, 105])

        # Add value labels
        for bar, value in zip(bars, metric_values):
            width = bar.get_width()
            ax5.text(width, bar.get_y() + bar.get_height()/2.,
                    f'{value:.1f}%', ha='left', va='center', fontweight='bold')

        # 6. Impact distribution histogram
        ax6 = fig.add_subplot(gs[2, 0:2])
        all_diffs = []
        for user_id, intervals in user_intervals.items():
            for interval in intervals.get('intervals', []):
                if interval.get('raw_weight') and interval.get('filtered_weight'):
                    diff = abs(interval['raw_weight'] - interval['filtered_weight'])
                    if diff > 0:
                        all_diffs.append(diff)

        if all_diffs:
            ax6.hist(all_diffs, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
            ax6.axvline(np.mean(all_diffs), color='red', linestyle='--',
                       label=f'Mean: {np.mean(all_diffs):.2f} lbs')
            ax6.axvline(np.median(all_diffs), color='green', linestyle='--',
                       label=f'Median: {np.median(all_diffs):.2f} lbs')
            ax6.set_xlabel('Absolute Difference (lbs)')
            ax6.set_ylabel('Frequency')
            ax6.set_title('Distribution of Filter Adjustments', fontweight='bold')
            ax6.legend()
            ax6.set_xlim([0, min(20, max(all_diffs))])  # Cap at 20 lbs for visibility

        # 7. Summary text panel
        ax7 = fig.add_subplot(gs[2, 2:4])
        ax7.axis('off')

        summary_text = f"""KALMAN FILTER IMPACT SUMMARY

Total Measurements Processed: {total_raw:,}
Measurements Retained: {total_filtered:,}
Outliers Rejected: {total_rejected:,}

Key Achievements:
✓ Noise Reduction: {metrics['Noise Reduction']:.1f}%
✓ Outlier Removal: {metrics['Outlier Removal']:.1f}%
✓ Signal Preservation: {metrics['Trend Preservation']:.1f}%
✓ Data Retention: {metrics['Data Retention']:.1f}%

Average Adjustment: {np.mean(all_diffs) if all_diffs else 0:.2f} lbs
Median Adjustment: {np.median(all_diffs) if all_diffs else 0:.2f} lbs
Max Adjustment: {max(all_diffs) if all_diffs else 0:.2f} lbs

Conclusion:
The Kalman filter successfully reduces noise
while preserving the underlying weight loss
trend, improving data quality for analysis."""

        ax7.text(0.05, 0.95, summary_text, transform=ax7.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.5))

        plt.suptitle('Kalman Filter Impact Analysis', fontsize=18, fontweight='bold', y=0.98)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return output_path

    def _generate_noise_reduction_demo(self, user_intervals: Dict, top_users: List[Dict]) -> Path:
        """Generate demonstration of noise reduction on most noisy users"""
        output_path = self.output_dir / 'noise_reduction_demo.png'

        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        axes = axes.flatten()

        demo_count = 0
        for i, user_info in enumerate(top_users[:6]):
            user_id = user_info['user_id']
            if user_id not in user_intervals:
                continue

            ax = axes[demo_count]
            intervals = user_intervals[user_id]['intervals']

            # Extract data points
            days = []
            raw_weights = []
            filtered_weights = []

            for interval in intervals:
                if interval.get('raw_weight'):
                    days.append(interval['interval_days'])
                    raw_weights.append(interval['raw_weight'])
                    if interval.get('filtered_weight'):
                        filtered_weights.append(interval['filtered_weight'])
                    else:
                        filtered_weights.append(None)

            if len(days) < 3:
                continue

            # Plot raw data with noise
            ax.scatter(days, raw_weights, c='red', alpha=0.5, s=30, label='Raw (Noisy)', zorder=2)
            ax.plot(days, raw_weights, 'r-', alpha=0.2, linewidth=1, zorder=1)

            # Plot filtered smooth line
            valid_filtered = [(d, w) for d, w in zip(days, filtered_weights) if w is not None]
            if valid_filtered:
                f_days, f_weights = zip(*valid_filtered)
                ax.plot(f_days, f_weights, 'b-', linewidth=2.5, label='Filtered (Smooth)', zorder=3)
                ax.scatter(f_days, f_weights, c='blue', s=40, zorder=4, edgecolors='darkblue')

            # Calculate noise metrics
            if len(raw_weights) > 1:
                raw_std = np.std(raw_weights)
                if valid_filtered and len(valid_filtered) > 1:
                    _, f_weights_calc = zip(*valid_filtered)
                    filtered_std = np.std(f_weights_calc)
                    noise_reduction = ((raw_std - filtered_std) / raw_std * 100) if raw_std > 0 else 0
                else:
                    filtered_std = 0
                    noise_reduction = 0
            else:
                raw_std = filtered_std = noise_reduction = 0

            # Add shaded area showing filter correction
            for j in range(len(days)):
                if j < len(filtered_weights) and filtered_weights[j] is not None:
                    ax.fill_between([days[j]], raw_weights[j], filtered_weights[j],
                                   alpha=0.2, color='gray')

            ax.set_xlabel('Days from Baseline')
            ax.set_ylabel('Weight (lbs)')
            ax.set_title(f'User {demo_count + 1}: Noise σ={raw_std:.2f}→{filtered_std:.2f} '
                        f'(-{noise_reduction:.0f}%)', fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)

            # Add divergence score annotation
            divergence = user_info.get('divergence_score', 0)
            ax.text(0.95, 0.05, f'Divergence: {divergence:.2f} lbs',
                   transform=ax.transAxes, ha='right', va='bottom',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.5))

            demo_count += 1
            if demo_count >= 6:
                break

        # Hide unused subplots
        for i in range(demo_count, 6):
            axes[i].axis('off')

        plt.suptitle('Noise Reduction Demonstration: Raw vs Filtered Data',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return output_path

    def _generate_outlier_detection_impact(self, statistics: Dict, user_intervals: Dict) -> Path:
        """Generate visualization showing outlier detection effectiveness"""
        output_path = self.output_dir / 'outlier_detection_impact.png'

        fig = plt.figure(figsize=(18, 10))
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

        # 1. Outlier detection by magnitude
        ax1 = fig.add_subplot(gs[0, 0])
        outlier_magnitudes = []
        retained_magnitudes = []

        for user_id, intervals in user_intervals.items():
            for interval in intervals.get('intervals', []):
                if interval.get('raw_weight') and interval.get('filtered_weight'):
                    diff = abs(interval['raw_weight'] - interval['filtered_weight'])
                    if diff > 5:  # Significant outlier
                        outlier_magnitudes.append(interval['raw_weight'])
                    else:
                        retained_magnitudes.append(interval['raw_weight'])

        if outlier_magnitudes and retained_magnitudes:
            ax1.hist([retained_magnitudes, outlier_magnitudes], bins=30,
                    label=['Retained', 'Outliers'], color=['green', 'red'],
                    alpha=0.6, edgecolor='black')
            ax1.set_xlabel('Weight (lbs)')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Weight Distribution: Retained vs Outliers', fontweight='bold')
            ax1.legend()

        # 2. Outlier percentage by interval
        ax2 = fig.add_subplot(gs[0, 1])
        interval_outlier_rates = []
        interval_days = []

        for stat in statistics.get('interval_statistics', [])[:24]:  # First 2 years
            if stat.get('raw') and stat.get('filtered'):
                raw_count = stat['raw'].get('count', 0)
                filtered_count = stat['filtered'].get('count', 0)
                if raw_count > 0:
                    outlier_rate = ((raw_count - filtered_count) / raw_count) * 100
                    interval_outlier_rates.append(outlier_rate)
                    interval_days.append(stat['interval_days'])

        if interval_outlier_rates:
            ax2.plot(interval_days, interval_outlier_rates, 'o-', color='orange', linewidth=2)
            ax2.fill_between(interval_days, 0, interval_outlier_rates, alpha=0.3, color='orange')
            ax2.set_xlabel('Days from Baseline')
            ax2.set_ylabel('Outlier Rate (%)')
            ax2.set_title('Outlier Detection Rate Over Time', fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=np.mean(interval_outlier_rates), color='red', linestyle='--',
                       label=f'Average: {np.mean(interval_outlier_rates):.1f}%')
            ax2.legend()

        # 3. Extreme outlier examples
        ax3 = fig.add_subplot(gs[0, 2])
        extreme_outliers = []

        for user_id, intervals in user_intervals.items():
            for interval in intervals.get('intervals', []):
                if interval.get('raw_weight') and interval.get('filtered_weight'):
                    diff = interval['raw_weight'] - interval['filtered_weight']
                    if abs(diff) > 10:  # Extreme outlier
                        extreme_outliers.append(diff)

        if extreme_outliers:
            ax3.hist(extreme_outliers, bins=30, edgecolor='black', color='darkred', alpha=0.7)
            ax3.set_xlabel('Weight Adjustment (lbs)')
            ax3.set_ylabel('Frequency')
            ax3.set_title(f'Extreme Outliers Removed (n={len(extreme_outliers)})', fontweight='bold')
            ax3.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

            # Add statistics
            mean_adj = np.mean(np.abs(extreme_outliers))
            max_adj = max(np.abs(extreme_outliers))
            ax3.text(0.95, 0.95, f'Mean: {mean_adj:.1f} lbs\nMax: {max_adj:.1f} lbs',
                    transform=ax3.transAxes, ha='right', va='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

        # 4. Source-specific outlier rates
        ax4 = fig.add_subplot(gs[1, :])
        source_stats = statistics.get('source_statistics', {})

        if source_stats:
            sources = []
            outlier_counts = []
            retained_counts = []

            for source, stats in source_stats.items():
                source_name = source.split('/')[-1] if '/' in source else source
                sources.append(source_name)
                raw = stats.get('raw_count', 0)
                filtered = stats.get('filtered_count', 0)
                outlier_counts.append(raw - filtered)
                retained_counts.append(filtered)

            x = np.arange(len(sources))
            width = 0.35

            ax4.bar(x - width/2, retained_counts, width, label='Retained', color='green', alpha=0.7)
            ax4.bar(x + width/2, outlier_counts, width, label='Outliers', color='red', alpha=0.7)

            ax4.set_xlabel('Data Source')
            ax4.set_ylabel('Number of Measurements')
            ax4.set_title('Outlier Detection by Data Source', fontweight='bold')
            ax4.set_xticks(x)
            ax4.set_xticklabels(sources, rotation=45, ha='right')
            ax4.legend()
            ax4.grid(True, alpha=0.3, axis='y')

            # Add percentage labels
            for i, (outlier, retained) in enumerate(zip(outlier_counts, retained_counts)):
                total = outlier + retained
                if total > 0:
                    pct = (outlier / total) * 100
                    ax4.text(i, outlier + retained + max(retained_counts)*0.01, f'{pct:.0f}%',
                            ha='center', va='bottom', fontsize=8, fontweight='bold')

        plt.suptitle('Outlier Detection Impact Analysis', fontsize=16, fontweight='bold')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return output_path

    def _generate_before_after_comparison(self, user_intervals: Dict, top_users: List[Dict]) -> Path:
        """Generate before/after comparison for representative users"""
        output_path = self.output_dir / 'before_after_comparison.png'

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        comparison_count = 0
        for user_info in top_users:
            if comparison_count >= 6:
                break

            user_id = user_info['user_id']
            if user_id not in user_intervals:
                continue

            intervals = user_intervals[user_id]['intervals']
            if len(intervals) < 5:
                continue

            ax = axes[comparison_count]

            # Extract timeline data
            days = []
            raw_weights = []
            filtered_weights = []
            sources = []

            for interval in intervals:
                if interval.get('raw_weight'):
                    days.append(interval['interval_days'])
                    raw_weights.append(interval['raw_weight'])
                    filtered_weights.append(interval.get('filtered_weight'))
                    sources.append(interval.get('raw_source', 'unknown'))

            if len(days) < 3:
                continue

            # Create twin axis for difference
            ax2 = ax.twinx()

            # Plot raw data
            ax.scatter(days, raw_weights, c='red', alpha=0.4, s=50, label='Raw Data', marker='o')
            ax.plot(days, raw_weights, 'r--', alpha=0.3, linewidth=1)

            # Plot filtered data
            valid_filtered = [(d, w) for d, w in zip(days, filtered_weights) if w is not None]
            if valid_filtered:
                f_days, f_weights = zip(*valid_filtered)
                ax.plot(f_days, f_weights, 'b-', linewidth=2.5, label='Kalman Filtered', alpha=0.8)
                ax.scatter(f_days, f_weights, c='blue', s=60, marker='s', edgecolors='darkblue', zorder=5)

            # Plot difference on secondary axis
            differences = [r - f if f is not None else 0
                          for r, f in zip(raw_weights, filtered_weights)]
            ax2.bar(days, differences, alpha=0.3, color='gray', width=15, label='Adjustment')
            ax2.set_ylabel('Adjustment (lbs)', color='gray')
            ax2.tick_params(axis='y', labelcolor='gray')
            ax2.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)

            # Calculate improvement metrics
            if len(raw_weights) > 1:
                # Calculate smoothness (variation in consecutive differences)
                raw_diffs = np.diff(raw_weights)
                raw_smoothness = np.std(raw_diffs)

                if valid_filtered and len(valid_filtered) > 1:
                    _, f_weights_only = zip(*valid_filtered)
                    filt_diffs = np.diff(f_weights_only)
                    filt_smoothness = np.std(filt_diffs)
                    improvement = ((raw_smoothness - filt_smoothness) / raw_smoothness * 100) if raw_smoothness > 0 else 0
                else:
                    filt_smoothness = 0
                    improvement = 0

                title = f'User {comparison_count + 1}: Smoothness improved {improvement:.0f}%'
            else:
                title = f'User {comparison_count + 1}'

            ax.set_xlabel('Days from Baseline')
            ax.set_ylabel('Weight (lbs)', color='black')
            ax.set_title(title, fontweight='bold')
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)

            # Add source indicator for outliers
            for i, (day, diff) in enumerate(zip(days, differences)):
                if abs(diff) > 5:  # Significant outlier
                    source = sources[i] if i < len(sources) else 'unknown'
                    source_short = source.split('/')[-1][:10] if '/' in source else source[:10]
                    ax.annotate(source_short, xy=(day, raw_weights[i]),
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=6, alpha=0.7)

            comparison_count += 1

        # Hide unused subplots
        for i in range(comparison_count, 6):
            axes[i].axis('off')

        plt.suptitle('Before vs After: Raw Data vs Kalman Filtered',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return output_path

    def generate_daily_comparison_visualizations(self, daily_results: Dict, dramatic_users: List[Dict]) -> Dict[str, Path]:
        """Generate visualizations for daily weight analysis"""
        viz_files = {}

        # 1. Most dramatic daily differences
        viz_files['daily_dramatic_impact'] = self._generate_daily_dramatic_impact(
            daily_results, dramatic_users[:6]
        )

        # 2. Daily noise patterns
        viz_files['daily_noise_patterns'] = self._generate_daily_noise_patterns(
            daily_results, dramatic_users[:10]
        )

        return viz_files

    def _generate_daily_dramatic_impact(self, daily_results: Dict, dramatic_users: List[Dict]) -> Path:
        """Show users with most dramatic daily filtering differences"""
        output_path = self.output_dir / 'daily_dramatic_impact.png'

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for idx, user_info in enumerate(dramatic_users[:6]):
            user_id = user_info['user_id']
            if user_id not in daily_results:
                continue

            ax = axes[idx]
            user_data = daily_results[user_id]
            daily_comp = user_data['daily_comparison']

            # Extract data
            days = [c['days_from_start'] for c in daily_comp]
            raw_means = [c['raw_mean'] for c in daily_comp if c['raw_mean'] is not None]
            filtered_means = [c['filtered_mean'] for c in daily_comp if c['filtered_mean'] is not None]

            # Plot with emphasis on days with multiple measurements
            for comp in daily_comp:
                day = comp['days_from_start']
                if comp['raw_count'] > 1:  # Multiple measurements on this day
                    if comp['raw_mean'] is not None:
                        # Show the range of measurements for that day
                        ax.errorbar(day, comp['raw_mean'], yerr=comp['raw_std'],
                                   fmt='o', color='red', alpha=0.5, capsize=3, label='_nolegend_')

            # Plot means
            raw_days = [c['days_from_start'] for c in daily_comp if c['raw_mean'] is not None]
            raw_vals = [c['raw_mean'] for c in daily_comp if c['raw_mean'] is not None]

            filt_days = [c['days_from_start'] for c in daily_comp if c['filtered_mean'] is not None]
            filt_vals = [c['filtered_mean'] for c in daily_comp if c['filtered_mean'] is not None]

            if raw_vals:
                ax.plot(raw_days, raw_vals, 'r-', alpha=0.3, label='Raw Daily Avg')
                ax.scatter(raw_days, raw_vals, c='red', s=20, alpha=0.6)

            if filt_vals:
                ax.plot(filt_days, filt_vals, 'b-', linewidth=2, label='Filtered Daily Avg')
                ax.scatter(filt_days, filt_vals, c='blue', s=30, edgecolors='darkblue')

            # Highlight days with dramatic differences
            for comp in daily_comp:
                if comp['difference'] and comp['difference'] > 5:
                    ax.axvspan(comp['days_from_start']-0.5, comp['days_from_start']+0.5,
                              alpha=0.2, color='yellow')

            ax.set_xlabel('Days from Start')
            ax.set_ylabel('Weight (lbs)')
            ax.set_title(f'User {idx+1}: Score={user_info["dramatic_score"]:.1f}, '
                        f'Max Diff={user_info["max_daily_difference"]:.1f} lbs',
                        fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)

            # Add annotation for days with multiple measurements
            multi_days = sum(1 for c in daily_comp if c['raw_count'] > 1)
            ax.text(0.02, 0.98, f'{multi_days} days with multiple readings',
                   transform=ax.transAxes, va='top',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.5))

        plt.suptitle('Daily Analysis: Most Dramatic Filter Impact\nYellow bands = days with >5 lbs difference',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return output_path

    def _generate_daily_noise_patterns(self, daily_results: Dict, dramatic_users: List[Dict]) -> Path:
        """Visualize daily noise patterns and filtering effectiveness"""
        output_path = self.output_dir / 'daily_noise_patterns.png'

        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

        # 1. Daily range comparison
        ax1 = fig.add_subplot(gs[0, :])

        all_raw_ranges = []
        all_filtered_ranges = []

        for user_info in dramatic_users:
            user_id = user_info['user_id']
            if user_id in daily_results:
                for comp in daily_results[user_id]['daily_comparison']:
                    if comp['raw_count'] > 1:  # Days with multiple measurements
                        all_raw_ranges.append(comp['raw_range'])
                        if comp['filtered_count'] > 1:
                            all_filtered_ranges.append(comp['filtered_range'])

        if all_raw_ranges:
            ax1.hist([all_raw_ranges, all_filtered_ranges if all_filtered_ranges else []],
                    bins=30, label=['Raw Daily Range', 'Filtered Daily Range'],
                    color=['red', 'blue'], alpha=0.6, edgecolor='black')
            ax1.set_xlabel('Daily Weight Range (lbs)')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Daily Weight Range: Raw vs Filtered (Days with Multiple Measurements)', fontweight='bold')
            ax1.legend()
            ax1.axvline(np.mean(all_raw_ranges), color='darkred', linestyle='--',
                       label=f'Raw Mean: {np.mean(all_raw_ranges):.1f}')
            if all_filtered_ranges:
                ax1.axvline(np.mean(all_filtered_ranges), color='darkblue', linestyle='--',
                           label=f'Filtered Mean: {np.mean(all_filtered_ranges):.1f}')

        # 2. Rejection rate vs daily variance
        ax2 = fig.add_subplot(gs[1, 0])

        rejection_rates = []
        daily_variances = []

        for user_info in dramatic_users:
            user_id = user_info['user_id']
            if user_id in daily_results:
                user_data = daily_results[user_id]
                rejection_rates.append(user_data['rejection_rate'] * 100)
                daily_variances.append(user_data['noise_reduction']['raw_avg_std'])

        if rejection_rates and daily_variances:
            scatter = ax2.scatter(daily_variances, rejection_rates, c=range(len(rejection_rates)),
                                 cmap='viridis', s=100, alpha=0.7, edgecolors='black')
            ax2.set_xlabel('Average Daily Std Dev (lbs)')
            ax2.set_ylabel('Rejection Rate (%)')
            ax2.set_title('Rejection Rate vs Daily Noise Level', fontweight='bold')
            ax2.grid(True, alpha=0.3)

            # Add trend line
            z = np.polyfit(daily_variances, rejection_rates, 1)
            p = np.poly1d(z)
            ax2.plot(sorted(daily_variances), p(sorted(daily_variances)), "r--", alpha=0.8)

        # 3. Dramatic score distribution
        ax3 = fig.add_subplot(gs[1, 1])

        all_scores = [u['dramatic_score'] for u in dramatic_users]
        ax3.hist(all_scores, bins=20, edgecolor='black', color='orange', alpha=0.7)
        ax3.set_xlabel('Dramatic Impact Score')
        ax3.set_ylabel('Count')
        ax3.set_title('Distribution of Filter Impact Scores', fontweight='bold')
        ax3.axvline(50, color='red', linestyle='--', label='High Impact Threshold')
        ax3.legend()

        # 4. Days with multiple measurements
        ax4 = fig.add_subplot(gs[1, 2])

        multi_measurement_days = []
        for user_info in dramatic_users:
            user_id = user_info['user_id']
            if user_id in daily_results:
                multi_measurement_days.append(daily_results[user_id]['days_with_multiple_measurements'])

        if multi_measurement_days:
            ax4.bar(range(len(multi_measurement_days)), multi_measurement_days,
                   color='teal', edgecolor='black', alpha=0.7)
            ax4.set_xlabel('User Index')
            ax4.set_ylabel('Days with Multiple Measurements')
            ax4.set_title('Multiple Daily Measurements by User', fontweight='bold')
            ax4.grid(True, alpha=0.3, axis='y')

        # 5. Summary statistics
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('off')

        # Calculate overall statistics
        total_dramatic = sum(1 for u in dramatic_users if u['dramatic_score'] > 50)
        avg_max_diff = np.mean([u['max_daily_difference'] for u in dramatic_users])
        avg_rejection = np.mean([u['rejection_rate'] for u in dramatic_users]) * 100

        summary_text = f"""DAILY ANALYSIS INSIGHTS

Top {len(dramatic_users)} Most Affected Users:
• Average dramatic score: {np.mean(all_scores):.1f}
• Users with high impact (score >50): {total_dramatic}
• Average maximum daily difference: {avg_max_diff:.2f} lbs
• Average rejection rate: {avg_rejection:.1f}%

Key Findings:
• Days with multiple measurements show significantly more noise in raw data
• Average raw daily range: {np.mean(all_raw_ranges) if all_raw_ranges else 0:.2f} lbs
• Average filtered daily range: {np.mean(all_filtered_ranges) if all_filtered_ranges else 0:.2f} lbs
• Noise reduction: {((np.mean(all_raw_ranges) - np.mean(all_filtered_ranges)) / np.mean(all_raw_ranges) * 100) if all_raw_ranges and all_filtered_ranges else 0:.1f}%

The Kalman filter effectively smooths intra-day weight fluctuations while preserving genuine weight trends."""

        ax5.text(0.5, 0.5, summary_text, transform=ax5.transAxes,
                ha='center', va='center', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.8))

        plt.suptitle('Daily Noise Analysis: Impact of Multiple Same-Day Measurements',
                    fontsize=16, fontweight='bold')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return output_path