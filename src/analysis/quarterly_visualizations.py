"""
Visualization module for quarterly reporting analysis.
Creates comprehensive visualizations comparing raw vs filtered weight loss data.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from .quarterly_reporting import CohortAnalysis, QuarterlyMetrics

logger = logging.getLogger(__name__)

# Set style for professional-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class QuarterlyVisualizationGenerator:
    """
    Generates visualizations for quarterly weight loss reporting.
    """

    def __init__(self, output_dir: str = "reports/quarterly"):
        """
        Initialize the visualization generator.

        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Color scheme for consistency
        self.colors = {
            'raw': '#FF6B6B',
            'filtered': '#4ECDC4',
            'improvement': '#95E77E',
            'decline': '#FFE66D'
        }

    def create_weight_loss_distribution_comparison(
        self,
        results_df: pd.DataFrame,
        raw_metrics: QuarterlyMetrics,
        filtered_metrics: QuarterlyMetrics
    ) -> str:
        """
        Create box plots comparing weight loss distributions for raw vs filtered data.

        Args:
            results_df: DataFrame with detailed weight loss results
            raw_metrics: Quarterly metrics for raw data
            filtered_metrics: Quarterly metrics for filtered data

        Returns:
            Path to saved visualization
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Box plot of percentage weight loss
        ax1 = axes[0, 0]
        data_to_plot = [
            results_df['raw_loss_pct'].dropna(),
            results_df['filtered_loss_pct'].dropna()
        ]

        bp = ax1.boxplot(data_to_plot, labels=['Raw', 'Filtered'], patch_artist=True,
                        widths=0.6, showfliers=True)
        for patch, color in zip(bp['boxes'], [self.colors['raw'], self.colors['filtered']]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax1.set_ylabel('Weight Loss (%)', fontsize=12)
        ax1.set_title('Weight Loss Distribution (90+ Day Users)', fontsize=13, fontweight='bold')
        ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax1.axhline(y=-5, color='green', linestyle='--', alpha=0.5, label='5% loss target')
        ax1.axhline(y=-10, color='blue', linestyle='--', alpha=0.5, label='10% loss target')

        # Add mean values as diamonds
        ax1.scatter([1], [-raw_metrics.mean_weight_loss_pct],
                   marker='D', s=150, color='darkred', zorder=5, label='Raw Mean')
        ax1.scatter([2], [-filtered_metrics.mean_weight_loss_pct],
                   marker='D', s=150, color='darkgreen', zorder=5, label='Filtered Mean')

        # Invert y-axis so weight loss appears positive
        ax1.invert_yaxis()
        ax1.legend(loc='upper right', fontsize=10)

        # 2. Violin plot for better distribution visualization
        ax2 = axes[0, 1]
        plot_data = pd.DataFrame({
            'Weight Loss (%)': pd.concat([
                results_df['raw_loss_pct'].dropna(),
                results_df['filtered_loss_pct'].dropna()
            ]),
            'Data Type': ['Raw'] * len(results_df['raw_loss_pct'].dropna()) +
                        ['Filtered'] * len(results_df['filtered_loss_pct'].dropna())
        })

        sns.violinplot(data=plot_data, x='Data Type', y='Weight Loss (%)',
                      hue='Data Type',
                      palette={'Raw': self.colors['raw'], 'Filtered': self.colors['filtered']},
                      legend=False,
                      ax=ax2)
        ax2.set_title('Weight Loss Distribution Density')
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

        # 3. Histogram comparison with better binning
        ax3 = axes[1, 0]

        # Create bins that align for both datasets
        all_data = pd.concat([results_df['raw_loss_pct'].dropna(),
                              results_df['filtered_loss_pct'].dropna()])
        bins = np.linspace(all_data.min(), all_data.max(), 25)

        ax3.hist(results_df['raw_loss_pct'].dropna(), bins=bins, alpha=0.5,
                label=f'Raw (n={len(results_df["raw_loss_pct"].dropna())})',
                color=self.colors['raw'], edgecolor='black', density=True)
        ax3.hist(results_df['filtered_loss_pct'].dropna(), bins=bins, alpha=0.5,
                label=f'Filtered (n={len(results_df["filtered_loss_pct"].dropna())})',
                color=self.colors['filtered'], edgecolor='black', density=True)
        ax3.set_xlabel('Weight Loss (%)', fontsize=11)
        ax3.set_ylabel('Probability Density', fontsize=11)
        ax3.set_title('Weight Loss Distribution (Normalized)', fontsize=12)
        ax3.legend()
        ax3.axvline(x=5, color='green', linestyle='--', alpha=0.5, label='5% target')
        ax3.axvline(x=10, color='blue', linestyle='--', alpha=0.5, label='10% target')
        ax3.grid(True, alpha=0.3, axis='y')

        # 4. Success rate comparison with improvement indicators
        ax4 = axes[1, 1]
        success_rates = pd.DataFrame({
            'Raw': [
                (results_df['raw_loss_pct'] >= 5).sum() / len(results_df['raw_loss_pct'].dropna()) * 100,
                (results_df['raw_loss_pct'] >= 10).sum() / len(results_df['raw_loss_pct'].dropna()) * 100,
                (results_df['raw_loss_pct'] >= 15).sum() / len(results_df['raw_loss_pct'].dropna()) * 100
            ],
            'Filtered': [
                (results_df['filtered_loss_pct'] >= 5).sum() / len(results_df['filtered_loss_pct'].dropna()) * 100,
                (results_df['filtered_loss_pct'] >= 10).sum() / len(results_df['filtered_loss_pct'].dropna()) * 100,
                (results_df['filtered_loss_pct'] >= 15).sum() / len(results_df['filtered_loss_pct'].dropna()) * 100
            ]
        }, index=['5% Loss', '10% Loss', '15% Loss'])

        x = np.arange(len(success_rates.index))
        width = 0.35

        bars1 = ax4.bar(x - width/2, success_rates['Raw'], width,
                       label='Raw', color=self.colors['raw'], alpha=0.7)
        bars2 = ax4.bar(x + width/2, success_rates['Filtered'], width,
                       label='Filtered', color=self.colors['filtered'], alpha=0.7)

        ax4.set_ylabel('Success Rate (%)', fontsize=11)
        ax4.set_title('Clinical Success Rates', fontsize=12)
        ax4.set_xlabel('Weight Loss Threshold', fontsize=11)
        ax4.set_xticks(x)
        ax4.set_xticklabels(success_rates.index)
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')

        # Add value labels and improvement arrows
        for bar1, bar2, raw_val, filt_val in zip(bars1, bars2,
                                                 success_rates['Raw'],
                                                 success_rates['Filtered']):
            # Labels
            ax4.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 1,
                    f'{raw_val:.1f}%', ha='center', va='bottom', fontsize=9)
            ax4.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 1,
                    f'{filt_val:.1f}%', ha='center', va='bottom', fontsize=9)

            # Improvement indicator
            if filt_val > raw_val:
                improvement = filt_val - raw_val
                ax4.annotate('', xy=(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 5),
                           xytext=(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 5),
                           arrowprops=dict(arrowstyle='->', color='green', lw=1.5))
                ax4.text((bar1.get_x() + bar2.get_x() + bar2.get_width())/2,
                        max(bar1.get_height(), bar2.get_height()) + 7,
                        f'+{improvement:.1f}%', ha='center', fontsize=8, color='green')

        plt.tight_layout()

        # Save figure
        file_path = self.output_dir / "quarterly_weight_loss_distribution.png"
        plt.savefig(file_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved weight loss distribution comparison to {file_path}")
        return str(file_path)

    def create_cohort_progression_analysis(
        self,
        cohort_results: List[CohortAnalysis]
    ) -> str:
        """
        Create visualizations showing weight loss progression at different time checkpoints.

        Args:
            cohort_results: List of cohort analysis results for different time points

        Returns:
            Path to saved visualization
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Use larger font for better readability
        plt.rcParams.update({'font.size': 10})

        # Extract data for plotting
        days = [c.day_checkpoint for c in cohort_results]
        raw_means = [c.raw_mean_loss_pct for c in cohort_results]
        filtered_means = [c.filtered_mean_loss_pct for c in cohort_results]
        raw_5pct = [c.raw_5pct_success_rate for c in cohort_results]
        filtered_5pct = [c.filtered_5pct_success_rate for c in cohort_results]
        raw_10pct = [c.raw_10pct_success_rate for c in cohort_results]
        filtered_10pct = [c.filtered_10pct_success_rate for c in cohort_results]

        # 1. Mean weight loss progression with improvement highlighting
        ax1 = axes[0, 0]
        ax1.plot(days, raw_means, marker='o', label='Raw',
                color=self.colors['raw'], linewidth=2.5, markersize=10)
        ax1.plot(days, filtered_means, marker='s', label='Filtered',
                color=self.colors['filtered'], linewidth=2.5, markersize=10)
        ax1.set_xlabel('Days in Program', fontsize=11)
        ax1.set_ylabel('Mean Weight Loss (%)', fontsize=11)
        ax1.set_title('Average Weight Loss Progression', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Add difference shading with better visibility
        for i in range(len(days) - 1):
            if filtered_means[i] > raw_means[i]:
                ax1.fill_between([days[i], days[i+1]],
                               [raw_means[i], raw_means[i+1]],
                               [filtered_means[i], filtered_means[i+1]],
                               color=self.colors['improvement'], alpha=0.25)

        # Add value annotations at key points
        for i, day in enumerate(days):
            if day in [90, 180]:  # Annotate key checkpoints
                ax1.annotate(f'{filtered_means[i]:.1f}%',
                           xy=(day, filtered_means[i]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, color=self.colors['filtered'])
                ax1.annotate(f'{raw_means[i]:.1f}%',
                           xy=(day, raw_means[i]),
                           xytext=(5, -10), textcoords='offset points',
                           fontsize=8, color=self.colors['raw'])

        ax1.legend(loc='best')

        # 2. Success rate progression (5% threshold)
        ax2 = axes[0, 1]
        ax2.plot(days, raw_5pct, marker='o', label='Raw',
                color=self.colors['raw'], linewidth=2, markersize=8)
        ax2.plot(days, filtered_5pct, marker='s', label='Filtered',
                color=self.colors['filtered'], linewidth=2, markersize=8)
        ax2.set_xlabel('Days in Program')
        ax2.set_ylabel('Success Rate (%)')
        ax2.set_title('5% Weight Loss Success Rate Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Success rate progression (10% threshold)
        ax3 = axes[0, 2]
        ax3.plot(days, raw_10pct, marker='o', label='Raw',
                color=self.colors['raw'], linewidth=2, markersize=8)
        ax3.plot(days, filtered_10pct, marker='s', label='Filtered',
                color=self.colors['filtered'], linewidth=2, markersize=8)
        ax3.set_xlabel('Days in Program')
        ax3.set_ylabel('Success Rate (%)')
        ax3.set_title('10% Weight Loss Success Rate Over Time')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Difference in mean weight loss with value labels
        ax4 = axes[1, 0]
        differences = [c.mean_loss_difference for c in cohort_results]
        bars = ax4.bar(days, differences, width=15, color=[
            self.colors['improvement'] if d > 0 else self.colors['decline']
            for d in differences
        ], alpha=0.7, edgecolor='black', linewidth=1)
        ax4.set_xlabel('Days in Program', fontsize=11)
        ax4.set_ylabel('Improvement in Mean Loss (%-points)', fontsize=11)
        ax4.set_title('Data Quality Impact on Weight Loss', fontsize=12)
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax4.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, diff in zip(bars, differences):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{diff:+.2f}%', ha='center',
                    va='bottom' if height > 0 else 'top',
                    fontsize=9, fontweight='bold')

        # 5. Data availability comparison
        ax5 = axes[1, 1]
        raw_users = [c.users_with_data for c in cohort_results]
        total_users = [c.total_users_at_checkpoint for c in cohort_results]

        ax5.bar([d - 2 for d in days], raw_users, width=4,
               label='Raw', color=self.colors['raw'], alpha=0.7)
        ax5.bar([d + 2 for d in days], raw_users, width=4,
               label='Filtered', color=self.colors['filtered'], alpha=0.7)
        ax5.set_xlabel('Days in Program')
        ax5.set_ylabel('Number of Users with Data')
        ax5.set_title('Data Availability at Each Checkpoint')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # 6. Standard deviation comparison with reduction percentages
        ax6 = axes[1, 2]
        raw_stds = [c.raw_std_loss_pct for c in cohort_results]
        filtered_stds = [c.filtered_std_loss_pct for c in cohort_results]

        x = np.arange(len(days))
        width = 0.35

        bars1 = ax6.bar(x - width/2, raw_stds, width, label='Raw',
                       color=self.colors['raw'], alpha=0.7)
        bars2 = ax6.bar(x + width/2, filtered_stds, width, label='Filtered',
                       color=self.colors['filtered'], alpha=0.7)

        ax6.set_xlabel('Days in Program', fontsize=11)
        ax6.set_ylabel('Standard Deviation (%)', fontsize=11)
        ax6.set_title('Measurement Variability Reduction', fontsize=12)
        ax6.set_xticks(x)
        ax6.set_xticklabels(days)
        ax6.legend()
        ax6.grid(True, alpha=0.3)

        # Add reduction percentages
        for i, (r_std, f_std) in enumerate(zip(raw_stds, filtered_stds)):
            if r_std > 0:
                reduction = ((r_std - f_std) / r_std) * 100
                ax6.text(i, max(r_std, f_std) + 0.5,
                        f'-{reduction:.0f}%', ha='center',
                        fontsize=8, color='green' if reduction > 0 else 'red')

        plt.tight_layout()

        # Save figure
        file_path = self.output_dir / "quarterly_cohort_progression.png"
        plt.savefig(file_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved cohort progression analysis to {file_path}")
        return str(file_path)

    def create_detailed_metrics_comparison(
        self,
        raw_metrics: QuarterlyMetrics,
        filtered_metrics: QuarterlyMetrics
    ) -> str:
        """
        Create a detailed comparison of quarterly metrics.

        Args:
            raw_metrics: Metrics for raw data
            filtered_metrics: Metrics for filtered data

        Returns:
            Path to saved visualization
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Key metrics comparison
        ax1 = axes[0, 0]
        metrics_names = ['Mean Loss', 'Median Loss', 'Std Dev']
        raw_values = [
            raw_metrics.mean_weight_loss_pct,
            raw_metrics.median_weight_loss_pct,
            raw_metrics.std_weight_loss_pct
        ]
        filtered_values = [
            filtered_metrics.mean_weight_loss_pct,
            filtered_metrics.median_weight_loss_pct,
            filtered_metrics.std_weight_loss_pct
        ]

        x = np.arange(len(metrics_names))
        width = 0.35

        bars1 = ax1.bar(x - width/2, raw_values, width, label='Raw',
                       color=self.colors['raw'])
        bars2 = ax1.bar(x + width/2, filtered_values, width, label='Filtered',
                       color=self.colors['filtered'])

        ax1.set_ylabel('Weight Loss (%)')
        ax1.set_title('Key Statistical Metrics Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics_names)
        ax1.legend()

        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.annotate(f'{height:.2f}%',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom')

        # 2. Quartile comparison
        ax2 = axes[0, 1]
        quartiles = ['Min', 'Q1', 'Median', 'Q3', 'Max']
        raw_quartiles = [
            raw_metrics.min_weight_loss,
            raw_metrics.q1_weight_loss,
            raw_metrics.median_weight_loss_pct,
            raw_metrics.q3_weight_loss,
            raw_metrics.max_weight_loss
        ]
        filtered_quartiles = [
            filtered_metrics.min_weight_loss,
            filtered_metrics.q1_weight_loss,
            filtered_metrics.median_weight_loss_pct,
            filtered_metrics.q3_weight_loss,
            filtered_metrics.max_weight_loss
        ]

        x2 = np.arange(len(quartiles))
        ax2.plot(x2, raw_quartiles, 'o-', label='Raw',
                color=self.colors['raw'], linewidth=2, markersize=8)
        ax2.plot(x2, filtered_quartiles, 's-', label='Filtered',
                color=self.colors['filtered'], linewidth=2, markersize=8)

        ax2.set_xticks(x2)
        ax2.set_xticklabels(quartiles)
        ax2.set_ylabel('Weight Loss (%)')
        ax2.set_title('Distribution Quartiles Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Success counts
        ax3 = axes[1, 0]
        thresholds = ['5% Loss', '10% Loss', '15% Loss']
        raw_counts = [
            raw_metrics.users_losing_5pct,
            raw_metrics.users_losing_10pct,
            raw_metrics.users_losing_15pct
        ]
        filtered_counts = [
            filtered_metrics.users_losing_5pct,
            filtered_metrics.users_losing_10pct,
            filtered_metrics.users_losing_15pct
        ]

        x3 = np.arange(len(thresholds))
        bars3 = ax3.bar(x3 - width/2, raw_counts, width, label='Raw',
                       color=self.colors['raw'])
        bars4 = ax3.bar(x3 + width/2, filtered_counts, width, label='Filtered',
                       color=self.colors['filtered'])

        ax3.set_ylabel('Number of Users')
        ax3.set_title('Success Counts by Threshold')
        ax3.set_xticks(x3)
        ax3.set_xticklabels(thresholds)
        ax3.legend()

        # Add value labels
        for bars in [bars3, bars4]:
            for bar in bars:
                height = bar.get_height()
                ax3.annotate(f'{int(height)}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom')

        # 4. Data quality metrics
        ax4 = axes[1, 1]
        quality_metrics = ['Valid Start', 'Valid Endpoint', 'Complete Data']
        raw_quality = [
            raw_metrics.users_with_valid_start,
            raw_metrics.users_with_valid_endpoint,
            min(raw_metrics.users_with_valid_start, raw_metrics.users_with_valid_endpoint)
        ]
        filtered_quality = [
            filtered_metrics.users_with_valid_start,
            filtered_metrics.users_with_valid_endpoint,
            min(filtered_metrics.users_with_valid_start, filtered_metrics.users_with_valid_endpoint)
        ]

        # Calculate percentages
        total_eligible = raw_metrics.eligible_users
        raw_pct = [x / total_eligible * 100 for x in raw_quality]
        filtered_pct = [x / total_eligible * 100 for x in filtered_quality]

        x4 = np.arange(len(quality_metrics))
        bars5 = ax4.bar(x4 - width/2, raw_pct, width, label='Raw',
                       color=self.colors['raw'])
        bars6 = ax4.bar(x4 + width/2, filtered_pct, width, label='Filtered',
                       color=self.colors['filtered'])

        ax4.set_ylabel('Percentage of Eligible Users (%)')
        ax4.set_title('Data Completeness Comparison')
        ax4.set_xticks(x4)
        ax4.set_xticklabels(quality_metrics, rotation=15)
        ax4.legend()

        # Add value labels
        for bars in [bars5, bars6]:
            for bar in bars:
                height = bar.get_height()
                ax4.annotate(f'{height:.1f}%',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)

        plt.suptitle('Quarterly Reporting Metrics: Raw vs Filtered Data', fontsize=14, y=1.02)
        plt.tight_layout()

        # Save figure
        file_path = self.output_dir / "quarterly_detailed_metrics.png"
        plt.savefig(file_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved detailed metrics comparison to {file_path}")
        return str(file_path)

    def create_impact_summary_dashboard(
        self,
        raw_metrics: QuarterlyMetrics,
        filtered_metrics: QuarterlyMetrics,
        cohort_results: List[CohortAnalysis]
    ) -> str:
        """
        Create a summary dashboard showing the impact of filtering on quarterly reporting.

        Args:
            raw_metrics: Overall metrics for raw data
            filtered_metrics: Overall metrics for filtered data
            cohort_results: Cohort analysis results

        Returns:
            Path to saved visualization
        """
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)

        # Main title
        fig.suptitle('Quarterly Reporting Impact Dashboard: 90+ Day Users',
                    fontsize=16, y=0.98, fontweight='bold')

        # 1. Average weight loss improvement
        ax1 = fig.add_subplot(gs[0, :])

        # Create text summary
        improvement = filtered_metrics.mean_weight_loss_pct - raw_metrics.mean_weight_loss_pct
        improvement_text = "IMPROVED" if improvement > 0 else "DECREASED"
        color = self.colors['improvement'] if improvement > 0 else self.colors['decline']

        ax1.text(0.5, 0.7, f"Average Weight Loss {improvement_text} by",
                ha='center', va='center', fontsize=20, weight='bold')
        ax1.text(0.5, 0.3, f"{abs(improvement):.2f}%",
                ha='center', va='center', fontsize=36, color=color, weight='bold')
        ax1.text(0.5, 0.05, f"Raw: {raw_metrics.mean_weight_loss_pct:.2f}% → "
                          f"Filtered: {filtered_metrics.mean_weight_loss_pct:.2f}%",
                ha='center', va='center', fontsize=12)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.axis('off')

        # 2. Data availability
        ax2 = fig.add_subplot(gs[1, 0])
        availability = [raw_metrics.users_with_valid_endpoint,
                       filtered_metrics.users_with_valid_endpoint]
        bars = ax2.bar(['Raw', 'Filtered'], availability,
                      color=[self.colors['raw'], self.colors['filtered']])
        ax2.set_ylabel('Users with Valid Data')
        ax2.set_title('Data Availability (90+ Day Users)')

        for bar in bars:
            height = bar.get_height()
            ax2.annotate(f'{int(height)}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom')

        # 3. Success rate changes
        ax3 = fig.add_subplot(gs[1, 1])
        success_improvements = {
            '5% Loss': (filtered_metrics.users_losing_5pct / filtered_metrics.users_with_valid_endpoint * 100) -
                      (raw_metrics.users_losing_5pct / raw_metrics.users_with_valid_endpoint * 100) if raw_metrics.users_with_valid_endpoint > 0 else 0,
            '10% Loss': (filtered_metrics.users_losing_10pct / filtered_metrics.users_with_valid_endpoint * 100) -
                       (raw_metrics.users_losing_10pct / raw_metrics.users_with_valid_endpoint * 100) if raw_metrics.users_with_valid_endpoint > 0 else 0,
            '15% Loss': (filtered_metrics.users_losing_15pct / filtered_metrics.users_with_valid_endpoint * 100) -
                       (raw_metrics.users_losing_15pct / raw_metrics.users_with_valid_endpoint * 100) if raw_metrics.users_with_valid_endpoint > 0 else 0
        }

        bars = ax3.bar(success_improvements.keys(), success_improvements.values(),
                      color=[self.colors['improvement'] if v > 0 else self.colors['decline']
                             for v in success_improvements.values()])
        ax3.set_ylabel('Change in Success Rate (%)')
        ax3.set_title('Success Rate Impact')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)

        for bar in bars:
            height = bar.get_height()
            ax3.annotate(f'{height:+.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3 if height > 0 else -15),
                       textcoords="offset points",
                       ha='center', va='bottom' if height > 0 else 'top')

        # 4. Variability reduction
        ax4 = fig.add_subplot(gs[1, 2])
        std_reduction = ((raw_metrics.std_weight_loss_pct - filtered_metrics.std_weight_loss_pct) /
                        raw_metrics.std_weight_loss_pct * 100) if raw_metrics.std_weight_loss_pct > 0 else 0

        ax4.text(0.5, 0.7, "Variability Reduced by", ha='center', va='center', fontsize=14)
        ax4.text(0.5, 0.3, f"{std_reduction:.1f}%", ha='center', va='center',
                fontsize=24, color=self.colors['improvement'] if std_reduction > 0 else self.colors['decline'],
                weight='bold')
        ax4.text(0.5, 0.1, f"Std Dev: {raw_metrics.std_weight_loss_pct:.2f}% → "
                          f"{filtered_metrics.std_weight_loss_pct:.2f}%",
                ha='center', va='center', fontsize=10)
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')

        # 5. Timeline impact
        ax5 = fig.add_subplot(gs[2, :])
        days = [c.day_checkpoint for c in cohort_results[-5:]]  # Last 5 checkpoints
        improvements = [c.mean_loss_difference for c in cohort_results[-5:]]

        bars = ax5.bar(days, improvements,
                      color=[self.colors['improvement'] if i > 0 else self.colors['decline']
                             for i in improvements])
        ax5.set_xlabel('Days in Program')
        ax5.set_ylabel('Improvement in Mean Loss (%)')
        ax5.set_title('Filtering Impact Over Time')
        ax5.axhline(y=0, color='black', linestyle='-', alpha=0.5)

        for bar in bars:
            height = bar.get_height()
            ax5.annotate(f'{height:+.2f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3 if height > 0 else -15),
                       textcoords="offset points",
                       ha='center', va='bottom' if height > 0 else 'top', fontsize=9)

        # Use subplots_adjust instead of tight_layout for GridSpec figures
        plt.subplots_adjust(hspace=0.3, wspace=0.3, top=0.95, bottom=0.05)

        # Save figure
        file_path = self.output_dir / "quarterly_impact_summary.png"
        plt.savefig(file_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved impact summary dashboard to {file_path}")
        return str(file_path)