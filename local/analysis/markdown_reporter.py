"""
Markdown Report Generator for Weight Loss Analysis

Generates comprehensive, human-readable reports with insights and analytics.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np


class MarkdownReporter:
    """Generate comprehensive Markdown reports with insights"""

    def __init__(self, output_dir: Path):
        """Initialize reporter with output directory"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_comprehensive_report(self,
                                     raw_data: pd.DataFrame,
                                     filtered_data: pd.DataFrame,
                                     user_intervals: Dict,
                                     statistics: Dict,
                                     top_users: List[Dict],
                                     csv_files: Dict,
                                     phase_timings: Dict) -> str:
        """
        Generate comprehensive Markdown report with all insights

        Args:
            raw_data: Raw measurement DataFrame
            filtered_data: Filtered measurement DataFrame
            user_intervals: User interval calculations
            statistics: Statistical analysis results
            top_users: Top divergent users
            csv_files: Generated CSV file paths
            phase_timings: Timing data for each phase

        Returns:
            Path to generated report file
        """
        report_lines = []

        # Header
        report_lines.extend([
            "# Weight Loss Analysis Report",
            f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
            "",
            "---",
            ""
        ])

        # Executive Summary
        report_lines.append("## 📊 Executive Summary")
        report_lines.append("")
        report_lines.extend(self._generate_executive_summary(
            raw_data, filtered_data, user_intervals, statistics
        ))
        report_lines.append("")

        # Key Metrics Dashboard
        report_lines.append("## 🎯 Key Metrics Dashboard")
        report_lines.append("")
        report_lines.extend(self._generate_metrics_dashboard(
            raw_data, filtered_data, user_intervals, statistics
        ))
        report_lines.append("")

        # Data Quality Analysis
        report_lines.append("## 🔍 Data Quality Analysis")
        report_lines.append("")
        report_lines.extend(self._generate_data_quality_analysis(
            raw_data, filtered_data, user_intervals
        ))
        report_lines.append("")

        # Interval Analysis
        report_lines.append("## 📈 Interval Progress Analysis")
        report_lines.append("")
        report_lines.extend(self._generate_interval_analysis(
            user_intervals, statistics
        ))
        report_lines.append("")

        # Weight Loss Patterns
        report_lines.append("## 🏃 Weight Loss Patterns")
        report_lines.append("")
        report_lines.extend(self._generate_weight_loss_patterns(
            user_intervals, statistics
        ))
        report_lines.append("")

        # Outlier Impact Analysis
        report_lines.append("## ⚠️ Outlier Impact Analysis")
        report_lines.append("")
        report_lines.extend(self._generate_outlier_analysis(
            raw_data, filtered_data, user_intervals, top_users
        ))
        report_lines.append("")

        # Top Divergent Users
        report_lines.append("## 👥 Top Divergent Users")
        report_lines.append("")
        report_lines.extend(self._generate_top_users_section(
            top_users, user_intervals
        ))
        report_lines.append("")

        # Data Source Analysis
        report_lines.append("## 📱 Data Source Analysis")
        report_lines.append("")
        report_lines.extend(self._generate_source_analysis(
            raw_data, filtered_data
        ))
        report_lines.append("")

        # Performance Metrics
        report_lines.append("## ⏱️ Report Generation Performance")
        report_lines.append("")
        report_lines.extend(self._generate_performance_section(phase_timings))
        report_lines.append("")

        # Generated Files
        report_lines.append("## 📁 Generated Files")
        report_lines.append("")
        report_lines.extend(self._generate_files_section(csv_files))
        report_lines.append("")

        # Recommendations
        report_lines.append("## 💡 Insights and Recommendations")
        report_lines.append("")
        report_lines.extend(self._generate_recommendations(
            raw_data, filtered_data, user_intervals, statistics, top_users
        ))

        # Footer
        report_lines.extend([
            "",
            "---",
            "",
            "*End of Report*",
            ""
        ])

        # Write report
        report_path = self.output_dir / f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))

        return str(report_path)

    def _generate_executive_summary(self, raw_data, filtered_data, user_intervals, statistics):
        """Generate executive summary section"""
        lines = []

        total_users = len(user_intervals)
        total_raw = len(raw_data)
        total_filtered = len(filtered_data)
        rejection_rate = (total_raw - total_filtered) / total_raw * 100 if total_raw > 0 else 0

        lines.append("### Overview")
        lines.append("")
        lines.append(f"This report analyzes weight tracking data for **{total_users:,} users** over multiple 30-day intervals.")
        lines.append(f"The Kalman filtering system processed **{total_raw:,} raw measurements** and retained ")
        lines.append(f"**{total_filtered:,} measurements** after outlier detection, representing a ")
        lines.append(f"**{rejection_rate:.1f}% rejection rate**.")
        lines.append("")

        # Key findings
        lines.append("### Key Findings")
        lines.append("")

        # Calculate average weight loss from interval statistics (it's a list, not dict)
        interval_stats_list = statistics.get('interval_statistics', [])

        # Find the 30-day and 90-day intervals (intervals 1 and 3)
        avg_loss_30d = 0
        avg_loss_90d = 0

        for interval_stat in interval_stats_list:
            if interval_stat.get('interval_num') == 1:  # 30 days
                filtered_stats = interval_stat.get('filtered', {})
                if filtered_stats and 'change' in filtered_stats:
                    avg_loss_30d = filtered_stats['change'].get('mean', 0)
            elif interval_stat.get('interval_num') == 3:  # 90 days
                filtered_stats = interval_stat.get('filtered', {})
                if filtered_stats and 'change' in filtered_stats:
                    avg_loss_90d = filtered_stats['change'].get('mean', 0)

        lines.append(f"- **Average 30-day weight change**: {avg_loss_30d:.2f} lbs")
        lines.append(f"- **Average 90-day weight change**: {avg_loss_90d:.2f} lbs")

        # Find users with best progress
        successful_users = sum(1 for uid, data in user_intervals.items()
                              if len(data['intervals']) >= 3 and
                              data['intervals'][0].get('filtered_weight') and
                              data['intervals'][2].get('filtered_weight') and
                              (data['intervals'][2]['filtered_weight'] < data['intervals'][0]['filtered_weight']))

        success_rate = successful_users / total_users * 100 if total_users > 0 else 0
        lines.append(f"- **Success rate** (weight loss at 90 days): {success_rate:.1f}%")

        # Data quality
        users_with_complete_data = sum(1 for uid, data in user_intervals.items()
                                      if all(i.get('filtered_weight') for i in data['intervals'][:4]))
        completeness = users_with_complete_data / total_users * 100 if total_users > 0 else 0
        lines.append(f"- **Data completeness** (4+ intervals): {completeness:.1f}%")

        return lines

    def _generate_metrics_dashboard(self, raw_data, filtered_data, user_intervals, statistics):
        """Generate key metrics dashboard"""
        lines = []

        # Create metrics table
        lines.append("| Metric | Raw Data | Filtered Data | Difference |")
        lines.append("|--------|----------|---------------|------------|")

        # Total measurements
        total_raw = len(raw_data)
        total_filtered = len(filtered_data)
        lines.append(f"| **Total Measurements** | {total_raw:,} | {total_filtered:,} | {total_raw - total_filtered:,} ({(total_raw - total_filtered)/total_raw*100:.1f}%) |")

        # Unique users
        users_raw = raw_data['user_id'].nunique()
        users_filtered = filtered_data['user_id'].nunique() if not filtered_data.empty else 0
        lines.append(f"| **Unique Users** | {users_raw:,} | {users_filtered:,} | {users_raw - users_filtered:,} |")

        # Average weight
        avg_weight_raw = raw_data['weight'].mean()
        avg_weight_filtered = filtered_data['weight'].mean() if not filtered_data.empty else 0
        lines.append(f"| **Average Weight** | {avg_weight_raw:.1f} lbs | {avg_weight_filtered:.1f} lbs | {abs(avg_weight_raw - avg_weight_filtered):.1f} lbs |")

        # Standard deviation
        std_raw = raw_data['weight'].std()
        std_filtered = filtered_data['weight'].std() if not filtered_data.empty else 0
        lines.append(f"| **Std Deviation** | {std_raw:.1f} lbs | {std_filtered:.1f} lbs | {abs(std_raw - std_filtered):.1f} lbs |")

        # Date range
        date_range = f"{raw_data['timestamp'].min().strftime('%Y-%m-%d')} to {raw_data['timestamp'].max().strftime('%Y-%m-%d')}"
        lines.append(f"| **Date Range** | {date_range} | Same | - |")

        return lines

    def _generate_data_quality_analysis(self, raw_data, filtered_data, user_intervals):
        """Generate data quality analysis section"""
        lines = []

        lines.append("### Measurement Distribution by Source")
        lines.append("")

        # Source distribution table
        source_counts_raw = raw_data['source'].value_counts()
        source_counts_filtered = filtered_data['source'].value_counts() if not filtered_data.empty else pd.Series()

        lines.append("| Source | Raw Count | Filtered Count | Rejection Rate |")
        lines.append("|--------|-----------|----------------|----------------|")

        for source in source_counts_raw.index:
            raw_count = source_counts_raw[source]
            filtered_count = source_counts_filtered.get(source, 0)
            rejection_rate = (raw_count - filtered_count) / raw_count * 100 if raw_count > 0 else 0
            lines.append(f"| {source} | {raw_count:,} | {filtered_count:,} | {rejection_rate:.1f}% |")

        lines.append("")
        lines.append("### User Engagement Metrics")
        lines.append("")

        # Calculate engagement metrics
        measurements_per_user = []
        intervals_with_data = []

        for uid, data in user_intervals.items():
            measurements_per_user.append(data['num_raw_measurements'])
            intervals_with_data.append(sum(1 for i in data['intervals'] if i.get('filtered_weight')))

        if measurements_per_user:
            lines.append(f"- **Average measurements per user**: {np.mean(measurements_per_user):.1f}")
            lines.append(f"- **Median measurements per user**: {np.median(measurements_per_user):.0f}")
            lines.append(f"- **Average intervals with data**: {np.mean(intervals_with_data):.1f} out of 12 possible")
            lines.append(f"- **Users with 6+ intervals**: {sum(1 for i in intervals_with_data if i >= 6):,}")

        return lines

    def _generate_interval_analysis(self, user_intervals, statistics):
        """Generate interval analysis section"""
        lines = []

        lines.append("### Weight Change by Interval")
        lines.append("")
        lines.append("Average weight changes from baseline (Day 0) for users with data:")
        lines.append("")

        lines.append("| Interval | Days | Users with Data | Avg Change (Raw) | Avg Change (Filtered) | Std Dev |")
        lines.append("|----------|------|-----------------|------------------|----------------------|---------|")

        # Get interval statistics (it's a list, not dict)
        interval_stats_list = statistics.get('interval_statistics', [])

        for interval_stat in interval_stats_list:
            interval_num = interval_stat.get('interval_num', 0)

            if interval_num == 0:
                continue  # Skip baseline

            days = interval_stat.get('interval_days', interval_num * 5)  # Now using 5-day intervals

            # Extract statistics from the structure
            raw_stats = interval_stat.get('raw', {})
            filtered_stats = interval_stat.get('filtered', {})
            n_users = interval_stat.get('n_users', {})

            count = n_users.get('filtered', 0)
            avg_change_raw = raw_stats.get('change', {}).get('mean', 0) if raw_stats else 0
            avg_change_filtered = filtered_stats.get('change', {}).get('mean', 0) if filtered_stats else 0
            std_dev = filtered_stats.get('change', {}).get('std', 0) if filtered_stats else 0

            # Format the change with arrows
            raw_arrow = "↓" if avg_change_raw < 0 else "↑"
            filtered_arrow = "↓" if avg_change_filtered < 0 else "↑"

            lines.append(f"| {interval_num} | {days} | {count:,} | {raw_arrow} {abs(avg_change_raw):.2f} lbs | {filtered_arrow} {abs(avg_change_filtered):.2f} lbs | {std_dev:.2f} |")

        lines.append("")
        lines.append("*Note: Negative values indicate weight loss*")

        return lines

    def _generate_weight_loss_patterns(self, user_intervals, statistics):
        """Generate weight loss patterns analysis"""
        lines = []

        lines.append("### Success Patterns")
        lines.append("")

        # Categorize users by weight loss success
        categories = {
            'consistent_loss': [],
            'plateau': [],
            'regain': [],
            'volatile': []
        }

        for uid, data in user_intervals.items():
            intervals = data['intervals']
            if len(intervals) < 4:
                continue

            # Get first 4 intervals
            weights = [i.get('filtered_weight') for i in intervals[:4]]
            if not all(weights):
                continue

            # Calculate changes between intervals
            changes = [weights[i+1] - weights[i] for i in range(3)]

            # Categorize
            if all(c <= 0 for c in changes):
                categories['consistent_loss'].append(uid)
            elif abs(sum(changes)) < 2:  # Less than 2 lbs total change
                categories['plateau'].append(uid)
            elif changes[0] < 0 and changes[-1] > 2:  # Lost then regained
                categories['regain'].append(uid)
            else:
                categories['volatile'].append(uid)

        total_categorized = sum(len(v) for v in categories.values())

        lines.append("| Pattern | Count | Percentage | Description |")
        lines.append("|---------|-------|------------|-------------|")
        lines.append(f"| **Consistent Loss** | {len(categories['consistent_loss'])} | {len(categories['consistent_loss'])/total_categorized*100:.1f}% | Weight loss in all intervals |")
        lines.append(f"| **Plateau** | {len(categories['plateau'])} | {len(categories['plateau'])/total_categorized*100:.1f}% | < 2 lbs total change |")
        lines.append(f"| **Regain** | {len(categories['regain'])} | {len(categories['regain'])/total_categorized*100:.1f}% | Initial loss followed by regain |")
        lines.append(f"| **Volatile** | {len(categories['volatile'])} | {len(categories['volatile'])/total_categorized*100:.1f}% | Mixed gains and losses |")

        lines.append("")
        lines.append(f"*Analysis based on {total_categorized} users with 4+ intervals of data*")

        return lines

    def _generate_outlier_analysis(self, raw_data, filtered_data, user_intervals, top_users):
        """Generate outlier impact analysis"""
        lines = []

        lines.append("### Outlier Detection Impact")
        lines.append("")

        # Overall impact
        total_raw = len(raw_data)
        total_filtered = len(filtered_data)
        outliers_removed = total_raw - total_filtered

        lines.append(f"- **Total outliers removed**: {outliers_removed:,} measurements ({outliers_removed/total_raw*100:.1f}%)")
        lines.append("")

        lines.append("### Users Most Affected by Filtering")
        lines.append("")
        lines.append("Top 10 users with highest divergence between raw and filtered data:")
        lines.append("")

        lines.append("| Rank | User ID | Divergence Score | Avg Diff | Max Diff | RMS Diff | Intervals |")
        lines.append("|------|---------|------------------|----------|----------|----------|-----------|")

        for i, user in enumerate(top_users[:10], 1):
            user_id = user['user_id']
            divergence = user['divergence_score']
            avg_diff = user.get('avg_diff', user.get('divergence_score', 0))
            max_diff = user.get('max_diff', 0)
            rms_diff = user.get('rms_diff', 0)
            num_intervals = user['num_intervals']

            # Get actual weights for comparison
            if user_id in user_intervals:
                user_data = user_intervals[user_id]
                raw_count = user_data['num_raw_measurements']
                filtered_count = user_data['num_filtered_measurements']
                reduction = f"{raw_count} → {filtered_count}"
            else:
                reduction = "N/A"

            lines.append(f"| {i} | {user_id[:8]}... | {divergence:.2f} | {avg_diff:.2f} lbs | {max_diff:.1f} lbs | {rms_diff:.2f} lbs | {num_intervals} |")

        lines.append("")
        lines.append("*Divergence = average difference between raw and filtered weights per interval*")

        return lines

    def _generate_top_users_section(self, top_users, user_intervals):
        """Generate detailed top users analysis"""
        lines = []

        if not top_users:
            lines.append("*No divergent users identified*")
            return lines

        lines.append("### Detailed Analysis of Top 5 Most Divergent Users")
        lines.append("")

        for i, user in enumerate(top_users[:5], 1):
            user_id = user['user_id']

            if user_id not in user_intervals:
                continue

            user_data = user_intervals[user_id]
            intervals = user_data['intervals']

            lines.append(f"#### User {i}: {user_id[:12]}...")
            lines.append("")
            lines.append(f"- **Signup Date**: {user_data['signup_date'].strftime('%Y-%m-%d')}")
            lines.append(f"- **Total Measurements**: Raw={user_data['num_raw_measurements']}, Filtered={user_data['num_filtered_measurements']}")
            lines.append(f"- **Divergence Score**: {user['divergence_score']:.2f} lbs")
            lines.append("")

            # Show interval progression
            lines.append("| Day | Raw Weight | Filtered Weight | Difference | Source |")
            lines.append("|-----|------------|-----------------|------------|--------|")

            for interval in intervals[:7]:  # Show first 6 months
                day = interval['interval_days']
                raw_w = f"{interval['raw_weight']:.1f}" if interval['raw_weight'] else "—"
                filt_w = f"{interval['filtered_weight']:.1f}" if interval['filtered_weight'] else "—"

                if interval['raw_weight'] and interval['filtered_weight']:
                    diff = interval['raw_weight'] - interval['filtered_weight']
                    diff_str = f"{diff:+.1f}"
                else:
                    diff_str = "—"

                source = interval.get('filtered_source', interval.get('raw_source', '—'))
                lines.append(f"| {day} | {raw_w} | {filt_w} | {diff_str} | {source} |")

            lines.append("")

        return lines

    def _generate_source_analysis(self, raw_data, filtered_data):
        """Generate data source analysis"""
        lines = []

        lines.append("### Source Reliability Analysis")
        lines.append("")

        # Calculate reliability metrics by source
        source_stats = []

        for source in raw_data['source'].unique():
            raw_source = raw_data[raw_data['source'] == source]
            filtered_source = filtered_data[filtered_data['source'] == source] if not filtered_data.empty else pd.DataFrame()

            if len(raw_source) > 0:
                retention_rate = len(filtered_source) / len(raw_source) * 100
                avg_weight = raw_source['weight'].mean()
                std_dev = raw_source['weight'].std()

                source_stats.append({
                    'source': source,
                    'retention': retention_rate,
                    'count': len(raw_source),
                    'avg_weight': avg_weight,
                    'std_dev': std_dev
                })

        # Sort by retention rate
        source_stats.sort(key=lambda x: x['retention'], reverse=True)

        lines.append("| Source | Measurements | Retention Rate | Avg Weight | Std Dev |")
        lines.append("|--------|--------------|----------------|------------|---------|")

        for stat in source_stats:
            retention_indicator = "🟢" if stat['retention'] > 90 else "🟡" if stat['retention'] > 80 else "🔴"
            lines.append(f"| {stat['source']} | {stat['count']:,} | {retention_indicator} {stat['retention']:.1f}% | {stat['avg_weight']:.1f} | {stat['std_dev']:.1f} |")

        lines.append("")
        lines.append("*🟢 >90% retention, 🟡 80-90% retention, 🔴 <80% retention*")

        return lines

    def _generate_performance_section(self, phase_timings):
        """Generate performance metrics section"""
        lines = []

        if not phase_timings:
            lines.append("*Performance data not available*")
            return lines

        total_time = sum(phase_timings.values())

        lines.append("| Phase | Time (s) | Percentage | Description |")
        lines.append("|-------|----------|------------|-------------|")

        phases = [
            ("phase1", "Data Loading (Raw)"),
            ("phase2", "Data Loading (Filtered)"),
            ("phase3", "Interval Calculation"),
            ("phase4", "Statistics Generation"),
            ("phase5", "Top User Identification"),
            ("phase6", "CSV Generation"),
            ("phase7", "Visualization"),
            ("phase8", "Report Generation")
        ]

        for phase_key, description in phases:
            if phase_key in phase_timings:
                time_val = phase_timings[phase_key]
                percentage = time_val / total_time * 100 if total_time > 0 else 0

                # Add performance indicator
                if time_val > 10:
                    indicator = "⚠️"
                elif time_val > 5:
                    indicator = "⏱️"
                else:
                    indicator = "✅"

                lines.append(f"| {description} | {time_val:.2f} | {percentage:.1f}% | {indicator} |")

        lines.append("")
        lines.append(f"**Total Time**: {total_time:.2f} seconds")

        return lines

    def _generate_files_section(self, csv_files):
        """Generate list of generated files"""
        lines = []

        lines.append("The following CSV files have been generated for further analysis:")
        lines.append("")

        for file_type, file_path in csv_files.items():
            file_name = Path(file_path).name
            lines.append(f"- **{file_type}**: `{file_name}`")

        return lines

    def _generate_recommendations(self, raw_data, filtered_data, user_intervals, statistics, top_users):
        """Generate insights and recommendations"""
        lines = []

        lines.append("Based on the analysis, here are key insights and recommendations:")
        lines.append("")

        # Calculate key metrics for recommendations
        rejection_rate = (len(raw_data) - len(filtered_data)) / len(raw_data) * 100

        # Data quality recommendations
        if rejection_rate > 20:
            lines.append("### ⚠️ High Outlier Rate Detected")
            lines.append(f"- The system rejected {rejection_rate:.1f}% of measurements")
            lines.append("- **Recommendation**: Review data collection processes, especially for sources with high rejection rates")
            lines.append("")

        # Check for sources with poor reliability
        source_rejection = {}
        for source in raw_data['source'].unique():
            raw_count = len(raw_data[raw_data['source'] == source])
            filtered_count = len(filtered_data[filtered_data['source'] == source]) if not filtered_data.empty else 0
            source_rejection[source] = (raw_count - filtered_count) / raw_count * 100

        poor_sources = [s for s, r in source_rejection.items() if r > 30]
        if poor_sources:
            lines.append("### 📱 Source Reliability Issues")
            lines.append(f"- Sources with >30% rejection: {', '.join(poor_sources)}")
            lines.append("- **Recommendation**: Investigate data quality from these sources or adjust validation thresholds")
            lines.append("")

        # User engagement
        users_with_gaps = sum(1 for uid, data in user_intervals.items()
                             if len([i for i in data['intervals'][:6] if i.get('filtered_weight')]) < 4)
        gap_percentage = users_with_gaps / len(user_intervals) * 100 if user_intervals else 0

        if gap_percentage > 40:
            lines.append("### 📊 Low User Engagement")
            lines.append(f"- {gap_percentage:.1f}% of users have significant data gaps in first 6 months")
            lines.append("- **Recommendation**: Implement engagement campaigns or reminders for regular weigh-ins")
            lines.append("")

        # Weight loss success patterns
        interval_stats_list = statistics.get('interval_statistics', [])
        if interval_stats_list:
            # Find day 90 stats (interval 3)
            day_90_stats = None
            for interval_stat in interval_stats_list:
                if interval_stat.get('interval_num') == 3:
                    day_90_stats = interval_stat
                    break

            if day_90_stats:
                filtered_stats = day_90_stats.get('filtered', {})
                avg_loss = filtered_stats.get('change', {}).get('mean', 0) if filtered_stats else 0
                if avg_loss < -5:
                    lines.append("### ✅ Positive Weight Loss Trends")
                    lines.append(f"- Average 90-day weight loss: {abs(avg_loss):.1f} lbs")
                    lines.append("- **Insight**: Program is showing effectiveness for engaged users")
                elif avg_loss > 0:
                    lines.append("### ⚠️ Weight Gain Trend Detected")
                    lines.append(f"- Average 90-day weight change: +{avg_loss:.1f} lbs")
                    lines.append("- **Recommendation**: Review program effectiveness and user adherence")
                lines.append("")

        # Top divergent users
        if top_users and top_users[0]['divergence_score'] > 10:
            lines.append("### 🔍 Significant Filtering Impact")
            lines.append(f"- Some users show >10 lb average divergence between raw and filtered data")
            lines.append("- **Recommendation**: Manual review of top divergent users for data quality issues")
            lines.append("")

        # General recommendations
        lines.append("### 💡 General Recommendations")
        lines.append("")
        lines.append("1. **Data Collection**: Focus on improving data quality from unreliable sources")
        lines.append("2. **User Engagement**: Target users with data gaps for re-engagement")
        lines.append("3. **Algorithm Tuning**: Consider adjusting outlier thresholds based on source reliability")
        lines.append("4. **Success Analysis**: Deep-dive into users with consistent weight loss to identify best practices")
        lines.append("5. **Intervention Timing**: Consider interventions for users showing plateau or regain patterns")

        return lines