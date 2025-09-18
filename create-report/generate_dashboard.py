#!/usr/bin/env python3
"""
Daily Weight Analysis Dashboard Generator
Creates comprehensive visualizations and insights from daily weight tracking data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import os

warnings.filterwarnings('ignore', category=FutureWarning)

# Configure plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DATA_FILE = Path("daily_weight_analysis.csv")
OUTPUT_DIR = Path("visualizations")

def load_data(filepath: Path) -> pd.DataFrame:
    """Load and preprocess daily weight analysis data"""
    logger.info(f"Loading data from {filepath}")
    
    df = pd.read_csv(filepath)
    
    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Convert boolean columns
    bool_columns = ['has_raw_measurement', 'has_filtered_measurement']
    for col in bool_columns:
        df[col] = df[col].astype(bool)
    
    # Clean numeric columns (handle empty strings and NaNs)
    numeric_columns = [
        'raw_weight', 'filtered_weight', 
        'raw_cumulative_loss_kg', 'filtered_cumulative_loss_kg',
        'raw_cumulative_loss_pct', 'filtered_cumulative_loss_pct',
        'divergence_kg', 'divergence_pct'
    ]
    
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    logger.info(f"Loaded {len(df)} records for {df['user_id'].nunique()} users")
    logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    return df

def calculate_statistics(df: pd.DataFrame) -> Dict:
    """Calculate comprehensive statistics from the data"""
    stats = {}
    
    # Overall statistics
    stats['total_users'] = df['user_id'].nunique()
    stats['total_records'] = len(df)
    stats['date_range'] = f"{df['date'].min().date()} to {df['date'].max().date()}"
    
    # Calculate per-user stats at day 90
    day_90 = df[df['day_number'] == 90].copy()
    
    if not day_90.empty:
        # Weight loss statistics
        stats['avg_raw_loss_kg'] = day_90['raw_cumulative_loss_kg'].mean()
        stats['avg_filtered_loss_kg'] = day_90['filtered_cumulative_loss_kg'].mean()
        stats['avg_raw_loss_pct'] = day_90['raw_cumulative_loss_pct'].mean()
        stats['avg_filtered_loss_pct'] = day_90['filtered_cumulative_loss_pct'].mean()
        
        # Success metrics (>5% weight loss)
        stats['raw_success_rate'] = (day_90['raw_cumulative_loss_pct'] > 5).mean() * 100
        stats['filtered_success_rate'] = (day_90['filtered_cumulative_loss_pct'] > 5).mean() * 100
        
        # Divergence statistics
        stats['avg_divergence_kg'] = day_90['divergence_kg'].mean()
        stats['max_divergence_kg'] = day_90['divergence_kg'].max()
    
    # Data quality metrics
    stats['raw_measurement_rate'] = df['has_raw_measurement'].mean() * 100
    stats['filtered_measurement_rate'] = df['has_filtered_measurement'].mean() * 100
    
    # Retention metrics
    retention_by_day = df.groupby('day_number')['user_id'].nunique()
    initial_users = retention_by_day.iloc[0] if not retention_by_day.empty else 1
    stats['day_30_retention'] = (retention_by_day.get(30, 0) / initial_users) * 100
    stats['day_60_retention'] = (retention_by_day.get(60, 0) / initial_users) * 100
    stats['day_90_retention'] = (retention_by_day.get(90, 0) / initial_users) * 100
    
    return stats

def create_dashboard(df: pd.DataFrame, stats: Dict, output_dir: Path) -> Path:
    """Create comprehensive dashboard with multiple visualizations

    Returns:
        Path to the generated dashboard file
    """
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(20, 24))
    gs = gridspec.GridSpec(6, 3, figure=fig, hspace=0.3, wspace=0.25)
    
    # Title and statistics header
    fig.suptitle('Daily Weight Analysis Dashboard', fontsize=20, fontweight='bold', y=0.995)
    
    # Add key metrics as text
    metrics_text = (
        f"Total Users: {stats['total_users']:,} | "
        f"Date Range: {stats['date_range']} | "
        f"90-Day Success Rate: Raw {stats.get('raw_success_rate', 0):.1f}% vs Filtered {stats.get('filtered_success_rate', 0):.1f}%"
    )
    fig.text(0.5, 0.98, metrics_text, ha='center', fontsize=12, style='italic')
    
    # 1. Average Weight Loss Progression
    ax1 = fig.add_subplot(gs[0, :])
    plot_weight_loss_progression(df, ax1)
    
    # 2. Success Rate Comparison
    ax2 = fig.add_subplot(gs[1, 0])
    plot_success_rates(df, ax2)
    
    # 3. Weight Loss Distribution
    ax3 = fig.add_subplot(gs[1, 1])
    plot_loss_distribution(df, ax3)
    
    # 4. Data Quality Metrics
    ax4 = fig.add_subplot(gs[1, 2])
    plot_data_quality(df, ax4)
    
    # 5. User Retention
    ax5 = fig.add_subplot(gs[2, 0])
    plot_retention_curve(df, ax5)
    
    # 6. Divergence Analysis
    ax6 = fig.add_subplot(gs[2, 1:])
    plot_divergence_analysis(df, ax6)
    
    # 7. Individual User Examples
    ax7 = fig.add_subplot(gs[3, :])
    plot_user_examples(df, ax7)
    
    # 8. Cohort Analysis
    ax8 = fig.add_subplot(gs[4, :2])
    plot_cohort_analysis(df, ax8)
    
    # 9. Statistical Summary Table
    ax9 = fig.add_subplot(gs[4, 2])
    plot_statistics_table(stats, ax9)
    
    # 10. Insights and Recommendations
    ax10 = fig.add_subplot(gs[5, :])
    plot_insights(df, stats, ax10)
    
    # Save the dashboard
    output_path = output_dir / f"dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(output_path, dpi=100, bbox_inches='tight', facecolor='white')
    logger.info(f"Dashboard saved to {output_path}")

    plt.close()

    return output_path

def plot_weight_loss_progression(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Plot average weight loss progression over time"""
    
    # Calculate average weight loss by day
    daily_avg = df.groupby('day_number').agg({
        'raw_cumulative_loss_pct': 'mean',
        'filtered_cumulative_loss_pct': 'mean',
        'user_id': 'nunique'
    }).reset_index()
    
    # Plot lines
    ax.plot(daily_avg['day_number'], daily_avg['raw_cumulative_loss_pct'], 
            label='Raw Data', linewidth=2, color='steelblue', alpha=0.8)
    ax.plot(daily_avg['day_number'], daily_avg['filtered_cumulative_loss_pct'], 
            label='Filtered Data', linewidth=2, color='darkorange', alpha=0.8)
    
    # Add confidence bands (standard error)
    for col, color in [('raw_cumulative_loss_pct', 'steelblue'), 
                       ('filtered_cumulative_loss_pct', 'darkorange')]:
        daily_std = df.groupby('day_number')[col].std()
        daily_count = df.groupby('day_number')[col].count()
        std_error = daily_std / np.sqrt(daily_count)
        
        ax.fill_between(daily_avg['day_number'],
                        daily_avg[col] - std_error,
                        daily_avg[col] + std_error,
                        alpha=0.2, color=color)
    
    # Add clinical target line
    ax.axhline(y=5, color='green', linestyle='--', alpha=0.5, label='5% Target')
    
    ax.set_xlabel('Day Number', fontsize=11)
    ax.set_ylabel('Average Weight Loss (%)', fontsize=11)
    ax.set_title('Average Weight Loss Progression Over 90 Days', fontsize=13, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 90)

def plot_success_rates(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Plot success rate comparison by threshold"""
    
    thresholds = [3, 5, 7, 10]
    day_90 = df[df['day_number'] == 90]
    
    raw_rates = []
    filtered_rates = []
    
    for threshold in thresholds:
        raw_rates.append((day_90['raw_cumulative_loss_pct'] >= threshold).mean() * 100)
        filtered_rates.append((day_90['filtered_cumulative_loss_pct'] >= threshold).mean() * 100)
    
    x = np.arange(len(thresholds))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, raw_rates, width, label='Raw Data', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, filtered_rates, width, label='Filtered Data', color='darkorange', alpha=0.8)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Weight Loss Threshold (%)', fontsize=11)
    ax.set_ylabel('Success Rate (%)', fontsize=11)
    ax.set_title('Success Rates by Weight Loss Threshold', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'≥{t}%' for t in thresholds])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

def plot_loss_distribution(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Plot distribution of weight loss at day 90"""
    
    day_90 = df[df['day_number'] == 90]
    
    # Create violin plot
    data_to_plot = [
        day_90['raw_cumulative_loss_pct'].dropna(),
        day_90['filtered_cumulative_loss_pct'].dropna()
    ]
    
    parts = ax.violinplot(data_to_plot, positions=[1, 2], widths=0.7,
                          showmeans=True, showmedians=True)
    
    # Customize colors
    colors = ['steelblue', 'darkorange']
    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    
    # Add box plot overlay
    bp = ax.boxplot(data_to_plot, positions=[1, 2], widths=0.3,
                    patch_artist=True, showfliers=False)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.3)
    
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Raw Data', 'Filtered Data'])
    ax.set_ylabel('Weight Loss at Day 90 (%)', fontsize=11)
    ax.set_title('Weight Loss Distribution at 90 Days', fontsize=12, fontweight='bold')
    ax.axhline(y=5, color='green', linestyle='--', alpha=0.5, label='5% Target')
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

def plot_data_quality(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Plot data quality metrics"""
    
    # Calculate measurement rates by time period
    df['period'] = pd.cut(df['day_number'], 
                          bins=[0, 30, 60, 90], 
                          labels=['Days 1-30', 'Days 31-60', 'Days 61-90'])
    
    quality_metrics = df.groupby('period').agg({
        'has_raw_measurement': 'mean',
        'has_filtered_measurement': 'mean'
    }).multiply(100)
    
    quality_metrics.plot(kind='bar', ax=ax, color=['steelblue', 'darkorange'], alpha=0.8)
    
    ax.set_xlabel('Time Period', fontsize=11)
    ax.set_ylabel('Measurement Rate (%)', fontsize=11)
    ax.set_title('Data Quality Over Time', fontsize=12, fontweight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.legend(['Raw Data', 'Filtered Data'], loc='best')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', fontsize=9)

def plot_retention_curve(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Plot user retention curve"""
    
    retention = df.groupby('day_number')['user_id'].nunique()
    initial_users = retention.iloc[0] if not retention.empty else 1
    retention_pct = (retention / initial_users) * 100
    
    ax.plot(retention_pct.index, retention_pct.values, 
            linewidth=2, color='darkgreen', marker='o', markersize=2)
    
    # Add key milestone markers
    milestones = [30, 60, 90]
    for milestone in milestones:
        if milestone in retention_pct.index:
            value = retention_pct[milestone]
            ax.plot(milestone, value, 'ro', markersize=8)
            ax.annotate(f'{value:.1f}%', xy=(milestone, value), 
                       xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    ax.set_xlabel('Day Number', fontsize=11)
    ax.set_ylabel('User Retention (%)', fontsize=11)
    ax.set_title('User Retention Over 90 Days', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 90)
    ax.set_ylim(0, 105)

def plot_divergence_analysis(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Plot divergence between raw and filtered data over time"""
    
    # Calculate average absolute divergence by day
    daily_divergence = df.groupby('day_number')['divergence_pct'].agg(['mean', 'std'])
    
    # Plot mean divergence
    ax.plot(daily_divergence.index, daily_divergence['mean'].abs(), 
            linewidth=2, color='purple', label='Mean Absolute Divergence')
    
    # Add confidence band
    ax.fill_between(daily_divergence.index,
                    (daily_divergence['mean'] - daily_divergence['std']).abs(),
                    (daily_divergence['mean'] + daily_divergence['std']).abs(),
                    alpha=0.3, color='purple')
    
    # Add trend line
    z = np.polyfit(daily_divergence.index, daily_divergence['mean'].abs(), 2)
    p = np.poly1d(z)
    ax.plot(daily_divergence.index, p(daily_divergence.index), 
            '--', color='red', alpha=0.5, label='Trend')
    
    ax.set_xlabel('Day Number', fontsize=11)
    ax.set_ylabel('Divergence (%)', fontsize=11)
    ax.set_title('Divergence Between Raw and Filtered Data Over Time', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 90)

def _process_user_journey(user_id: str, df: pd.DataFrame, index: int) -> Dict:
    """Process a single user's journey data for plotting."""
    user_data = df[df['user_id'] == user_id].sort_values('day_number')
    if not user_data.empty:
        final_loss = user_data['filtered_cumulative_loss_pct'].iloc[-1]
        return {
            'index': index,
            'user_id': user_id,
            'days': user_data['day_number'].values,
            'loss': user_data['filtered_cumulative_loss_pct'].values,
            'final_loss': final_loss
        }
    return None


def plot_user_examples(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Plot examples of individual user journeys with parallel data processing"""

    start_time = time.time()

    # Select diverse users based on outcomes at day 90
    day_90 = df[df['day_number'] == 90]

    # Get percentiles for selection
    percentiles = [10, 25, 50, 75, 90]
    selected_users = []

    for p in percentiles:
        target_value = day_90['filtered_cumulative_loss_pct'].quantile(p/100)
        closest_user = day_90.iloc[(day_90['filtered_cumulative_loss_pct'] - target_value).abs().argsort()[:1]]
        if not closest_user.empty:
            selected_users.append(closest_user['user_id'].values[0])

    # Limit to 5 users max
    selected_users = selected_users[:5]

    # Process user data in parallel
    user_journeys = []
    n_workers = min(len(selected_users), 4)  # Limit workers for small datasets

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        # Submit all user processing tasks
        futures = {
            executor.submit(_process_user_journey, user_id, df, i): (user_id, i)
            for i, user_id in enumerate(selected_users)
        }

        # Collect results as they complete
        for future in as_completed(futures):
            try:
                journey = future.result()
                if journey:
                    user_journeys.append(journey)
            except Exception as e:
                logger.error(f"Failed to process user journey: {e}")

    # Sort by original index to maintain consistent ordering
    user_journeys.sort(key=lambda x: x['index'])

    # Plot each user's journey (matplotlib plotting must be done in main thread)
    colors = plt.cm.viridis(np.linspace(0, 1, len(user_journeys)))

    for journey, color in zip(user_journeys, colors):
        label = f"User {journey['index']+1} (Final: {journey['final_loss']:.1f}%)"
        ax.plot(journey['days'], journey['loss'],
                linewidth=1.5, color=color, alpha=0.7, label=label)

    ax.axhline(y=5, color='green', linestyle='--', alpha=0.5, label='5% Target')
    ax.set_xlabel('Day Number', fontsize=11)
    ax.set_ylabel('Cumulative Weight Loss (%)', fontsize=11)
    ax.set_title('Sample Individual User Journeys (Filtered Data)', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 90)

    elapsed = time.time() - start_time
    logger.debug(f"User journeys plot completed in {elapsed:.2f}s (parallel)")

def plot_cohort_analysis(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Create cohort analysis heatmap"""
    
    # Create weekly cohorts
    df['week'] = df['day_number'] // 7
    
    # Calculate average weight loss by week for cohorts
    cohort_data = df.pivot_table(
        values='filtered_cumulative_loss_pct',
        index='week',
        columns='day_number',
        aggfunc='mean'
    )
    
    # Select every 7th day for cleaner visualization
    days_to_show = list(range(0, 91, 7))
    cohort_data = cohort_data[cohort_data.columns.intersection(days_to_show)]
    
    # Create heatmap
    sns.heatmap(cohort_data, ax=ax, cmap='RdYlGn', center=5, 
                annot=False, fmt='.1f', cbar_kws={'label': 'Weight Loss (%)'})
    
    ax.set_xlabel('Day Number', fontsize=11)
    ax.set_ylabel('Week Started', fontsize=11)
    ax.set_title('Weight Loss Progression by Weekly Cohorts', fontsize=12, fontweight='bold')

def plot_statistics_table(stats: Dict, ax: plt.Axes) -> None:
    """Create a statistical summary table"""
    
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    table_data = [
        ['Metric', 'Value'],
        ['Total Users', f"{stats.get('total_users', 0):,}"],
        ['Total Records', f"{stats.get('total_records', 0):,}"],
        ['Avg Loss (Raw)', f"{stats.get('avg_raw_loss_pct', 0):.2f}%"],
        ['Avg Loss (Filtered)', f"{stats.get('avg_filtered_loss_pct', 0):.2f}%"],
        ['Success Rate (Raw)', f"{stats.get('raw_success_rate', 0):.1f}%"],
        ['Success Rate (Filtered)', f"{stats.get('filtered_success_rate', 0):.1f}%"],
        ['90-Day Retention', f"{stats.get('day_90_retention', 0):.1f}%"],
        ['Max Divergence', f"{stats.get('max_divergence_kg', 0):.2f} kg"],
    ]
    
    table = ax.table(cellText=table_data, loc='center', cellLoc='left',
                    colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # Style header row
    for i in range(2):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data)):
        for j in range(2):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    ax.set_title('Statistical Summary', fontsize=12, fontweight='bold', y=0.95)

def plot_insights(df: pd.DataFrame, stats: Dict, ax: plt.Axes) -> None:
    """Generate and display key insights"""
    
    ax.axis('off')
    
    # Generate insights
    insights = []
    
    # Success rate improvement
    if 'filtered_success_rate' in stats and 'raw_success_rate' in stats:
        improvement = stats['filtered_success_rate'] - stats['raw_success_rate']
        if improvement > 0:
            insights.append(f"✓ Filtering improves success rate by {improvement:.1f} percentage points")
        else:
            insights.append(f"⚠ Raw data shows {-improvement:.1f}pp higher success rate than filtered")
    
    # Average weight loss comparison
    if 'avg_filtered_loss_pct' in stats and 'avg_raw_loss_pct' in stats:
        diff = stats['avg_filtered_loss_pct'] - stats['avg_raw_loss_pct']
        insights.append(f"📊 Filtered data shows {abs(diff):.2f}% {'higher' if diff > 0 else 'lower'} average weight loss")
    
    # Retention analysis
    if 'day_90_retention' in stats:
        retention = stats['day_90_retention']
        if retention > 70:
            insights.append(f"⭐ Excellent 90-day retention rate of {retention:.1f}%")
        elif retention > 50:
            insights.append(f"✓ Good 90-day retention rate of {retention:.1f}%")
        else:
            insights.append(f"⚠ Low 90-day retention rate of {retention:.1f}% needs attention")
    
    # Data quality
    if 'filtered_measurement_rate' in stats and 'raw_measurement_rate' in stats:
        quality_diff = stats['raw_measurement_rate'] - stats['filtered_measurement_rate']
        if quality_diff > 10:
            insights.append(f"🔍 Filtering removes {quality_diff:.1f}% of measurements - check filter sensitivity")
    
    # Calculate trend consistency
    day_30 = df[df['day_number'] == 30]
    day_60 = df[df['day_number'] == 60]
    day_90 = df[df['day_number'] == 90]
    
    if not day_30.empty and not day_60.empty and not day_90.empty:
        avg_30 = day_30['filtered_cumulative_loss_pct'].mean()
        avg_60 = day_60['filtered_cumulative_loss_pct'].mean()
        avg_90 = day_90['filtered_cumulative_loss_pct'].mean()
        
        if avg_30 > 0 and avg_60 > avg_30 and avg_90 > avg_60:
            insights.append("📈 Consistent progressive weight loss pattern observed across all periods")
        elif avg_90 < avg_60:
            insights.append("⚠ Weight loss plateau or reversal detected after day 60")
    
    # Add recommendations
    recommendations = [
        "\nRecommendations:",
        "• Focus on users with <3% loss at day 30 for early intervention",
        "• Investigate outliers with extreme divergence between raw and filtered data",
        "• Consider adjusting filter parameters if >15% of data is being filtered out",
        "• Implement engagement strategies for users showing plateau patterns"
    ]
    
    # Display insights
    y_position = 0.95
    for insight in insights:
        ax.text(0.05, y_position, insight, fontsize=11, transform=ax.transAxes,
                verticalalignment='top', wrap=True)
        y_position -= 0.12
    
    y_position -= 0.05
    for rec in recommendations:
        style = 'italic' if rec.startswith("•") else 'normal'
        weight = 'normal' if rec.startswith("•") else 'bold'
        ax.text(0.05, y_position, rec, fontsize=10, transform=ax.transAxes,
                verticalalignment='top', style=style, weight=weight, wrap=True)
        y_position -= 0.08
    
    ax.set_title('Key Insights & Recommendations', fontsize=13, fontweight='bold', 
                loc='left', x=0, y=1.05)

def main():
    """Main execution function"""

    logger.info("Starting Dashboard Generation")
    logger.info("=" * 50)

    try:
        # Load data
        df = load_data(DATA_FILE)

        # Calculate statistics
        logger.info("Calculating statistics...")
        stats = calculate_statistics(df)

        # Log key metrics
        logger.info("\nKey Metrics:")
        logger.info(f"  Total Users: {stats['total_users']:,}")
        logger.info(f"  Date Range: {stats['date_range']}")
        logger.info(f"  90-Day Success Rate (>5% loss):")
        logger.info(f"    Raw Data: {stats.get('raw_success_rate', 0):.1f}%")
        logger.info(f"    Filtered Data: {stats.get('filtered_success_rate', 0):.1f}%")
        logger.info(f"  Average Weight Loss at 90 Days:")
        logger.info(f"    Raw Data: {stats.get('avg_raw_loss_pct', 0):.2f}%")
        logger.info(f"    Filtered Data: {stats.get('avg_filtered_loss_pct', 0):.2f}%")

        # Create dashboard
        logger.info("\nGenerating dashboard visualizations...")
        dashboard_file = create_dashboard(df, stats, OUTPUT_DIR)

        # Save statistics to JSON
        import json
        stats_file = OUTPUT_DIR / "dashboard_stats.json"

        # Convert non-serializable values
        stats_export = {k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                       for k, v in stats.items()}

        with open(stats_file, 'w') as f:
            json.dump(stats_export, f, indent=2)
        logger.info(f"Statistics saved to {stats_file}")

        logger.info("\n✅ Dashboard generation complete!")

        return dashboard_file

    except Exception as e:
        logger.error(f"Error generating dashboard: {str(e)}")
        raise

if __name__ == "__main__":
    main()