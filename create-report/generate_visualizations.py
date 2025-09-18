#!/usr/bin/env python3
"""
Visualization Generation Module
Creates charts comparing raw vs filtered data effectiveness
"""

# Set matplotlib backend to non-interactive before any other matplotlib imports
# This prevents GUI operations in threads which causes crashes on macOS
import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

# Configure plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

# Constants
DATA_DIR = Path("../data")
# These can be overridden by run_analysis.py
RAW_FILE = DATA_DIR / "2025-09-05_nocon.csv"
FILTERED_FILE = DATA_DIR / "2025-09-05_nocon_filtered.csv"

def create_weight_loss_distribution(df_90_day: pd.DataFrame, output_dir: Path) -> None:
    """
    Chart 1: Weight Loss Distribution Histogram
    Compares raw vs filtered weight loss distributions
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Filter to complete data
    complete = df_90_day[
        (df_90_day['raw_loss_pct'].notna()) & 
        (df_90_day['filtered_loss_pct'].notna())
    ]
    
    if complete.empty:
        logging.warning("No complete data for distribution plot")
        return
    
    # Define bins for consistency
    bins = np.arange(-10, 25, 2.5)
    
    # Plot raw distribution
    ax1.hist(complete['raw_loss_pct'], bins=bins, alpha=0.7, color='steelblue', 
             edgecolor='black', label='Raw Data')
    ax1.axvline(complete['raw_loss_pct'].mean(), color='darkblue', 
                linestyle='--', linewidth=2, label=f'Mean: {complete["raw_loss_pct"].mean():.1f}%')
    ax1.axvline(0, color='red', linestyle='-', alpha=0.3, linewidth=1)
    ax1.set_xlabel('Weight Loss (%)', fontsize=12)
    ax1.set_ylabel('Number of Users', fontsize=12)
    ax1.set_title('Raw Data: 90-Day Weight Loss Distribution', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add success rate annotation
    success_rate_raw = (complete['raw_loss_pct'] > 0).mean() * 100
    ax1.text(0.95, 0.95, f'Success Rate: {success_rate_raw:.1f}%', 
             transform=ax1.transAxes, ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot filtered distribution
    ax2.hist(complete['filtered_loss_pct'], bins=bins, alpha=0.7, color='seagreen', 
             edgecolor='black', label='Filtered Data')
    ax2.axvline(complete['filtered_loss_pct'].mean(), color='darkgreen', 
                linestyle='--', linewidth=2, label=f'Mean: {complete["filtered_loss_pct"].mean():.1f}%')
    ax2.axvline(0, color='red', linestyle='-', alpha=0.3, linewidth=1)
    ax2.set_xlabel('Weight Loss (%)', fontsize=12)
    ax2.set_ylabel('Number of Users', fontsize=12)
    ax2.set_title('Filtered Data: 90-Day Weight Loss Distribution', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add success rate annotation
    success_rate_filtered = (complete['filtered_loss_pct'] > 0).mean() * 100
    ax2.text(0.95, 0.95, f'Success Rate: {success_rate_filtered:.1f}%', 
             transform=ax2.transAxes, ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('90-Day Weight Loss: Raw vs Filtered Data Comparison', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_file = output_dir / "chart1_distribution.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close(fig)  # Explicitly close the specific figure
    logging.info(f"Saved distribution chart to {output_file}")

def create_individual_journeys(df_90_day: pd.DataFrame, output_dir: Path) -> None:
    """
    Chart 2: Individual Journey Comparison
    Shows 6 representative users with raw vs filtered trajectories
    """
    # Load full data for selected users
    df_raw = pd.read_csv(RAW_FILE, usecols=['user_id', 'effectiveDateTime', 'weight'])
    df_filtered = pd.read_csv(FILTERED_FILE, usecols=['user_id', 'effectiveDateTime', 'weight'])
    df_raw['effectiveDateTime'] = pd.to_datetime(df_raw['effectiveDateTime'])
    df_filtered['effectiveDateTime'] = pd.to_datetime(df_filtered['effectiveDateTime'])
    
    # Select representative users
    complete = df_90_day[
        (df_90_day['raw_loss_pct'].notna()) & 
        (df_90_day['filtered_loss_pct'].notna())
    ].copy()
    
    if len(complete) < 6:
        logging.warning("Not enough users for journey plots")
        return
    
    # Select users with different outcomes
    selected_users = []
    
    # High success (>10% loss)
    high_success = complete[complete['filtered_loss_pct'] > 10]
    if not high_success.empty:
        selected_users.extend(high_success.sample(min(2, len(high_success)))['user_id'].tolist())
    
    # Moderate success (5-10% loss)
    moderate = complete[(complete['filtered_loss_pct'] > 5) & (complete['filtered_loss_pct'] <= 10)]
    if not moderate.empty:
        selected_users.extend(moderate.sample(min(2, len(moderate)))['user_id'].tolist())
    
    # Minimal/no loss
    minimal = complete[complete['filtered_loss_pct'] <= 5]
    if not minimal.empty:
        selected_users.extend(minimal.sample(min(2, len(minimal)))['user_id'].tolist())
    
    # Ensure we have 6 users
    selected_users = selected_users[:6]
    
    if len(selected_users) < 6:
        # Fill with random users if needed
        remaining = complete[~complete['user_id'].isin(selected_users)]
        if not remaining.empty:
            additional = remaining.sample(min(6 - len(selected_users), len(remaining)))
            selected_users.extend(additional['user_id'].tolist())
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, user_id in enumerate(selected_users[:6]):
        ax = axes[idx]
        
        # Get user data
        user_info = complete[complete['user_id'] == user_id].iloc[0]
        user_raw = df_raw[df_raw['user_id'] == user_id].sort_values('effectiveDateTime')
        user_filtered = df_filtered[df_filtered['user_id'] == user_id].sort_values('effectiveDateTime')
        
        if user_raw.empty and user_filtered.empty:
            continue
        
        # Calculate days from start
        start_date = pd.to_datetime(user_info['start_date'])
        
        if not user_raw.empty:
            user_raw['days_from_start'] = (user_raw['effectiveDateTime'] - start_date).dt.days
            # Plot raw as scatter points
            ax.scatter(user_raw['days_from_start'], user_raw['weight'], 
                      alpha=0.3, color='lightcoral', s=20, label='Raw (all)')
        
        if not user_filtered.empty:
            user_filtered['days_from_start'] = (user_filtered['effectiveDateTime'] - start_date).dt.days
            # Plot filtered as line
            ax.plot(user_filtered['days_from_start'], user_filtered['weight'], 
                   color='darkgreen', linewidth=2, marker='o', markersize=4, 
                   label='Filtered (accepted)', alpha=0.8)
        
        # Mark start and 90-day points
        if user_info['raw_start_weight']:
            ax.scatter([0], [user_info['raw_start_weight']], color='blue', 
                      s=100, marker='s', label='Start', zorder=5)
        
        if user_info['filtered_90_day_weight']:
            ax.scatter([90], [user_info['filtered_90_day_weight']], color='red', 
                      s=100, marker='^', label='90-day', zorder=5)
        
        # Add vertical line at 90 days
        ax.axvline(90, color='gray', linestyle='--', alpha=0.5)
        
        # Labels and title
        loss_pct = user_info['filtered_loss_pct']
        diff_pct = user_info['difference_pct']
        ax.set_title(f'User {idx+1}: {loss_pct:.1f}% loss (Δ{diff_pct:+.1f}%)', 
                    fontsize=11, fontweight='bold')
        ax.set_xlabel('Days from Start', fontsize=10)
        ax.set_ylabel('Weight (kg)', fontsize=10)
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
        
        # Set x-axis limits
        ax.set_xlim(-10, 100)
    
    plt.suptitle('Individual Weight Loss Journeys: Raw vs Filtered Data', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_file = output_dir / "chart2_journeys.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close(fig)  # Explicitly close the specific figure
    logging.info(f"Saved journey chart to {output_file}")

def create_filtering_impact_timeline(output_dir: Path) -> None:
    """
    Chart 3: Filtering Impact by Time
    Shows average weight loss over time for raw vs filtered
    """
    # This requires interval analysis data - we'll generate it
    from analyze_90_day import load_eligible_users, get_weight_at_date
    
    # Load data
    df_raw = pd.read_csv(RAW_FILE, usecols=['user_id', 'effectiveDateTime', 'weight'])
    df_filtered = pd.read_csv(FILTERED_FILE, usecols=['user_id', 'effectiveDateTime', 'weight'])
    df_raw['effectiveDateTime'] = pd.to_datetime(df_raw['effectiveDateTime'])
    df_filtered['effectiveDateTime'] = pd.to_datetime(df_filtered['effectiveDateTime'])
    
    # Get eligible users
    user_start_dates = load_eligible_users()
    
    # Calculate average weight loss at intervals
    intervals = [0, 30, 60, 90, 120, 150, 180]
    raw_avg_loss = []
    filtered_avg_loss = []
    user_counts = []
    
    for days in intervals:
        raw_losses = []
        filtered_losses = []
        
        for user_id, start_date in list(user_start_dates.items())[:200]:  # Sample for performance
            target_date = start_date + timedelta(days=days)
            
            # Get user data
            user_raw = df_raw[df_raw['user_id'] == user_id]
            user_filtered = df_filtered[df_filtered['user_id'] == user_id]
            
            # Get weights
            start_weight_raw = get_weight_at_date(user_raw, start_date)
            current_weight_raw = get_weight_at_date(user_raw, target_date)
            
            start_weight_filtered = get_weight_at_date(user_filtered, start_date)
            current_weight_filtered = get_weight_at_date(user_filtered, target_date)
            
            # Calculate losses
            if start_weight_raw and current_weight_raw:
                loss_pct = ((start_weight_raw - current_weight_raw) / start_weight_raw) * 100
                raw_losses.append(loss_pct)
            
            if start_weight_filtered and current_weight_filtered:
                loss_pct = ((start_weight_filtered - current_weight_filtered) / start_weight_filtered) * 100
                filtered_losses.append(loss_pct)
        
        raw_avg_loss.append(np.mean(raw_losses) if raw_losses else 0)
        filtered_avg_loss.append(np.mean(filtered_losses) if filtered_losses else 0)
        user_counts.append(len(filtered_losses))
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[3, 1])
    
    # Main plot
    ax1.plot(intervals, raw_avg_loss, 'o-', color='steelblue', linewidth=2.5, 
             markersize=8, label='Raw Data', alpha=0.8)
    ax1.plot(intervals, filtered_avg_loss, 's-', color='seagreen', linewidth=2.5, 
             markersize=8, label='Filtered Data', alpha=0.8)
    
    # Fill area between lines
    ax1.fill_between(intervals, raw_avg_loss, filtered_avg_loss, 
                     where=[f >= r for f, r in zip(filtered_avg_loss, raw_avg_loss)],
                     color='green', alpha=0.2, label='Filtering Improvement')
    ax1.fill_between(intervals, raw_avg_loss, filtered_avg_loss, 
                     where=[f < r for f, r in zip(filtered_avg_loss, raw_avg_loss)],
                     color='red', alpha=0.2, label='Filtering Reduction')
    
    # Mark 90-day point
    ax1.axvline(90, color='red', linestyle='--', linewidth=2, alpha=0.7, label='90-Day Mark')
    ax1.axhline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    
    # Annotations
    for i, days in enumerate(intervals):
        if days in [30, 90, 180]:
            diff = filtered_avg_loss[i] - raw_avg_loss[i]
            ax1.annotate(f'Δ{diff:+.1f}%', 
                        xy=(days, (filtered_avg_loss[i] + raw_avg_loss[i])/2),
                        xytext=(10, 0), textcoords='offset points',
                        fontsize=9, ha='left',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
    
    ax1.set_ylabel('Average Weight Loss (%)', fontsize=12)
    ax1.set_title('Weight Loss Progression: Impact of Filtering Over Time', 
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-5, 185)
    
    # User count subplot
    ax2.bar(intervals, user_counts, width=15, color='lightgray', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Days from Start', fontsize=12)
    ax2.set_ylabel('Users with Data', fontsize=10)
    ax2.set_xlim(-5, 185)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_file = output_dir / "chart3_timeline.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close(fig)  # Explicitly close the specific figure
    logging.info(f"Saved timeline chart to {output_file}")

def create_quality_metrics_dashboard(df_90_day: pd.DataFrame, output_dir: Path) -> None:
    """
    Chart 4: Data Quality Metrics Dashboard
    Multi-panel visualization of filtering effectiveness
    """
    # Load full data for quality analysis
    df_raw = pd.read_csv(RAW_FILE, usecols=['user_id', 'effectiveDateTime', 'weight'])
    df_filtered = pd.read_csv(FILTERED_FILE, usecols=['user_id', 'effectiveDateTime', 'weight'])
    df_raw['effectiveDateTime'] = pd.to_datetime(df_raw['effectiveDateTime'])
    df_filtered['effectiveDateTime'] = pd.to_datetime(df_filtered['effectiveDateTime'])
    
    # Get users with complete 90-day data
    complete = df_90_day[
        (df_90_day['raw_loss_pct'].notna()) & 
        (df_90_day['filtered_loss_pct'].notna())
    ]
    
    if complete.empty:
        logging.warning("No complete data for quality metrics")
        return
    
    # Calculate metrics for each user
    variance_data = []
    smoothness_data = []
    removal_rates = []
    consistency_data = []
    
    for user_id in complete['user_id'].sample(min(100, len(complete))):
        user_raw = df_raw[df_raw['user_id'] == user_id]['weight'].values
        user_filtered = df_filtered[df_filtered['user_id'] == user_id]['weight'].values
        
        if len(user_raw) > 1 and len(user_filtered) > 1:
            # Variance reduction
            var_raw = np.var(user_raw)
            var_filtered = np.var(user_filtered)
            if var_raw > 0:
                var_reduction = ((var_raw - var_filtered) / var_raw) * 100
                variance_data.append(var_reduction)
            
            # Smoothness (using first differences)
            if len(user_raw) > 2 and len(user_filtered) > 2:
                raw_diffs = np.diff(user_raw)
                filtered_diffs = np.diff(user_filtered)
                raw_smoothness = np.var(raw_diffs)
                filtered_smoothness = np.var(filtered_diffs)
                if raw_smoothness > 0:
                    smooth_improvement = ((raw_smoothness - filtered_smoothness) / raw_smoothness) * 100
                    smoothness_data.append(smooth_improvement)
            
            # Removal rate
            removal_rate = ((len(user_raw) - len(user_filtered)) / len(user_raw)) * 100
            removal_rates.append(removal_rate)
            
            # Temporal consistency (autocorrelation)
            if len(user_filtered) > 5:
                filtered_series = pd.Series(user_filtered)
                autocorr = filtered_series.autocorr(lag=1)
                consistency_data.append(autocorr)
    
    # Create dashboard
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Panel A: Variance Reduction
    ax1 = axes[0, 0]
    if variance_data:
        ax1.hist(variance_data, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
        ax1.axvline(np.mean(variance_data), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(variance_data):.1f}%')
        ax1.set_xlabel('Variance Reduction (%)', fontsize=11)
        ax1.set_ylabel('Number of Users', fontsize=11)
        ax1.set_title('A. Variance Reduction After Filtering', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Panel B: Smoothness Improvement
    ax2 = axes[0, 1]
    if smoothness_data:
        ax2.hist(smoothness_data, bins=20, color='seagreen', alpha=0.7, edgecolor='black')
        ax2.axvline(np.mean(smoothness_data), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(smoothness_data):.1f}%')
        ax2.set_xlabel('Smoothness Improvement (%)', fontsize=11)
        ax2.set_ylabel('Number of Users', fontsize=11)
        ax2.set_title('B. Trend Smoothness Improvement', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Panel C: Outlier Removal Rate
    ax3 = axes[1, 0]
    if removal_rates:
        ax3.hist(removal_rates, bins=20, color='coral', alpha=0.7, edgecolor='black')
        ax3.axvline(np.mean(removal_rates), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(removal_rates):.1f}%')
        ax3.set_xlabel('Measurements Removed (%)', fontsize=11)
        ax3.set_ylabel('Number of Users', fontsize=11)
        ax3.set_title('C. Outlier Removal Rate', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Panel D: Temporal Consistency
    ax4 = axes[1, 1]
    if consistency_data:
        ax4.hist(consistency_data, bins=20, color='mediumpurple', alpha=0.7, edgecolor='black')
        ax4.axvline(np.mean(consistency_data), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(consistency_data):.3f}')
        ax4.set_xlabel('Autocorrelation (Lag-1)', fontsize=11)
        ax4.set_ylabel('Number of Users', fontsize=11)
        ax4.set_title('D. Temporal Consistency (Higher = Better)', fontsize=12, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Data Quality Metrics: Impact of Filtering', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_file = output_dir / "chart4_quality_metrics.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close(fig)  # Explicitly close the specific figure
    logging.info(f"Saved quality metrics chart to {output_file}")

def main(input_file: Path = Path("90_day_analysis.csv"), output_dir: Path = Path("visualizations")):
    """
    Generate all visualizations from 90-day analysis data.
    
    Args:
        input_file: Path to 90_day_analysis.csv
        output_dir: Directory to save visualization files
    """
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Load 90-day analysis results
    if not input_file.exists():
        logging.error(f"Input file not found: {input_file}")
        logging.info("Running 90-day analysis first...")
        from analyze_90_day import main as analyze_main
        df_90_day, _, _ = analyze_main(output_dir=input_file.parent)
    else:
        df_90_day = pd.read_csv(input_file)
        # Convert date columns
        df_90_day['start_date'] = pd.to_datetime(df_90_day['start_date'])
    
    logging.info("Generating visualizations in parallel...")

    # Define visualization tasks
    def gen_distribution():
        logging.info("  Generating weight loss distribution chart...")
        create_weight_loss_distribution(df_90_day, output_dir)
        logging.info("  Distribution chart complete")
        return "chart1_distribution"

    def gen_journeys():
        logging.info("  Generating individual journeys chart...")
        create_individual_journeys(df_90_day, output_dir)
        logging.info("  Journeys chart complete")
        return "chart2_journeys"

    def gen_timeline():
        logging.info("  Generating timeline impact chart...")
        create_filtering_impact_timeline(output_dir)
        logging.info("  Timeline chart complete")
        return "chart3_timeline"

    def gen_dashboard():
        logging.info("  Generating quality metrics dashboard...")
        create_quality_metrics_dashboard(df_90_day, output_dir)
        logging.info("  Dashboard complete")
        return "chart4_quality_metrics"

    # Generate all charts in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(gen_distribution),
            executor.submit(gen_journeys),
            executor.submit(gen_timeline),
            executor.submit(gen_dashboard)
        ]

        # Wait for all to complete
        completed_charts = []
        for future in futures:
            try:
                chart_name = future.result()
                completed_charts.append(chart_name)
            except Exception as e:
                logging.error(f"Visualization generation failed: {e}")

    logging.info(f"\nGenerated {len(completed_charts)} visualizations in {output_dir}/")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate filtering effectiveness visualizations")
    parser.add_argument('--input', type=Path, default=Path("90_day_analysis.csv"),
                       help='Input CSV file from 90-day analysis')
    parser.add_argument('--output-dir', type=Path, default=Path("visualizations"),
                       help='Output directory for charts')
    args = parser.parse_args()
    
    main(args.input, args.output_dir)