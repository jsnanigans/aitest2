#!/usr/bin/env python3
"""
Create focused visualization charts showing:
1. Baseline comparison
2. Kalman filter overlay
3. Data source breakdown
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import Rectangle
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.error("matplotlib not installed. Run: uv pip install matplotlib")
    exit(1)

def parse_datetime(dt_string):
    """Parse datetime string to datetime object."""
    try:
        return datetime.fromisoformat(dt_string.replace('Z', '+00:00'))
    except:
        try:
            return datetime.strptime(dt_string, '%Y-%m-%d %H:%M:%S')
        except:
            return datetime.strptime(dt_string.split('.')[0], '%Y-%m-%dT%H:%M:%S')

def create_baseline_chart(user_id: str, data: dict, output_dir: Path):
    """Create chart focusing on baseline establishment and comparison."""
    
    time_series = data.get('time_series', [])
    if not time_series:
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[2, 1])
    
    # Extract data
    dates = [parse_datetime(ts['date']) for ts in time_series]
    weights = [ts['weight'] for ts in time_series]
    
    # Main weight chart with baseline
    ax1.scatter(dates, weights, c='blue', s=20, alpha=0.6, label='Measurements')
    ax1.plot(dates, weights, 'b-', alpha=0.2, linewidth=0.5)
    
    # Add baseline if available
    baseline = data.get('baseline_weight')
    if baseline:
        ax1.axhline(y=baseline, color='purple', linestyle='--', linewidth=2,
                   label=f'Baseline: {baseline:.1f} kg', alpha=0.8)
        
        # Add ±3% band
        band = baseline * 0.03
        ax1.fill_between([dates[0], dates[-1]], 
                        [baseline - band]*2, 
                        [baseline + band]*2,
                        alpha=0.1, color='purple', label='±3% from baseline')
        
        # Mark baseline establishment period
        if data.get('signup_date'):
            signup = parse_datetime(data['signup_date'])
            baseline_end = None
            for i, d in enumerate(dates):
                if (d - signup).days > data.get('baseline_window_days', 7):
                    baseline_end = d
                    break
            
            if baseline_end:
                ax1.axvspan(signup, baseline_end, alpha=0.1, color='green', 
                          label='Baseline period')
    
    ax1.set_title(f'Weight Tracking with Baseline - User {user_id[:8]}...', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Weight (kg)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')
    
    # Deviation from baseline chart
    if baseline:
        deviations = [(w - baseline) for w in weights]
        colors = ['green' if abs(d) < baseline * 0.03 else 'orange' if abs(d) < baseline * 0.05 else 'red' 
                  for d in deviations]
        ax2.bar(range(len(deviations)), deviations, color=colors, alpha=0.6)
        ax2.axhline(y=0, color='purple', linestyle='-', linewidth=1)
        ax2.set_ylabel('Deviation from\nBaseline (kg)', fontsize=12)
        ax2.set_xlabel('Reading Number', fontsize=12)
        ax2.grid(True, alpha=0.3, axis='y')
    
    # Format dates
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add stats box
    stats_text = f"Total Readings: {data.get('total_readings', len(weights))}\n"
    if baseline:
        stats_text += f"Baseline: {baseline:.1f} kg\n"
        stats_text += f"Baseline Confidence: {data.get('baseline_confidence', 'N/A')}\n"
        if data.get('average_deviation_from_baseline'):
            stats_text += f"Avg Deviation: {data['average_deviation_from_baseline']:.2f} kg"
    
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
            verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    output_file = output_dir / f'baseline_{user_id}.png'
    plt.savefig(output_file, dpi=100, bbox_inches='tight')
    plt.close()
    logger.info(f"Created baseline chart: {output_file.name}")

def create_kalman_chart(user_id: str, data: dict, output_dir: Path):
    """Create chart focusing on Kalman filter performance."""
    
    time_series = data.get('time_series', [])
    if not time_series:
        return
    
    # Check if we have Kalman data
    kalman_data = [(i, ts) for i, ts in enumerate(time_series) 
                   if ts.get('kalman_filtered') is not None]
    
    if not kalman_data:
        logger.warning(f"No Kalman data for user {user_id[:8]}...")
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
    
    # Extract data
    all_dates = [parse_datetime(ts['date']) for ts in time_series]
    all_weights = [ts['weight'] for ts in time_series]
    
    # Kalman specific data
    kalman_indices = [i for i, _ in kalman_data]
    kalman_dates = [all_dates[i] for i in kalman_indices]
    kalman_measured = [ts['weight'] for _, ts in kalman_data]
    kalman_filtered = [ts['kalman_filtered'] for _, ts in kalman_data]
    kalman_uncertainty = [ts['kalman_uncertainty'] for _, ts in kalman_data]
    
    # Chart 1: Kalman Filter Overlay
    ax1.scatter(all_dates, all_weights, c='lightblue', s=15, alpha=0.5, label='All Measurements')
    ax1.plot(kalman_dates, kalman_measured, 'b-', alpha=0.3, linewidth=1, label='Measured (with Kalman)')
    ax1.plot(kalman_dates, kalman_filtered, 'g-', linewidth=2, alpha=0.9, label='Kalman Filtered')
    
    # Add uncertainty bands
    upper_bound = [f + 2*u for f, u in zip(kalman_filtered, kalman_uncertainty)]
    lower_bound = [f - 2*u for f, u in zip(kalman_filtered, kalman_uncertainty)]
    ax1.fill_between(kalman_dates, lower_bound, upper_bound, 
                     alpha=0.2, color='green', label='±2σ Uncertainty')
    
    ax1.set_title('Kalman Filter Performance', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Weight (kg)', fontsize=10)
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Chart 2: Innovation (Prediction Error)
    innovations = [m - f for m, f in zip(kalman_measured, kalman_filtered)]
    ax2.scatter(kalman_dates, innovations, c='red', s=10, alpha=0.6)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax2.axhline(y=np.std(innovations)*2, color='red', linestyle='--', alpha=0.5)
    ax2.axhline(y=-np.std(innovations)*2, color='red', linestyle='--', alpha=0.5)
    ax2.set_title('Kalman Innovation (Prediction Error)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Innovation (kg)', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Chart 3: Uncertainty Evolution
    ax3.plot(kalman_dates, kalman_uncertainty, 'orange', linewidth=2, alpha=0.8)
    ax3.fill_between(kalman_dates, 0, kalman_uncertainty, alpha=0.3, color='orange')
    ax3.set_title('Kalman Uncertainty Over Time', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Uncertainty (kg)', fontsize=10)
    ax3.set_xlabel('Date', fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Chart 4: Difference Between Measured and Filtered
    differences = [abs(m - f) for m, f in zip(kalman_measured, kalman_filtered)]
    ax4.hist(differences, bins=20, color='purple', alpha=0.7, edgecolor='black')
    ax4.axvline(x=np.mean(differences), color='red', linestyle='--', 
               label=f'Mean: {np.mean(differences):.3f} kg')
    ax4.set_title('Distribution of |Measured - Filtered|', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Absolute Difference (kg)', fontsize=10)
    ax4.set_ylabel('Frequency', fontsize=10)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Format dates
    for ax in [ax1, ax2, ax3]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add Kalman summary stats
    if 'kalman_summary' in data:
        ks = data['kalman_summary']
        stats_text = f"Kalman Statistics:\n"
        stats_text += f"Final Weight: {ks.get('final_filtered_weight', 'N/A'):.1f} kg\n"
        stats_text += f"Mean Uncertainty: ±{ks.get('mean_uncertainty', 0):.3f} kg\n"
        stats_text += f"Outliers: {ks.get('kalman_outliers', 0)} ({ks.get('kalman_outlier_rate', 0)*100:.1f}%)\n"
        stats_text += f"Mean Innovation: {ks.get('mean_innovation', 0):.3f} kg"
        
        fig.text(0.98, 0.02, stats_text, fontsize=9,
                horizontalalignment='right', verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.suptitle(f'Kalman Filter Analysis - User {user_id[:8]}...', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_file = output_dir / f'kalman_{user_id}.png'
    plt.savefig(output_file, dpi=100, bbox_inches='tight')
    plt.close()
    logger.info(f"Created Kalman chart: {output_file.name}")

def create_source_chart(user_id: str, data: dict, output_dir: Path):
    """Create chart analyzing data sources and their quality."""
    
    time_series = data.get('time_series', [])
    if not time_series:
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
    
    # Extract data by source
    sources = {}
    for ts in time_series:
        source = ts.get('source', 'unknown')
        if source not in sources:
            sources[source] = {'dates': [], 'weights': [], 'confidences': []}
        sources[source]['dates'].append(parse_datetime(ts['date']))
        sources[source]['weights'].append(ts['weight'])
        sources[source]['confidences'].append(ts.get('confidence', 0.5))
    
    # Chart 1: Weight by Source
    colors = {
        'care-team-upload': 'green',
        'patient-device': 'blue', 
        'internal-questionnaire': 'purple',
        'patient-upload': 'orange',
        'unknown': 'gray'
    }
    
    for source, source_data in sources.items():
        color = colors.get(source, 'gray')
        ax1.scatter(source_data['dates'], source_data['weights'], 
                   c=color, s=20, alpha=0.6, label=f'{source} (n={len(source_data["dates"])})')
    
    ax1.set_title('Weight Measurements by Data Source', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Weight (kg)', fontsize=10)
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Chart 2: Source Distribution (Pie)
    source_counts = {s: len(d['dates']) for s, d in sources.items()}
    ax2.pie(source_counts.values(), labels=source_counts.keys(), autopct='%1.1f%%',
           colors=[colors.get(s, 'gray') for s in source_counts.keys()])
    ax2.set_title('Distribution of Data Sources', fontsize=12, fontweight='bold')
    
    # Chart 3: Confidence by Source
    source_names = []
    confidence_data = []
    for source, source_data in sources.items():
        source_names.append(source.replace('-', '\n'))
        confidence_data.append(source_data['confidences'])
    
    bp = ax3.boxplot(confidence_data, labels=source_names, patch_artist=True)
    for patch, source in zip(bp['boxes'], sources.keys()):
        patch.set_facecolor(colors.get(source, 'gray'))
        patch.set_alpha(0.6)
    
    ax3.set_title('Confidence Score Distribution by Source', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Confidence Score', fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Chart 4: Temporal Distribution of Sources
    # Create timeline showing when each source was used
    ax4.set_title('Timeline of Data Sources', fontsize=12, fontweight='bold')
    
    y_positions = {source: i for i, source in enumerate(sources.keys())}
    
    for source, source_data in sources.items():
        y_pos = y_positions[source]
        ax4.scatter(source_data['dates'], [y_pos]*len(source_data['dates']), 
                   c=colors.get(source, 'gray'), s=30, alpha=0.6, label=source)
    
    ax4.set_yticks(list(y_positions.values()))
    ax4.set_yticklabels(list(y_positions.keys()))
    ax4.set_ylabel('Data Source', fontsize=10)
    ax4.set_xlabel('Date', fontsize=10)
    ax4.grid(True, alpha=0.3, axis='x')
    
    # Format dates
    for ax in [ax1, ax4]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add source quality stats
    stats_text = "Source Statistics:\n"
    for source in sources:
        avg_conf = np.mean(sources[source]['confidences'])
        stats_text += f"{source}: {avg_conf:.2f} avg conf\n"
    
    fig.text(0.98, 0.02, stats_text, fontsize=8,
            horizontalalignment='right', verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.suptitle(f'Data Source Analysis - User {user_id[:8]}...', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_file = output_dir / f'sources_{user_id}.png'
    plt.savefig(output_file, dpi=100, bbox_inches='tight')
    plt.close()
    logger.info(f"Created sources chart: {output_file.name}")

def main():
    """Generate focused charts for users with Kalman filter data."""
    
    # Find the most recent baseline results file
    output_dir = Path('output')
    baseline_files = list(output_dir.glob('baseline_results_*.json'))
    
    if not baseline_files:
        logger.error("No baseline results found. Run: uv run python baseline_processor.py")
        return
    
    results_file = max(baseline_files, key=lambda p: p.stat().st_mtime)
    logger.info(f"Loading data from: {results_file.name}")
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    # Create output directory
    viz_dir = output_dir / 'kalman_charts'
    viz_dir.mkdir(exist_ok=True, parents=True)
    
    # Find users with Kalman filters and good data
    kalman_users = []
    for user_id, user_data in data.items():
        if (user_data.get('kalman_filter_initialized') and 
            user_data.get('total_readings', 0) >= 20 and
            'time_series' in user_data):
            
            # Check if time_series actually has Kalman data
            has_kalman = any(ts.get('kalman_filtered') is not None 
                           for ts in user_data['time_series'])
            if has_kalman:
                kalman_users.append((user_id, user_data))
    
    # Sort by number of readings
    kalman_users.sort(key=lambda x: x[1]['total_readings'], reverse=True)
    
    # Process top 5 users
    max_users = min(5, len(kalman_users))
    
    logger.info(f"\nGenerating charts for top {max_users} users with Kalman filters...")
    logger.info("Creating 3 charts per user: Baseline, Kalman, and Sources\n")
    
    for i, (user_id, user_data) in enumerate(kalman_users[:max_users]):
        logger.info(f"Processing user {i+1}/{max_users}: {user_id[:8]}... ({user_data['total_readings']} readings)")
        
        try:
            # Create all three chart types
            create_baseline_chart(user_id, user_data, viz_dir)
            create_kalman_chart(user_id, user_data, viz_dir)
            create_source_chart(user_id, user_data, viz_dir)
        except Exception as e:
            logger.error(f"  Error creating charts for user {user_id[:8]}...: {e}")
    
    logger.info(f"\n✅ Chart generation complete!")
    logger.info(f"📁 Output directory: {viz_dir}")
    logger.info(f"📊 Files created: {max_users * 3} charts ({max_users} users × 3 chart types)")
    logger.info("\nChart types:")
    logger.info("  • baseline_<user_id>.png - Baseline establishment and comparison")
    logger.info("  • kalman_<user_id>.png - Kalman filter performance (4 subplots)")
    logger.info("  • sources_<user_id>.png - Data source analysis and quality")

if __name__ == "__main__":
    main()