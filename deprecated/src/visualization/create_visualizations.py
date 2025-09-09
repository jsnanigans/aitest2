#!/usr/bin/env python3
"""
Simple visualization for weight tracking data with baseline.
Creates clean graphs showing weight readings over time with baseline reference.
"""

import json
import os
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.error("matplotlib not installed. Run: uv pip install matplotlib")


def load_results(json_file: str):
    """Load results from JSON file."""
    with open(json_file, 'r') as f:
        return json.load(f)


def parse_datetime(dt_string):
    """Parse datetime string to datetime object."""
    try:
        return datetime.fromisoformat(dt_string.replace('Z', '+00:00'))
    except:
        try:
            return datetime.strptime(dt_string, '%Y-%m-%d %H:%M:%S')
        except:
            return datetime.strptime(dt_string.split('.')[0], '%Y-%m-%dT%H:%M:%S')


def create_user_graph(user_id: str, data: dict, output_dir: Path):
    """Create a simple weight graph for a single user with Kalman filter overlay."""
    if not HAS_MATPLOTLIB:
        return

    # Check if we're using the new format with time_series
    if 'time_series' in data:
        time_series = data['time_series']
        if not time_series:
            logger.warning(f"No time series data for user {user_id}")
            return
        
        dates = [parse_datetime(ts['date']) for ts in time_series]
        weights = [ts['weight'] for ts in time_series]
        confidences = [ts.get('confidence', 0.5) for ts in time_series]
        sources = [ts.get('source', 'unknown') for ts in time_series]
        
        # Extract Kalman data if available
        kalman_filtered = [ts.get('kalman_filtered') for ts in time_series]
        kalman_uncertainty = [ts.get('kalman_uncertainty') for ts in time_series]
        has_kalman = any(kf is not None for kf in kalman_filtered)
    else:
        # Old format fallback
        readings = data.get('readings', [])
        if not readings:
            logger.warning(f"No readings for user {user_id}")
            return
        
        dates = []
        weights = []
        confidence_scores = data.get('readings_confidence_scores', {})
        confidences = []
        sources = []
        
        for reading in readings:
            dates.append(parse_datetime(reading['effectivDateTime']))
            weights.append(reading['weight'])
            reading_id = reading.get('id', '')
            confidences.append(confidence_scores.get(reading_id, 0.5))
            sources.append(reading.get('source', 'unknown'))
        
        has_kalman = False
        kalman_filtered = None
        kalman_uncertainty = None

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Define markers and colors for each source type
    source_styles = {
        'care-team-upload': {'marker': '^', 'color': 'darkgreen', 'label': 'Care Team Upload'},
        'patient-device': {'marker': 's', 'color': 'darkblue', 'label': 'Patient Device'},
        'internal-questionnaire': {'marker': 'D', 'color': 'purple', 'label': 'Internal Questionnaire'},
        'patient-upload': {'marker': 'v', 'color': 'darkorange', 'label': 'Patient Upload'},
        'https://connectivehealth.io': {'marker': 'p', 'color': 'teal', 'label': 'Connective Health'},
        'https://api.iglucose.com': {'marker': 'h', 'color': 'maroon', 'label': 'iGlucose API'},
        'unknown': {'marker': 'o', 'color': 'gray', 'label': 'Unknown Source'}
    }

    # Plot weight readings with source-specific markers and confidence-based transparency
    for i, (date, weight, confidence, source) in enumerate(zip(dates, weights, confidences, sources)):
        # Get source style, with fallback for unknown sources
        style = source_styles.get(source, source_styles['unknown'])
        
        # Adjust alpha based on confidence
        if confidence >= 0.75:
            alpha = 0.9
            size = 60
        elif confidence >= 0.5:
            alpha = 0.6
            size = 50
        else:
            alpha = 0.4
            size = 40
        
        ax.scatter(date, weight, 
                  marker=style['marker'], 
                  c=style['color'], 
                  s=size, 
                  alpha=alpha,
                  edgecolors='white', 
                  linewidths=0.5)

    # Connect measured points with a thin line
    ax.plot(dates, weights, 'b-', alpha=0.2, linewidth=0.5, label='Measured')
    
    # Add Kalman filtered line if available
    if has_kalman and kalman_filtered is not None and kalman_uncertainty is not None:
        # Filter out None values for plotting
        kalman_dates = []
        kalman_values = []
        kalman_upper = []
        kalman_lower = []
        
        for i, (d, kf, ku) in enumerate(zip(dates, kalman_filtered, kalman_uncertainty)):
            if kf is not None and ku is not None:
                kalman_dates.append(d)
                kalman_values.append(kf)
                kalman_upper.append(kf + 2 * ku)  # 2-sigma confidence band
                kalman_lower.append(kf - 2 * ku)
        
        if kalman_values:
            # Plot Kalman filtered line
            ax.plot(kalman_dates, kalman_values, 'g-', linewidth=2, alpha=0.8, label='Kalman Filtered')
            
            # Add uncertainty band
            ax.fill_between(kalman_dates, kalman_lower, kalman_upper, 
                           alpha=0.15, color='green', label='Kalman ±2σ')

    # Add baseline if established
    if data.get('baseline_established') and data.get('baseline_weight'):
        baseline = data['baseline_weight']
        ax.axhline(y=baseline, color='purple', linestyle='--', linewidth=2,
                   label=f'Baseline: {baseline:.1f} kg', alpha=0.8)

        # Add ±3% confidence band around baseline
        band_width = baseline * 0.03
        ax.fill_between(dates,
                        [baseline - band_width] * len(dates),
                        [baseline + band_width] * len(dates),
                        alpha=0.1, color='purple', label='±3% range')

    # Formatting
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Weight (kg)', fontsize=12)
    ax.set_title(f'Weight Tracking - User {user_id[:8]}...', fontsize=14, fontweight='bold')

    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Add grid
    ax.grid(True, alpha=0.3, linestyle=':')
    
    # Create legend entries for source types
    unique_sources = list(set(sources))
    for source in unique_sources:
        style = source_styles.get(source, source_styles['unknown'])
        ax.scatter([], [], 
                  marker=style['marker'], 
                  c=style['color'], 
                  s=60, 
                  alpha=0.9,
                  edgecolors='white', 
                  linewidths=0.5, 
                  label=style['label'])

    # Add legend
    ax.legend(loc='best', framealpha=0.9)

    # Add statistics text
    total_readings = len(weights) if weights else 0
    stats_text = f"Total readings: {total_readings}"
    
    if data.get('baseline_weight'):
        stats_text += f"\nBaseline: {data['baseline_weight']:.1f} kg"
        if data.get('baseline_readings_count'):
            stats_text += f" (from {data['baseline_readings_count']} readings)"
    elif data.get('baseline_established'):
        # Old format
        stats_text += f"\nBaseline: {data.get('baseline_weight', 'N/A'):.1f} kg"
    else:
        stats_text += "\nBaseline: Not established"
    
    # Add Kalman filter stats if available
    if data.get('kalman_summary'):
        ks = data['kalman_summary']
        if ks.get('final_filtered_weight'):
            stats_text += f"\nKalman filtered: {ks['final_filtered_weight']:.1f} kg"
        if ks.get('mean_uncertainty'):
            stats_text += f" (±{ks['mean_uncertainty']:.2f} kg)"

    # Place stats in top left
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Add confidence legend
    legend_text = "Marker Size/Opacity:\n● Large/Solid: High conf (≥0.75)\n● Medium: Med conf (0.5-0.75)\n● Small/Faint: Low conf (<0.5)"
    ax.text(0.98, 0.02, legend_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Adjust layout and save
    plt.tight_layout()
    output_file = output_dir / f'weight_graph_{user_id}.png'
    plt.savefig(output_file, dpi=100, bbox_inches='tight')
    plt.close()

    logger.info(f"Created graph for user {user_id[:8]}...")


def create_summary_graph(all_data: dict, output_dir: Path):
    """Create a summary graph showing baseline establishment success."""
    if not HAS_MATPLOTLIB:
        return

    # Count users with and without baselines
    with_baseline = sum(1 for d in all_data.values() if d.get('baseline_established'))
    without_baseline = len(all_data) - with_baseline

    # Create pie chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Pie chart for baseline status
    sizes = [with_baseline, without_baseline]
    labels = [f'With Baseline\n({with_baseline} users)',
              f'No Baseline\n({without_baseline} users)']
    colors = ['#2ecc71', '#e74c3c']
    explode = (0.05, 0)

    ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.set_title('Baseline Establishment Status', fontsize=14, fontweight='bold')

    # Bar chart for reading counts
    reading_counts = []
    baseline_counts = []

    for data in all_data.values():
        if data.get('baseline_established'):
            reading_counts.append(len(data.get('readings', [])))
            baseline_counts.append(data.get('baseline_count', 0))

    if reading_counts:
        ax2.hist(reading_counts, bins=20, alpha=0.7, color='blue', label='Total readings', edgecolor='black')
        ax2.hist(baseline_counts, bins=20, alpha=0.7, color='purple', label='Unique readings for baseline', edgecolor='black')
        ax2.set_xlabel('Number of Readings')
        ax2.set_ylabel('Number of Users')
        ax2.set_title('Reading Distribution (Users with Baseline)', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.suptitle(f'Weight Tracking Summary - {len(all_data)} Users', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_file = output_dir / 'summary.png'
    plt.savefig(output_file, dpi=100, bbox_inches='tight')
    plt.close()

    logger.info(f"Created summary graph: {output_file}")


def main():
    """Main execution function."""
    # Check for matplotlib
    if not HAS_MATPLOTLIB:
        return

    # Look for the most recent baseline results file (which includes Kalman data)
    output_dir = Path('output')
    baseline_files = list(output_dir.glob('baseline_results_*.json'))
    
    if baseline_files:
        # Use the most recent baseline file
        results_file = max(baseline_files, key=lambda p: p.stat().st_mtime)
        logger.info(f"Using baseline results file: {results_file.name}")
    else:
        # Fall back to test results
        results_file = Path('output/results_test.json')
        if not results_file.exists():
            logger.error("No results file found")
            logger.info("Please run: uv run python baseline_processor.py first")
            return

    # Load data
    logger.info(f"Loading data from {results_file}")
    all_data = load_results(str(results_file))
    logger.info(f"Loaded data for {len(all_data)} users")

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # viz_dir = Path('output') / f'graphs_{timestamp}'
    viz_dir = Path('output') / f'graphs_debug'
    viz_dir.mkdir(exist_ok=True, parents=True)

    # Create summary graph
    logger.info("\nCreating summary graph...")
    create_summary_graph(all_data, viz_dir)

    # Create individual graphs - prioritize users with Kalman filters
    users_with_kalman = [(uid, data) for uid, data in all_data.items()
                         if data.get('kalman_filter_initialized')]
    
    # Also get users with baseline but no Kalman
    users_with_baseline_only = [(uid, data) for uid, data in all_data.items()
                                if data.get('baseline_weight') and not data.get('kalman_filter_initialized')]
    
    # Combine lists - Kalman users first
    users_to_graph = users_with_kalman + users_with_baseline_only
    
    # Sort by number of readings (or time_series entries)
    users_to_graph.sort(key=lambda x: len(x[1].get('time_series', x[1].get('readings', []))), reverse=True)

    # Create graphs for top users
    max_graphs = min(20, len(users_to_graph))
    if users_with_kalman:
        logger.info(f"\nCreating graphs for top {max_graphs} users ({len(users_with_kalman)} with Kalman filters)...")
    else:
        logger.info(f"\nCreating graphs for top {max_graphs} users with baselines...")

    for i, (user_id, user_data) in enumerate(users_to_graph[:max_graphs]):
        readings_count = user_data.get('total_readings', len(user_data.get('time_series', user_data.get('readings', []))))
        has_kalman = " [Kalman]" if user_data.get('kalman_filter_initialized') else ""
        logger.info(f"  [{i+1}/{max_graphs}] User {user_id[:8]}... ({readings_count} readings{has_kalman})")
        try:
            create_user_graph(user_id, user_data, viz_dir)
        except Exception as e:
            logger.warning(f"    Failed to create graph: {e}")

    logger.info(f"\n✓ Visualization complete!")
    logger.info(f"Output directory: {viz_dir}")
    logger.info(f"Files created:")
    logger.info(f"  - summary.png")
    logger.info(f"  - weight_graph_<user_id>.png (for top {max_graphs} users)")


if __name__ == "__main__":
    main()
