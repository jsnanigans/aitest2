"""
Kalman Filter Processing Evaluation Dashboard
Focuses on evaluating processor quality and Kalman filter performance
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from dateutil.relativedelta import relativedelta


def categorize_rejection(reason: str) -> str:
    """Categorize rejection reason into high-level category."""
    reason_lower = reason.lower()
    
    if "outside bounds" in reason_lower:
        return "Bounds"
    elif "extreme deviation" in reason_lower:
        return "Extreme"
    elif "session variance" in reason_lower or "different user" in reason_lower:
        return "Variance"
    elif "sustained" in reason_lower:
        return "Sustained"
    elif "daily fluctuation" in reason_lower:
        return "Daily"
    elif "meals" in reason_lower and "hydration" in reason_lower:
        return "Medium"
    elif ("hydration" in reason_lower or "bathroom" in reason_lower) and "meals" not in reason_lower:
        return "Short"
    elif "exceeds" in reason_lower and "limit" in reason_lower:
        return "Limit"
    else:
        return "Other"


def get_rejection_color_map():
    """Get consistent color mapping for rejection categories."""
    return {
        "Extreme": "#D32F2F",     # Red - most severe
        "Bounds": "#E91E63",       # Pink - out of bounds
        "Sustained": "#FF6F00",    # Dark Orange - long-term issue
        "Variance": "#FFA726",     # Orange - session variance
        "Daily": "#FFD54F",        # Yellow - daily fluctuation
        "Medium": "#66BB6A",       # Green - medium-term
        "Short": "#42A5F5",        # Blue - short-term
        "Limit": "#AB47BC",        # Purple - physiological limit
        "Other": "#9E9E9E"         # Gray - uncategorized
    }


def normalize_source_type(source: str) -> str:
    """Normalize source type string to a standard category."""
    if not source:
        return "unknown"
    
    source_lower = source.lower()
    
    if "patient-device" in source_lower or "device" in source_lower:
        return "device"
    elif "connectivehealth" in source_lower or "api" in source_lower or "https://" in source_lower:
        return "connected"
    elif "questionnaire" in source_lower or "internal-questionnaire" in source_lower:
        return "questionnaire"
    elif "patient-upload" in source_lower or "upload" in source_lower or "manual" in source_lower:
        return "manual"
    elif "scale" in source_lower:
        return "device"
    elif "test" in source_lower:
        return "test"
    else:
        return "other"


def get_source_style():
    """Get visual style configuration for different source types."""
    return {
        "device": {
            "marker": "o",
            "size": 60,
            "alpha": 0.9,
            "label": "Device",
            "color": "#2E7D32",  # Green - most reliable
            "priority": 1
        },
        "connected": {
            "marker": "s",
            "size": 55,
            "alpha": 0.85,
            "label": "Connected",
            "color": "#1976D2",  # Blue - API/connected
            "priority": 2
        },
        "questionnaire": {
            "marker": "^",
            "size": 50,
            "alpha": 0.75,
            "label": "Questionnaire",
            "color": "#7B1FA2",  # Purple - self-reported
            "priority": 3
        },
        "manual": {
            "marker": "D",
            "size": 45,
            "alpha": 0.7,
            "label": "Manual",
            "color": "#F57C00",  # Orange - manual entry
            "priority": 4
        },
        "test": {
            "marker": "p",
            "size": 40,
            "alpha": 0.6,
            "label": "Test",
            "color": "#616161",  # Gray - test data
            "priority": 5
        },
        "other": {
            "marker": ".",
            "size": 35,
            "alpha": 0.5,
            "label": "Other",
            "color": "#9E9E9E",  # Light gray - unknown
            "priority": 6
        },
        "unknown": {
            "marker": ".",
            "size": 35,
            "alpha": 0.5,
            "label": "Unknown",
            "color": "#BDBDBD",  # Very light gray
            "priority": 7
        }
    }


def cluster_rejections(rejected_results: List[dict], time_window_hours: float = 24) -> List[Dict]:
    """
    Cluster nearby rejections to avoid annotation overlap.
    Returns list of clusters with representative rejection and count.
    """
    if not rejected_results:
        return []
    
    clusters = []
    current_cluster = {
        'timestamp': rejected_results[0]['timestamp'],
        'reasons': defaultdict(int),
        'weights': [],
        'count': 0
    }
    
    for result in rejected_results:
        ts = result['timestamp']
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts.replace('Z', '+00:00'))
        
        if current_cluster['count'] == 0:
            current_cluster['timestamp'] = ts
            current_cluster['reasons'][result.get('reason', 'Unknown')] += 1
            current_cluster['weights'].append(result['raw_weight'])
            current_cluster['count'] = 1
        else:
            last_ts = current_cluster['timestamp']
            if isinstance(last_ts, str):
                last_ts = datetime.fromisoformat(last_ts.replace('Z', '+00:00'))
            
            time_diff = abs((ts - last_ts).total_seconds() / 3600)
            
            if time_diff <= time_window_hours:
                current_cluster['reasons'][result.get('reason', 'Unknown')] += 1
                current_cluster['weights'].append(result['raw_weight'])
                current_cluster['count'] += 1
                current_cluster['timestamp'] = ts
            else:
                clusters.append(current_cluster)
                current_cluster = {
                    'timestamp': ts,
                    'reasons': defaultdict(int),
                    'weights': [],
                    'count': 0
                }
                current_cluster['reasons'][result.get('reason', 'Unknown')] += 1
                current_cluster['weights'].append(result['raw_weight'])
                current_cluster['count'] = 1
    
    if current_cluster['count'] > 0:
        clusters.append(current_cluster)
    
    return clusters


def identify_interesting_rejections(
    rejected_results: List[dict], 
    clusters: List[Dict],
    max_annotations: int = 4
) -> List[Tuple[datetime, str, float]]:
    """
    Identify which rejections to annotate based on importance.
    Returns list of (timestamp, annotation_text, weight) tuples.
    """
    annotations = []
    
    seen_categories = set()
    
    for cluster in clusters[:max_annotations]:
        if cluster['count'] >= 3:
            most_common_reason = max(cluster['reasons'].items(), key=lambda x: x[1])
            category = categorize_rejection(most_common_reason[0])
            
            if cluster['count'] > 10:
                text = f"{cluster['count']}x"
            elif cluster['count'] > 5:
                text = f"{cluster['count']}x"
            else:
                text = category  # Just the single-word category
            
            avg_weight = np.mean(cluster['weights'])
            annotations.append((cluster['timestamp'], text, avg_weight))
        
        elif cluster['count'] == 1:
            reason = list(cluster['reasons'].keys())[0]
            category = categorize_rejection(reason)
            
            if category not in seen_categories and len(annotations) < max_annotations:
                # Just use the single-word category
                annotations.append((
                    cluster['timestamp'],
                    category,
                    cluster['weights'][0]
                ))
                seen_categories.add(category)
    
    return annotations


def add_rejection_annotations(
    ax, 
    rejected_results: List[dict], 
    rejected_timestamps: List[datetime],
    rejected_weights: List[float],
    date_range: Optional[Tuple[datetime, datetime]] = None
):
    """Add smart annotations for rejected measurements to a plot."""
    if not rejected_results:
        return
    
    filtered_results = []
    filtered_timestamps = []
    filtered_weights = []
    
    if date_range:
        start_date, end_date = date_range
        for r, ts, w in zip(rejected_results, rejected_timestamps, rejected_weights):
            if start_date <= ts <= end_date:
                filtered_results.append(r)
                filtered_timestamps.append(ts)
                filtered_weights.append(w)
    else:
        filtered_results = rejected_results
        filtered_timestamps = rejected_timestamps
        filtered_weights = rejected_weights
    
    if not filtered_results:
        return
    
    clusters = cluster_rejections(filtered_results, time_window_hours=48)  # Wider clustering window
    annotations = identify_interesting_rejections(filtered_results, clusters)
    
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    color_map = get_rejection_color_map()
    
    # Smart positioning to avoid overlaps
    positions_used = []
    
    for i, (timestamp, text, weight) in enumerate(annotations):
        if date_range and not (date_range[0] <= timestamp <= date_range[1]):
            continue
        
        # Find the category for this annotation to get the right color
        category = text  # For single rejections, text is the category
        if 'x' in text:  # For clusters, extract category from original data
            # Find rejection at this timestamp
            for r in filtered_results:
                r_ts = r['timestamp']
                if isinstance(r_ts, str):
                    r_ts = datetime.fromisoformat(r_ts.replace('Z', '+00:00'))
                if abs((r_ts - timestamp).total_seconds()) < 86400:  # Within a day
                    category = categorize_rejection(r.get('reason', 'Unknown'))
                    break
        
        annotation_color = color_map.get(category, '#9E9E9E')
        
        # Calculate base position
        base_y = 20
        
        # Check for overlaps and adjust
        for prev_ts, prev_y in positions_used:
            time_diff = abs((timestamp - prev_ts).total_seconds() / 86400)  # Days
            if time_diff < 7:  # If within a week
                base_y = max(base_y, prev_y + 25)
        
        # Alternate left/right for variety
        offset_x = -15 if i % 2 == 0 else 15
        offset_y = base_y
        
        # Cap maximum height
        if offset_y > 60:
            offset_y = 20
            offset_x = offset_x * 2  # Move further horizontally
        
        positions_used.append((timestamp, offset_y))
        
        ax.annotate(
            text,
            xy=(timestamp, weight),
            xytext=(offset_x, offset_y),
            textcoords='offset points',
            fontsize=10,
            color=annotation_color,
            bbox=dict(
                boxstyle='round,pad=0.4',
                facecolor='white',
                edgecolor=annotation_color,
                alpha=0.95,
                linewidth=1
            ),
            arrowprops=dict(
                arrowstyle='-',
                connectionstyle='arc3,rad=0.3',
                color=annotation_color,
                alpha=0.5,
                linewidth=1
            ),
            ha='center',
            va='bottom',
            fontweight='bold'
        )


def create_dashboard(user_id: str, results: list, output_dir: str, viz_config: dict):
    """
    Create a processor evaluation dashboard for a single user.

    Args:
        user_id: User identifier
        results: List of processed weight measurements (accepted and rejected)
        output_dir: Directory to save the dashboard
        viz_config: Visualization configuration dictionary
    """
    all_results = [r for r in results if r]
    valid_results = [r for r in all_results if r.get("accepted")]
    rejected_results = [r for r in all_results if not r.get("accepted")]

    if not all_results:
        return

    fig = plt.figure(figsize=(26, 14))
    fig.suptitle(
        f"Kalman Filter Processing Evaluation - User {user_id[:8]}",
        fontsize=18,
        fontweight="bold",
        y=0.98
    )
    
    from matplotlib.gridspec import GridSpec
    # New layout: Top row full width for main chart, bottom rows for other charts and stats
    gs = GridSpec(3, 5, figure=fig, height_ratios=[5, 2, 2], width_ratios=[1, 1, 1, 1, 1.2], 
                  hspace=0.35, wspace=0.35, top=0.95, bottom=0.05, left=0.05, right=0.98)

    def parse_timestamps(results):
        timestamps = [r["timestamp"] for r in results]
        if timestamps and isinstance(timestamps[0], str):
            timestamps = [
                (
                    datetime.fromisoformat(t.replace("Z", "+00:00"))
                    if isinstance(t, str)
                    else t
                )
                for t in timestamps
            ]
        return timestamps

    all_timestamps = parse_timestamps(all_results)
    valid_timestamps = parse_timestamps(valid_results) if valid_results else []
    rejected_timestamps = parse_timestamps(rejected_results) if rejected_results else []

    all_raw_weights = [r["raw_weight"] for r in all_results]
    valid_raw_weights = [r["raw_weight"] for r in valid_results] if valid_results else []
    rejected_raw_weights = [r["raw_weight"] for r in rejected_results] if rejected_results else []

    filtered_weights = [
        r.get("filtered_weight", r["raw_weight"]) for r in valid_results
    ] if valid_results else []

    # Calculate date range for cropped views - last N months of user data with padding
    cropped_months = viz_config.get('cropped_months', 12)  # Default to 12 months if not configured
    
    if all_timestamps:
        # Find the actual date range of the data
        min_date = min(all_timestamps)
        max_date = max(all_timestamps)
        
        # Calculate N months before the most recent data point
        crop_start = max_date - relativedelta(months=cropped_months)
        
        # If all data is within N months, show all with padding
        if min_date > crop_start:
            crop_start = min_date - relativedelta(days=7)  # 1 week padding before
        
        # Add slight padding at the end (1 week)
        crop_end = max_date + relativedelta(days=7)
    else:
        # Fallback if no data
        crop_start = datetime.now() - relativedelta(months=cropped_months)
        crop_end = datetime.now()

    # Filter data for cropped views
    def filter_by_date_range(timestamps, data, start_date, end_date):
        """Filter data based on date range."""
        if not timestamps:
            return [], []
        filtered_ts = []
        filtered_data = []
        for ts, d in zip(timestamps, data):
            if start_date <= ts <= end_date:
                filtered_ts.append(ts)
                filtered_data.append(d)
        return filtered_ts, filtered_data

    # Create cropped versions of data
    valid_ts_cropped, valid_raw_cropped = filter_by_date_range(valid_timestamps, valid_raw_weights, crop_start, crop_end)
    valid_ts_cropped2, filtered_cropped = filter_by_date_range(valid_timestamps, filtered_weights, crop_start, crop_end)
    rejected_ts_cropped, rejected_raw_cropped = filter_by_date_range(rejected_timestamps, rejected_raw_weights, crop_start, crop_end)

    # For results-based data, we need to filter the full results
    valid_results_cropped = []
    if valid_results:
        for r, ts in zip(valid_results, valid_timestamps):
            if crop_start <= ts <= crop_end:
                valid_results_cropped.append(r)

    # Main chart takes full width of top row (all 5 columns)
    ax1 = fig.add_subplot(gs[0, :])

    # No need to show raw data - we only plot accepted/rejected results

    if valid_results:
        ax1.plot(valid_timestamps, filtered_weights, '-', linewidth=3.5,
                color='#1565C0', label='Kalman Filtered', zorder=2)
        
        # Group valid results by source type for plotting
        source_styles = get_source_style()
        source_groups = defaultdict(lambda: {'timestamps': [], 'weights': []})
        
        for r, ts, w in zip(valid_results, valid_timestamps, valid_raw_weights):
            source = normalize_source_type(r.get('source', 'unknown'))
            source_groups[source]['timestamps'].append(ts)
            source_groups[source]['weights'].append(w)
        
        # Plot each source type with its specific style
        for source_type in sorted(source_groups.keys(), key=lambda x: source_styles.get(x, {}).get('priority', 999)):
            if source_groups[source_type]['timestamps']:
                style = source_styles.get(source_type, source_styles['other'])
                ax1.scatter(
                    source_groups[source_type]['timestamps'],
                    source_groups[source_type]['weights'],
                    marker=style['marker'],
                    s=style['size'],
                    alpha=style['alpha'],
                    color=style['color'],
                    label=f"{style['label']} ({len(source_groups[source_type]['timestamps'])})",
                    zorder=5,
                    edgecolors='white',
                    linewidth=0.7
                )

        uncertainty_band = []
        for r in valid_results:
            if "normalized_innovation" in r:
                ni = r["normalized_innovation"]
                uncertainty = 0.5 + ni * 0.5
                uncertainty_band.append(uncertainty)
            else:
                uncertainty_band.append(0.5)

        if uncertainty_band:
            upper_band = [f + u for f, u in zip(filtered_weights, uncertainty_band)]
            lower_band = [f - u for f, u in zip(filtered_weights, uncertainty_band)]
            ax1.fill_between(valid_timestamps, lower_band, upper_band,
                            alpha=0.25, color='#64B5F6', label='Uncertainty')

    if rejected_results:
        # Color-code rejections by category and show source type
        color_map = get_rejection_color_map()
        source_styles = get_source_style()
        
        # Group rejections by category AND source for more detailed plotting
        categories_data = defaultdict(lambda: {'timestamps': [], 'weights': [], 'sources': []})
        
        for r, ts, w in zip(rejected_results, rejected_timestamps, rejected_raw_weights):
            category = categorize_rejection(r.get('reason', 'Unknown'))
            source = normalize_source_type(r.get('source', 'unknown'))
            categories_data[category]['timestamps'].append(ts)
            categories_data[category]['weights'].append(w)
            categories_data[category]['sources'].append(source)
        
        # Plot each category with its color, using X marker for all rejections
        for category, data in categories_data.items():
            if data['timestamps']:  # Only plot if there's data
                color = color_map.get(category, '#9E9E9E')
                # Use different marker sizes based on source reliability
                sizes = []
                for source in data['sources']:
                    source_style = source_styles.get(source, source_styles['other'])
                    # Scale rejection marker size based on source priority (more reliable = bigger X)
                    base_size = 100
                    size_modifier = 1.2 - (source_style['priority'] * 0.1)
                    sizes.append(base_size * size_modifier)
                
                ax1.scatter(data['timestamps'], data['weights'],
                           marker='x', color=color, s=sizes, alpha=0.9, linewidth=3,
                           label=f'{category} ({len(data["timestamps"])})', zorder=6)
        
        add_rejection_annotations(ax1, rejected_results, rejected_timestamps, rejected_raw_weights)

    if filtered_weights:
        baseline = np.median(filtered_weights[: min(10, len(filtered_weights))])
        ax1.axhline(baseline, color='#F57C00', linestyle='--', alpha=0.6, linewidth=2,
                   label=f'Baseline: {baseline:.1f}kg')
    
    # Add reset indicators
    reset_points = []
    for r in all_results:
        if r.get('was_reset', False):
            reset_points.append({
                'timestamp': r['timestamp'] if isinstance(r['timestamp'], datetime) else datetime.fromisoformat(r['timestamp'].replace('Z', '+00:00')),
                'gap_days': r.get('gap_days', 0),
                'weight': r['raw_weight']
            })
    
    if reset_points:
        for reset in reset_points:
            ax1.axvline(reset['timestamp'], color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
            ax1.annotate(f"{int(reset['gap_days'])}d gap",
                        xy=(reset['timestamp'], reset['weight']),
                        xytext=(5, 10), textcoords='offset points',
                        fontsize=9, color='gray', alpha=0.8)
        
        # Add to legend
        ax1.plot([], [], color='gray', linestyle='--', alpha=0.5, linewidth=1.5, label='State Reset')

    ax1.set_xlabel("Date", fontsize=14)
    ax1.set_ylabel("Weight (kg)", fontsize=14)
    
    # Show recent data in main chart, with option to see full range
    if len(all_timestamps) > 100:  # If we have substantial history
        ax1.set_xlim([crop_start, crop_end])
        ax1.set_title("Kalman Filter Output vs Raw Data", fontsize=16, fontweight='bold', pad=12)
    else:
        ax1.set_title("Kalman Filter Output vs Raw Data", fontsize=16, fontweight='bold', pad=12)
    
    # Sort legend entries by count (most common first)
    handles, labels = ax1.get_legend_handles_labels()
    if len(handles) > 3:  # If we have rejection categories
        # Extract counts from labels and sort
        try:
            sorted_pairs = sorted(zip(handles[3:], labels[3:]), 
                                key=lambda x: int(x[1].split('(')[1].split(')')[0]) if '(' in x[1] else 0, 
                                reverse=True)
            if sorted_pairs:
                sorted_handles, sorted_labels = zip(*sorted_pairs)
                handles = handles[:3] + list(sorted_handles)
                labels = labels[:3] + list(sorted_labels)
        except:
            pass  # If sorting fails, use original order
    
    ax1.legend(handles, labels, loc="best", ncol=min(len(handles), 6), fontsize=10, framealpha=0.95)
    ax1.grid(True, alpha=0.3, linewidth=0.5)
    ax1.grid(True, which='minor', alpha=0.15, linewidth=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Statistics Panel (right side of middle row)
    ax_stats = fig.add_subplot(gs[1:, 4])
    ax_stats.axis('off')
    
    # Build statistics content with better formatting
    stats_sections = []
    
    # Processing Overview
    total_measurements = len(all_results)
    accepted_pct = 100*len(valid_results)/len(all_results) if all_results else 0
    rejected_pct = 100*len(rejected_results)/len(all_results) if all_results else 0
    
    stats_sections.append([
        "PROCESSING OVERVIEW",
        f"Total Measurements: {total_measurements:,}",
        f"Accepted: {len(valid_results):,} ({accepted_pct:.1f}%)",
        f"Rejected: {len(rejected_results):,} ({rejected_pct:.1f}%)",
        ""
    ])
    
    # Current Status
    if valid_results and filtered_weights:
        current_weight = filtered_weights[-1]
        baseline = np.median(filtered_weights[:min(10, len(filtered_weights))])
        weight_change = current_weight - baseline
        
        all_trends = [r.get("trend", 0) for r in valid_results]
        current_trend = all_trends[-1] * 7 if all_trends else 0
        
        stats_sections.append([
            "CURRENT STATUS",
            f"Weight: {current_weight:.1f} kg",
            f"Baseline: {baseline:.1f} kg",
            f"Change: {weight_change:+.1f} kg",
            f"Trend: {current_trend:+.2f} kg/week",
            ""
        ])
    
    # Filter Performance
    if valid_results:
        all_innovations = [r.get('innovation', 0) for r in valid_results]
        all_normalized_innovations = [r.get('normalized_innovation', 0) for r in valid_results]
        all_confidences = [r.get("confidence", 0.5) for r in valid_results]
        
        stats_sections.append([
            "FILTER PERFORMANCE",
            f"Mean Innovation: {np.mean(all_innovations):.3f} kg",
            f"Std Innovation: {np.std(all_innovations):.3f} kg",
            f"Mean Norm. Innov: {np.mean(all_normalized_innovations):.2f}σ",
            f"Max Norm. Innov: {max(all_normalized_innovations):.2f}σ",
            f"Mean Confidence: {np.mean(all_confidences):.2f}",
            ""
        ])
    
    # Weight Statistics
    if filtered_weights:
        stats_sections.append([
            "WEIGHT STATISTICS",
            f"Mean: {np.mean(filtered_weights):.1f} kg",
            f"Std Dev: {np.std(filtered_weights):.2f} kg",
            f"Min: {min(filtered_weights):.1f} kg",
            f"Max: {max(filtered_weights):.1f} kg",
            f"Range: {max(filtered_weights) - min(filtered_weights):.1f} kg",
            ""
        ])
    
    # Source Distribution Analysis
    source_styles = get_source_style()
    source_counts = defaultdict(lambda: {'total': 0, 'accepted': 0, 'rejected': 0})
    
    for r in all_results:
        source = normalize_source_type(r.get('source', 'unknown'))
        source_counts[source]['total'] += 1
        if r.get('accepted'):
            source_counts[source]['accepted'] += 1
        else:
            source_counts[source]['rejected'] += 1
    
    if source_counts:
        source_lines = ["SOURCE ANALYSIS"]
        # Sort by priority (most reliable first)
        sorted_sources = sorted(
            source_counts.items(),
            key=lambda x: source_styles.get(x[0], {}).get('priority', 999)
        )
        
        for source, counts in sorted_sources[:4]:  # Top 4 sources
            style = source_styles.get(source, source_styles['other'])
            accept_rate = 100 * counts['accepted'] / counts['total'] if counts['total'] > 0 else 0
            source_lines.append(
                f"{style['label']:12s}: {counts['total']:3d} ({accept_rate:.0f}% acc)"
            )
        source_lines.append("")
        stats_sections.append(source_lines)
    
    # Top Rejection Reasons
    if rejected_results:
        rejection_categories = defaultdict(int)
        for r in rejected_results:
            reason = r.get('reason', 'Unknown')
            category = categorize_rejection(reason)
            rejection_categories[category] += 1
        
        top_categories = sorted(rejection_categories.items(), key=lambda x: x[1], reverse=True)[:4]
        
        rejection_lines = ["REJECTION ANALYSIS"]
        for category, count in top_categories:
            percentage = 100 * count / len(rejected_results)
            if len(category) > 20:
                category = category[:17] + "..."
            rejection_lines.append(f"{category}: {count} ({percentage:.0f}%)")
        rejection_lines.append("")
        stats_sections.append(rejection_lines)
    
    # Render all sections
    y_position = 0.98
    for section in stats_sections:
        for i, line in enumerate(section):
            if i == 0:  # Section header
                ax_stats.text(0.05, y_position, line, transform=ax_stats.transAxes,
                            fontsize=11, fontweight='bold', color='#1565C0',
                            verticalalignment='top')
                y_position -= 0.04
            elif line:  # Content lines
                ax_stats.text(0.05, y_position, line, transform=ax_stats.transAxes,
                            fontsize=10, verticalalignment='top', fontfamily='monospace')
                y_position -= 0.035
            else:  # Empty line for spacing
                y_position -= 0.02
    
    # Add a subtle border around stats panel
    from matplotlib.patches import Rectangle
    rect = Rectangle((0.02, 0.02), 0.96, 0.96, transform=ax_stats.transAxes,
                    linewidth=1, edgecolor='#E0E0E0', facecolor='none')
    ax_stats.add_patch(rect)

    if valid_results_cropped:
        innovations_cropped = [r.get("innovation", 0) for r in valid_results_cropped]
        normalized_innovations_cropped = [r.get("normalized_innovation", 0) for r in valid_results_cropped]

        # Innovation chart in middle row, first column
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(valid_ts_cropped, innovations_cropped, 'o-', markersize=6, alpha=0.8, color='#0288D1', linewidth=2)
        ax2.axhline(0, color='black', linestyle='-', alpha=0.5, linewidth=1.5)
        ax2.fill_between(valid_ts_cropped, 0, innovations_cropped, alpha=0.4, color='#81D4FA')
        
        # Add reset indicators to innovation plot
        if reset_points:
            for reset in reset_points:
                if crop_start <= reset['timestamp'] <= crop_end:
                    ax2.axvline(reset['timestamp'], color='gray', linestyle='--', alpha=0.3, linewidth=1)
        
        ax2.set_xlabel("Date", fontsize=11)
        ax2.set_ylabel("Innovation (kg)", fontsize=11)
        ax2.set_title("Kalman Innovation\n(Measurement - Prediction)", fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax2.set_xlim([crop_start, crop_end])

        # Normalized Innovation in middle row, second column
        ax3 = fig.add_subplot(gs[1, 1])
        colors = ['#2E7D32' if ni <= 2 else '#FFA726' if ni <= 3 else '#D32F2F'
                 for ni in normalized_innovations_cropped]
        ax3.scatter(valid_ts_cropped, normalized_innovations_cropped, c=colors, alpha=0.8, s=50, edgecolors='white', linewidth=0.7)
        ax3.axhline(2, color='#2E7D32', linestyle='--', alpha=0.6, linewidth=2, label='Good (≤2σ)')
        ax3.axhline(3, color='#FF6F00', linestyle='--', alpha=0.6, linewidth=2, label='Warning (3σ)')
        ax3.axhline(5, color='#D32F2F', linestyle='--', alpha=0.6, linewidth=2, label='Extreme (5σ)')
        ax3.set_xlabel("Date", fontsize=11)
        ax3.set_ylabel("Normalized Innovation (σ)", fontsize=11)
        ax3.set_title("Normalized Innovation\n(Statistical Significance)", fontsize=12, fontweight='bold')
        ax3.legend(loc='best', fontsize=9)
        ax3.grid(True, alpha=0.3)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax3.set_xlim([crop_start, crop_end])

        confidences_cropped = [r.get("confidence", 0.5) for r in valid_results_cropped]
        # Confidence in middle row, third column
        ax4 = fig.add_subplot(gs[1, 2])
        ax4.plot(valid_ts_cropped, confidences_cropped, 'o-', markersize=6, alpha=0.8, color='#6A1B9A', linewidth=2)
        ax4.fill_between(valid_ts_cropped, 0, confidences_cropped, alpha=0.4, color='#CE93D8')
        ax4.axhline(0.95, color='#2E7D32', linestyle='--', alpha=0.6, linewidth=2, label='High (>0.95)')
        ax4.axhline(0.7, color='#FF6F00', linestyle='--', alpha=0.6, linewidth=2, label='Medium (>0.7)')
        ax4.axhline(0.5, color='#D32F2F', linestyle='--', alpha=0.6, linewidth=2, label='Low (<0.5)')
        ax4.set_xlabel("Date", fontsize=11)
        ax4.set_ylabel("Confidence", fontsize=11)
        ax4.set_title("Measurement Confidence\n(Based on Innovation)", fontsize=12, fontweight='bold')
        ax4.set_ylim([0, 1.05])
        ax4.legend(loc='best', fontsize=9)
        ax4.grid(True, alpha=0.3)
        ax4.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax4.set_xlim([crop_start, crop_end])

        # Trend component removed - not needed

    # Daily Change Distribution moved to middle row, fourth column
    ax8 = fig.add_subplot(gs[1, 3])
    if valid_results_cropped and len(filtered_cropped) > 2:
        diffs = np.diff(filtered_cropped)
        time_diffs = [(valid_ts_cropped[i+1] - valid_ts_cropped[i]).days
                     for i in range(len(valid_ts_cropped)-1)]
        daily_changes = [d/td if td > 0 else 0 for d, td in zip(diffs, time_diffs)]
        ax8.hist(daily_changes, bins=20, edgecolor='#1B5E20', alpha=0.8, color='#4CAF50', linewidth=1.5)
        ax8.axvline(0, color='black', linestyle='-', linewidth=2.5, alpha=0.7)
        ax8.set_xlabel("Daily Change (kg/day)", fontsize=11)
        ax8.set_ylabel("Frequency", fontsize=11)
        ax8.set_title("Daily Change Distribution\n(Filtered Weights)", fontsize=12, fontweight='bold')
        ax8.grid(True, alpha=0.3)

    # Rejection categories chart in bottom row (first 2 columns)
    ax9 = fig.add_subplot(gs[2, :2])
    
    # Create rejection categories bar chart
    if rejected_results:
        rejection_categories = defaultdict(int)
        for r in rejected_results:
            reason = r.get('reason', 'Unknown')
            category = categorize_rejection(reason)
            rejection_categories[category] += 1
        
        # Sort and limit to top categories
        sorted_categories = sorted(rejection_categories.items(), key=lambda x: x[1], reverse=True)[:6]
        
        if sorted_categories:
            categories = []
            counts = []
            for cat, count in sorted_categories:
                # Categories are already single words, no need to shorten
                categories.append(cat)
                counts.append(count)
            
            # Create horizontal bar chart with category-specific colors
            y_pos = np.arange(len(categories))
            color_map = get_rejection_color_map()
            colors = [color_map.get(cat, '#9E9E9E') for cat in categories]
            
            bars = ax9.barh(y_pos, counts, color=colors, alpha=0.8, edgecolor='#424242', linewidth=1.5)
            ax9.set_yticks(y_pos)
            ax9.set_yticklabels(categories, fontsize=10)
            ax9.set_xlabel('Count', fontsize=11)
            ax9.set_title(f'Rejection Categories (n={len(rejected_results)})', fontsize=12, fontweight='bold')
            ax9.grid(True, alpha=0.3, axis='x')
            
            # Add count labels on bars
            for i, (bar, count) in enumerate(zip(bars, counts)):
                width = bar.get_width()
                pct = 100 * count / len(rejected_results)
                ax9.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
                        f'{count} ({pct:.0f}%)', 
                        ha='left', va='center', fontsize=9)
    else:
        ax9.text(0.5, 0.5, 'No Rejections', transform=ax9.transAxes,
                fontsize=14, ha='center', va='center', color='#2E7D32', fontweight='bold')
        ax9.set_title('Rejection Categories', fontsize=12, fontweight='bold')
        ax9.axis('off')

    # Source Distribution chart in bottom row (columns 2-4)
    ax10 = fig.add_subplot(gs[2, 2:4])
    
    # Create source distribution stacked bar chart
    source_styles = get_source_style()
    source_data = defaultdict(lambda: {'accepted': 0, 'rejected': 0})
    
    for r in all_results:
        source = normalize_source_type(r.get('source', 'unknown'))
        if r.get('accepted'):
            source_data[source]['accepted'] += 1
        else:
            source_data[source]['rejected'] += 1
    
    if source_data:
        # Sort by total count
        sorted_sources = sorted(
            source_data.items(),
            key=lambda x: x[1]['accepted'] + x[1]['rejected'],
            reverse=True
        )[:5]  # Top 5 sources
        
        source_names = []
        accepted_counts = []
        rejected_counts = []
        
        for source, counts in sorted_sources:
            style = source_styles.get(source, source_styles['other'])
            source_names.append(style['label'])
            accepted_counts.append(counts['accepted'])
            rejected_counts.append(counts['rejected'])
        
        x_pos = np.arange(len(source_names))
        
        # Create stacked bar chart
        bars1 = ax10.bar(x_pos, accepted_counts, color='#2E7D32', alpha=0.8, label='Accepted')
        bars2 = ax10.bar(x_pos, rejected_counts, bottom=accepted_counts, color='#D32F2F', alpha=0.8, label='Rejected')
        
        ax10.set_xticks(x_pos)
        ax10.set_xticklabels(source_names, fontsize=10)
        ax10.set_ylabel('Count', fontsize=11)
        ax10.set_title('Measurements by Source Type', fontsize=12, fontweight='bold')
        ax10.legend(loc='upper right', fontsize=9)
        ax10.grid(True, alpha=0.3, axis='y')
        
        # Add percentage labels on bars
        for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
            total = accepted_counts[i] + rejected_counts[i]
            if total > 0:
                accept_pct = 100 * accepted_counts[i] / total
                # Add count on top of bar
                ax10.text(bar1.get_x() + bar1.get_width()/2, total + 0.5,
                         f'{total}\n({accept_pct:.0f}%)',
                         ha='center', va='bottom', fontsize=9)
    else:
        ax10.text(0.5, 0.5, 'No Source Data', transform=ax10.transAxes,
                 fontsize=14, ha='center', va='center', color='#757575')
        ax10.set_title('Measurements by Source Type', fontsize=12, fontweight='bold')
        ax10.axis('off')

    # Don't use tight_layout with complex GridSpec - we set the spacing manually
    # This avoids the warning about incompatible layouts

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    filename = output_path / f"{user_id}.png"
    plt.savefig(filename, dpi=viz_config["dashboard_dpi"], bbox_inches="tight")
    plt.close()

    return str(filename)

