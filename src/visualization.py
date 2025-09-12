"""
Kalman Filter Processing Evaluation Dashboard
Focuses on evaluating processor quality and Kalman filter performance
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import colorsys

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from dateutil.relativedelta import relativedelta


def categorize_rejection(reason: str) -> str:
    """Categorize rejection reason into clear, high-level category."""
    reason_lower = reason.lower()

    # BMI and unit conversion categories
    if "bmi" in reason_lower:
        return "BMI Value"
    elif "unit" in reason_lower or "pound" in reason_lower or "conversion" in reason_lower:
        return "Unit Convert"
    elif "physiological" in reason_lower:
        return "Physio Limit"
    # Original categories
    elif "outside bounds" in reason_lower:
        return "Out of Bounds"
    elif "extreme deviation" in reason_lower:
        return "Extreme Dev"
    elif "session variance" in reason_lower or "different user" in reason_lower:
        return "High Variance"
    elif "sustained" in reason_lower:
        return "Sustained"
    elif "daily fluctuation" in reason_lower:
        return "Daily Flux"
    elif "meals" in reason_lower and "hydration" in reason_lower:
        return "Medium Term"
    elif ("hydration" in reason_lower or "bathroom" in reason_lower) and "meals" not in reason_lower:
        return "Short Term"
    elif "exceeds" in reason_lower and "limit" in reason_lower:
        return "Limit Exceed"
    else:
        return "Other"


def get_rejection_color_map():
    """Red-orange-black gradient for rejection categories based on severity."""
    return {
        # Most Severe (Dark Red to Black)
        "BMI Value": "#1A0000",        # Near black - BMI detected
        "Unit Convert": "#330000",     # Very dark red - unit conversion
        "Physio Limit": "#4D0000",     # Dark red - physiological bounds
        
        # High Severity (Dark Red)
        "Extreme Dev": "#660000",      # Dark red - extreme deviation
        "Out of Bounds": "#800000",    # Maroon - statistical bounds
        
        # Medium Severity (Red to Red-Orange)
        "High Variance": "#990000",    # Medium red - variance issues
        "Sustained": "#B30000",        # Bright red - sustained change
        "Limit Exceed": "#CC0000",     # Pure red - limit exceeded
        
        # Lower Severity (Orange-Red to Orange)
        "Daily Flux": "#E63300",       # Red-orange - daily fluctuation
        "Medium Term": "#FF6600",      # Orange - medium-term
        "Short Term": "#FF9933",       # Light orange - short-term
        
        # Unknown
        "Other": "#333333"             # Dark grey - uncategorized
    }


def get_unique_sources(results: List[dict]) -> Dict[str, int]:
    """Extract all unique source strings and their frequencies."""
    source_counts = defaultdict(int)
    for r in results:
        source = r.get('source', 'unknown')
        if source:
            source_counts[source] += 1
    return dict(source_counts)


def truncate_source_label(source: str, max_length: int = 20) -> str:
    """Truncate long source strings for display."""
    if not source:
        return "Unknown"

    if len(source) <= max_length:
        return source

    # Try to truncate at a meaningful boundary
    if '/' in source:
        parts = source.split('/')
        return parts[-1][:max_length-3] + "..." if len(parts[-1]) > max_length else parts[-1]
    elif '-' in source:
        parts = source.split('-')
        if len(parts) > 2:
            return f"{parts[0][:8]}-...-{parts[-1][:8]}"

    return source[:max_length-3] + "..."


def generate_marker_sequence() -> List[str]:
    """Generate an ordered list of distinct matplotlib markers."""
    return [
        'o',  # circle
        's',  # square
        '^',  # triangle up
        'v',  # triangle down
        'D',  # diamond
        'p',  # pentagon
        'h',  # hexagon
        '*',  # star
        'P',  # plus (filled)
        'X',  # x (filled)
        'd',  # thin diamond
        '<',  # triangle left
        '>',  # triangle right
        '8',  # octagon
        'H',  # hexagon rotated
    ]


def get_normalized_marker_size(marker_type: str, base_size: int = 60) -> int:
    """Get normalized size for different marker types to ensure visual consistency."""
    size_factors = {
        'o': 1.0,   # circle - baseline
        's': 0.85,  # square - appears larger
        '^': 1.15,  # triangle up - appears smaller
        'v': 1.15,  # triangle down
        'D': 0.9,   # diamond - appears larger
        'p': 1.05,  # pentagon
        'h': 0.95,  # hexagon - appears slightly larger
        '*': 1.25,  # star - appears smaller
        'P': 1.1,   # plus
        'X': 1.1,   # x
        'd': 1.2,   # thin diamond - appears smaller
        '<': 1.15,  # triangle left
        '>': 1.15,  # triangle right
        '8': 0.95,  # octagon - appears slightly larger
        'H': 0.95,  # hexagon rotated
        '.': 1.5,   # point - needs to be much larger
    }
    
    factor = size_factors.get(marker_type, 1.0)
    return int(base_size * factor)


def generate_color_palette(n_colors: int) -> List[str]:
    """Generate a visually distinct color palette."""
    # Primary distinct colors
    base_colors = [
        '#2E7D32',  # Green
        '#1976D2',  # Blue
        '#7B1FA2',  # Purple
        '#F57C00',  # Orange
        '#C62828',  # Red
        '#00796B',  # Teal
        '#5D4037',  # Brown
        '#455A64',  # Blue Grey
        '#E91E63',  # Pink
        '#FBC02D',  # Yellow
        '#512DA8',  # Deep Purple
        '#0288D1',  # Light Blue
        '#388E3C',  # Light Green
        '#D32F2F',  # Light Red
        '#1565C0',  # Darker Blue
    ]

    if n_colors <= len(base_colors):
        return base_colors[:n_colors]

    # Generate additional colors by varying brightness
    import colorsys
    colors = base_colors.copy()

    for base_color in base_colors:
        if len(colors) >= n_colors:
            break
        # Convert hex to RGB
        hex_color = base_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16)/255.0 for i in (0, 2, 4))
        h, s, v = colorsys.rgb_to_hsv(r, g, b)

        # Create variations
        for v_mod in [0.7, 1.3]:
            new_v = min(1.0, v * v_mod)
            new_rgb = colorsys.hsv_to_rgb(h, s, new_v)
            new_hex = '#{:02x}{:02x}{:02x}'.format(
                int(new_rgb[0]*255), int(new_rgb[1]*255), int(new_rgb[2]*255)
            )
            colors.append(new_hex)
            if len(colors) >= n_colors:
                break

    return colors[:n_colors]


def get_global_source_patterns():
    """Define canonical source patterns for consistent visual mapping across all users."""
    return [
        # Most reliable sources first
        ('patient-device-scale', 0),      # Physical scales
        ('patient-device', 1),            # Other devices
        ('internal-questionnaire', 2),    # Questionnaires
        ('connectivehealth', 3),          # ConnectiveHealth API
        ('api.iglucose.com', 4),          # iGlucose API
        ('iglucose', 5),                  # iGlucose variants
        ('patient-upload', 6),            # Manual uploads
        ('manual', 7),                    # Manual entry
        ('test', 8),                      # Test data
        ('https://', 9),                  # Other APIs
        ('http://', 10),                  # Other APIs (non-secure)
        ('scale', 11),                    # Generic scale
        ('device', 12),                   # Generic device
        ('upload', 13),                   # Generic upload
        ('questionnaire', 14),            # Generic questionnaire
    ]


def match_source_to_pattern(source: str, patterns: List[Tuple[str, int]]) -> int:
    """Match a source string to its canonical pattern and return priority."""
    if not source:
        return 999
    
    source_lower = source.lower()
    
    # Check each pattern in order
    for pattern, priority in patterns:
        if pattern in source_lower:
            return priority
    
    # Hash the source to get a consistent but arbitrary priority for unknown sources
    # This ensures the same unknown source always gets the same priority
    import hashlib
    hash_val = int(hashlib.md5(source.encode()).hexdigest()[:8], 16)
    return 100 + (hash_val % 100)  # Priority 100-199 for unknown sources


def create_source_registry(results: List[dict], max_primary_sources: int = 15) -> Dict[str, Dict]:
    """Create a registry mapping unique sources to visual properties with global consistency."""
    source_counts = get_unique_sources(results)
    patterns = get_global_source_patterns()
    
    # Assign priorities based on global patterns, not local frequency
    source_priorities = {}
    for source in source_counts:
        source_priorities[source] = match_source_to_pattern(source, patterns)
    
    # Sort by global priority first, then by frequency as tiebreaker
    sorted_sources = sorted(
        source_counts.items(),
        key=lambda x: (source_priorities[x[0]], -x[1])  # Priority ascending, frequency descending
    )
    
    markers = generate_marker_sequence()
    colors = generate_color_palette(max_primary_sources)
    
    registry = {}
    
    # Track used colors to ensure uniqueness for primary sources
    used_colors = set()
    color_index = 0
    
    for i, (source, count) in enumerate(sorted_sources):
        priority = source_priorities[source]
        
        if i < max_primary_sources:  # Primary sources
            # Use consistent marker based on global priority
            marker_idx = priority % len(markers)
            
            # For colors, ensure uniqueness for primary sources
            if priority < 100:  # Known sources with defined patterns
                # Try to use the priority-based color first
                preferred_color_idx = priority % len(colors)
                if colors[preferred_color_idx] not in used_colors:
                    color_idx = preferred_color_idx
                else:
                    # Find next available color
                    while color_index < len(colors) and colors[color_index] in used_colors:
                        color_index += 1
                    color_idx = color_index if color_index < len(colors) else 0
            else:
                # Unknown sources get next available color
                while color_index < len(colors) and colors[color_index] in used_colors:
                    color_index += 1
                color_idx = color_index if color_index < len(colors) else 0
            
            selected_color = colors[color_idx] if color_idx < len(colors) else '#9E9E9E'
            used_colors.add(selected_color)
            color_index = color_idx + 1
            
            registry[source] = {
                'id': f'source_{priority:03d}',
                'marker': markers[marker_idx],
                'color': selected_color,
                'size': get_normalized_marker_size(markers[marker_idx], 60),  # Normalized size
                'label': truncate_source_label(source),
                'frequency': count,
                'priority': priority,  # Use global priority
                'is_primary': True
            }
        else:
            # Secondary sources grouped as "other"
            registry[source] = {
                'id': f'source_{priority:03d}',
                'marker': '.',
                'color': '#9E9E9E',
                'size': get_normalized_marker_size('.', 40),  # Normalized size for dots
                'label': 'Other',
                'frequency': count,
                'priority': 999,
                'is_primary': False
            }
    
    return registry





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
    # Removed - no longer adding rejection annotations
    return


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

    # Create source registry for all results
    source_registry = create_source_registry(all_results)

    # Track which sources we've added to legend
    legend_sources = set()

    if valid_results:
        ax1.plot(valid_timestamps, filtered_weights, '-', linewidth=1,
                color='#1565C0', label='Kalman Filtered', zorder=2)

        # Group valid results by unique source for plotting
        source_groups = defaultdict(lambda: {'timestamps': [], 'weights': []})

        for r, ts, w in zip(valid_results, valid_timestamps, valid_raw_weights):
            source = r.get('source', 'unknown')
            source_groups[source]['timestamps'].append(ts)
            source_groups[source]['weights'].append(w)

        # Plot each unique source with its specific style
        # Sort by priority (frequency) to ensure most common sources are plotted last (on top)
        for source in sorted(source_groups.keys(),
                           key=lambda x: source_registry.get(x, {}).get('priority', 999)):
            if source_groups[source]['timestamps']:
                style = source_registry.get(source, {
                    'marker': '.', 'size': 40, 'color': '#9E9E9E',
                    'label': 'Unknown', 'is_primary': False
                })

                # Only add primary sources to legend individually
                if style['is_primary'] and source not in legend_sources:
                    label = f"{style['label']} ({len(source_groups[source]['timestamps'])})"
                    legend_sources.add(source)
                elif not style['is_primary'] and 'Other' not in legend_sources:
                    # Group all secondary sources under "Other"
                    label = f"Other Sources"
                    legend_sources.add('Other')
                else:
                    label = None  # Don't add to legend again

                ax1.scatter(
                    source_groups[source]['timestamps'],
                    source_groups[source]['weights'],
                    marker=style['marker'],
                    s=style['size'],
                    alpha=0.9,  # Slightly higher alpha for clarity
                    color=style['color'],
                    label=label,
                    zorder=5,
                    edgecolors='white',  # White outline for overlap visibility
                    linewidth=0.5  # Subtle outline
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
        # Plot rejections with source-specific markers and rejection-colored outlines
        color_map = get_rejection_color_map()
        
        # Group rejections by source and reason for combined plotting
        rejection_data = defaultdict(lambda: {'timestamps': [], 'weights': [], 'reasons': [], 'categories': []})
        
        for r, ts, w in zip(rejected_results, rejected_timestamps, rejected_raw_weights):
            source = r.get('source', 'unknown')
            reason = r.get('reason', 'Unknown')
            category = categorize_rejection(reason)
            rejection_data[source]['timestamps'].append(ts)
            rejection_data[source]['weights'].append(w)
            rejection_data[source]['reasons'].append(reason)
            rejection_data[source]['categories'].append(category)
        
        # Track rejection categories for legend
        rejection_legend_items = {}
        
        # Plot each source with rejection-colored outlines
        for source in sorted(rejection_data.keys(),
                           key=lambda x: source_registry.get(x, {}).get('priority', 999)):
            if rejection_data[source]['timestamps']:
                style = source_registry.get(source, {
                    'marker': '.', 'size': 40, 'color': '#9E9E9E',
                    'label': 'Unknown', 'is_primary': False
                })
                
                # Plot each point with its specific rejection color outline
                for ts, w, category in zip(rejection_data[source]['timestamps'],
                                          rejection_data[source]['weights'],
                                          rejection_data[source]['categories']):
                    edge_color = color_map.get(category, '#757575')
                    
                    # Track categories for legend
                    if category not in rejection_legend_items:
                        rejection_legend_items[category] = {'count': 0, 'color': edge_color}
                    rejection_legend_items[category]['count'] += 1
                    
                    ax1.scatter(
                        [ts], [w],
                        marker=style['marker'],
                        s=style['size'],
                        alpha=0.7,  # Semi-transparent fill
                        color=style['color'],
                        edgecolors=edge_color,  # Red-orange-black gradient color
                        linewidth=2.5,  # Visible but not overwhelming
                        zorder=4
                    )
        
        # Add legend entries for rejection categories
        for category, info in sorted(rejection_legend_items.items(), 
                                    key=lambda x: x[1]['count'], reverse=True):
            # Create a dummy plot for legend
            ax1.plot([], [], 'o', color='none', markeredgecolor=info['color'], 
                    markeredgewidth=2.5, markersize=8,
                    label=f'Rej: {category} ({info["count"]})')

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
            # Position gap label at the bottom of the chart
            y_min = ax1.get_ylim()[0]
            ax1.annotate(f"{int(reset['gap_days'])}d gap",
                        xy=(reset['timestamp'], y_min),
                        xytext=(0, 10), textcoords='offset points',
                        fontsize=9, color='gray', alpha=0.8,
                        ha='center', va='bottom')

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

    # BMI and Data Quality Statistics
    bmi_detections = 0
    unit_conversions = 0
    physio_rejections = 0

    for r in all_results:
        if r.get('bmi_details'):
            if r['bmi_details'].get('bmi_converted'):
                bmi_detections += 1
            if r['bmi_details'].get('unit_converted'):
                unit_conversions += 1
        if r.get('rejection_insights'):
            if r['rejection_insights'].get('category') == 'Physiological_Limit':
                physio_rejections += 1

    if bmi_detections > 0 or unit_conversions > 0:
        stats_sections.append([
            "DATA QUALITY",
            f"BMI Detections: {bmi_detections}",
            f"Unit Conversions: {unit_conversions}",
            f"Physio Rejections: {physio_rejections}",
            ""
        ])

    # Source Distribution Analysis
    source_counts = defaultdict(lambda: {'total': 0, 'accepted': 0, 'rejected': 0})

    for r in all_results:
        source = r.get('source', 'unknown')
        source_counts[source]['total'] += 1
        if r.get('accepted'):
            source_counts[source]['accepted'] += 1
        else:
            source_counts[source]['rejected'] += 1

    if source_counts:
        source_lines = ["SOURCE ANALYSIS"]
        # Sort by total count (most common first)
        sorted_sources = sorted(
            source_counts.items(),
            key=lambda x: x[1]['total'],
            reverse=True
        )

        # Show top sources
        num_primary = sum(1 for s in sorted_sources if source_registry.get(s[0], {}).get('is_primary', False))
        num_shown = min(5, len(sorted_sources))

        for source, counts in sorted_sources[:num_shown]:
            style = source_registry.get(source, {'label': truncate_source_label(source)})
            accept_rate = 100 * counts['accepted'] / counts['total'] if counts['total'] > 0 else 0
            label = style['label'][:15]  # Truncate for stats panel
            source_lines.append(
                f"{label:15s}: {counts['total']:3d} ({accept_rate:.0f}%)"
            )

        if len(sorted_sources) > num_shown:
            # Show summary of remaining sources
            other_total = sum(c[1]['total'] for c in sorted_sources[num_shown:])
            other_accepted = sum(c[1]['accepted'] for c in sorted_sources[num_shown:])
            other_rate = 100 * other_accepted / other_total if other_total > 0 else 0
            source_lines.append(
                f"{'Other':15s}: {other_total:3d} ({other_rate:.0f}%)"
            )

        source_lines.append("")
        stats_sections.append(source_lines)

    # BMI Analysis
    if all_results:
        bmi_stats = {
            'bmi_detected': 0,
            'unit_converted': 0,
            'categories': defaultdict(int)
        }

        for r in all_results:
            bmi_details = r.get('bmi_details', {})
            if bmi_details:
                if bmi_details.get('bmi_converted'):
                    bmi_stats['bmi_detected'] += 1
                if bmi_details.get('unit_converted'):
                    bmi_stats['unit_converted'] += 1
                category = bmi_details.get('bmi_category')
                if category:
                    bmi_stats['categories'][category] += 1

        if bmi_stats['bmi_detected'] > 0 or bmi_stats['unit_converted'] > 0:
            bmi_lines = ["BMI & CONVERSION"]
            if bmi_stats['bmi_detected'] > 0:
                bmi_lines.append(f"BMI→Weight: {bmi_stats['bmi_detected']}")
            if bmi_stats['unit_converted'] > 0:
                bmi_lines.append(f"Unit Conv: {bmi_stats['unit_converted']}")

            # Show BMI distribution if available
            if bmi_stats['categories']:
                for cat in ['underweight', 'normal', 'overweight', 'obese']:
                    if cat in bmi_stats['categories']:
                        count = bmi_stats['categories'][cat]
                        pct = 100 * count / len(all_results)
                        bmi_lines.append(f"{cat[:10]:10s}: {pct:.0f}%")
            bmi_lines.append("")
            stats_sections.append(bmi_lines)

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

    # Create source distribution stacked bar chart for unique sources
    source_data = defaultdict(lambda: {'accepted': 0, 'rejected': 0})

    for r in all_results:
        source = r.get('source', 'unknown')
        if r.get('accepted'):
            source_data[source]['accepted'] += 1
        else:
            source_data[source]['rejected'] += 1

    if source_data:
        # Sort by total count and get top sources
        sorted_sources = sorted(
            source_data.items(),
            key=lambda x: x[1]['accepted'] + x[1]['rejected'],
            reverse=True
        )

        # Show top primary sources plus "Other" if needed
        max_bars = 8  # Maximum number of bars to show

        source_names = []
        accepted_counts = []
        rejected_counts = []
        colors = []

        other_accepted = 0
        other_rejected = 0

        for i, (source, counts) in enumerate(sorted_sources):
            if i < max_bars - 1:  # Reserve one spot for "Other"
                style = source_registry.get(source, {'label': truncate_source_label(source, 12), 'color': '#9E9E9E'})
                source_names.append(style['label'][:12])  # Truncate for x-axis
                accepted_counts.append(counts['accepted'])
                rejected_counts.append(counts['rejected'])
                colors.append(style['color'])
            else:
                # Accumulate remaining sources as "Other"
                other_accepted += counts['accepted']
                other_rejected += counts['rejected']

        # Add "Other" category if there are more sources
        if len(sorted_sources) > max_bars - 1:
            source_names.append('Other')
            accepted_counts.append(other_accepted)
            rejected_counts.append(other_rejected)
            colors.append('#9E9E9E')

        x_pos = np.arange(len(source_names))

        # Create stacked bar chart with source-specific colors
        bars1 = ax10.bar(x_pos, accepted_counts, color=colors, alpha=0.8,
                        edgecolor='white', linewidth=1.5, label='Accepted')
        bars2 = ax10.bar(x_pos, rejected_counts, bottom=accepted_counts,
                        color=colors, alpha=0.4, edgecolor='red', linewidth=1.5,
                        linestyle='--', label='Rejected')

        ax10.set_xticks(x_pos)
        ax10.set_xticklabels(source_names, fontsize=9, rotation=45, ha='right')
        ax10.set_ylabel('Count', fontsize=11)
        ax10.set_title(f'Measurements by Source ({len(source_data)} unique sources)',
                      fontsize=12, fontweight='bold')
        ax10.legend(loc='upper right', fontsize=9)
        ax10.grid(True, alpha=0.3, axis='y')

        # Add percentage labels on bars
        for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
            total = accepted_counts[i] + rejected_counts[i]
            if total > 0:
                accept_pct = 100 * accepted_counts[i] / total
                # Add count on top of bar
                y_pos = total + max(accepted_counts + rejected_counts) * 0.02
                ax10.text(bar1.get_x() + bar1.get_width()/2, y_pos,
                         f'{total}\n{accept_pct:.0f}%',
                         ha='center', va='bottom', fontsize=8)
    else:
        ax10.text(0.5, 0.5, 'No Source Data', transform=ax10.transAxes,
                 fontsize=14, ha='center', va='center', color='#757575')
        ax10.set_title('Measurements by Source', fontsize=12, fontweight='bold')
        ax10.axis('off')

    # Don't use tight_layout with complex GridSpec - we set the spacing manually
    # This avoids the warning about incompatible layouts

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    filename = output_path / f"{user_id}.png"
    plt.savefig(filename, dpi=viz_config["dashboard_dpi"], bbox_inches="tight")
    plt.close()

    return str(filename)

