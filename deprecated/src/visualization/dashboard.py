"""Enhanced dashboard visualization for weight analysis with baseline tracking."""

import logging
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple

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
    logger.warning("matplotlib not installed. Visualization disabled.")


def create_unified_visualization(user_id: str, data: Dict[str, Any], output_dir: Path, config: Dict[str, Any]):
    """Create an enhanced dashboard showing weight trajectory with baseline establishment."""

    if not HAS_MATPLOTLIB:
        return

    # Get both validated time series and all readings (including rejected)
    time_series = data.get('time_series', [])
    all_readings = data.get('all_readings_with_validation', [])
    
    # If we have all_readings, use that for complete visualization
    if all_readings:
        # Use all_readings for visualization (includes rejected)
        use_all_readings = True
    else:
        # Fall back to time_series only
        use_all_readings = False
        if not time_series:
            logger.debug(f"No time series data for user {user_id}")
            return

    # Create figure with 5 subplots - increased size for new zoomed view
    fig = plt.figure(figsize=(20, 18), constrained_layout=False)

    # Main weight timeline (top) - shows all data including rejected
    ax1 = plt.subplot2grid((4, 2), (0, 0), colspan=2, rowspan=1)

    # Zoomed weight timeline (second row) - focused view of 2025 accepted data
    ax_zoom = plt.subplot2grid((4, 2), (1, 0), colspan=2, rowspan=1)

    # Kalman innovation (third row)
    ax2 = plt.subplot2grid((4, 2), (2, 0), colspan=2, rowspan=1)

    # Statistics panel (bottom left)
    ax3 = plt.subplot2grid((4, 2), (3, 0), colspan=1)

    # Weight distribution (bottom right)
    ax4 = plt.subplot2grid((4, 2), (3, 1), colspan=1)

    # Process time series data
    if use_all_readings:
        # Process all readings including rejected ones
        all_dates = []
        all_weights = []
        all_sources = []
        all_validated = []
        all_rejection_reasons = []
        
        for reading in all_readings:
            try:
                date = datetime.fromisoformat(reading['date'].replace('Z', '+00:00')) if isinstance(reading['date'], str) else reading['date']
                all_dates.append(date)
                all_weights.append(reading['weight'])
                all_sources.append(reading.get('source', 'unknown'))
                all_validated.append(reading.get('validated', False))
                all_rejection_reasons.append(reading.get('rejection_reason', None))
            except:
                continue
        
        # Also get the validated subset for Kalman display
        dates = [datetime.fromisoformat(ts['date'].replace('Z', '+00:00')) for ts in time_series]
        weights = [ts['weight'] for ts in time_series]
        confidences = [ts.get('confidence', 0.5) for ts in time_series]
    else:
        # Use only validated time series
        dates = [datetime.fromisoformat(ts['date'].replace('Z', '+00:00')) for ts in time_series]
        weights = [ts['weight'] for ts in time_series]
        confidences = [ts.get('confidence', 0.5) for ts in time_series]
        all_dates = dates
        all_weights = weights
        all_sources = [ts.get('source', 'unknown') for ts in time_series]
        all_validated = [True] * len(dates)
        all_rejection_reasons = [None] * len(dates)

    # Get Kalman filtered data if available
    kalman_filtered = [ts.get('kalman_filtered') for ts in time_series]
    kalman_uncertainty = [ts.get('kalman_uncertainty') for ts in time_series]
    has_kalman = any(kf is not None for kf in kalman_filtered)

    # Get rejected measurements from all_readings if available
    rejected_measurements = []
    if use_all_readings and 'all_readings' in data:
        # Build set of accepted dates for quick lookup
        accepted_dates = set()
        for ts in time_series:
            try:
                ts_date = datetime.fromisoformat(ts['date'].replace('Z', '+00:00'))
                accepted_dates.add(ts_date.date())  # Use date only for matching
            except:
                pass
        
        # Find rejected readings
        for reading in data['all_readings']:
            if reading.get('is_rejected'):
                try:
                    rej_date = datetime.fromisoformat(reading['date'].replace('Z', '+00:00'))
                    rejected_measurements.append({
                        'date': rej_date,
                        'weight': reading['weight'],
                        'source': reading.get('source', 'unknown'),
                        'reason': reading.get('rejection_reason', 'unknown')
                    })
                except Exception as e:
                    logger.debug(f"Error processing rejected reading: {e}")
    elif 'kalman_time_series' in data:
        # Fallback to kalman_time_series for backward compatibility
        for kts in data['kalman_time_series']:
            if kts.get('measurement_accepted') == False:
                try:
                    kts_date = datetime.fromisoformat(kts['date'].replace('Z', '+00:00'))
                    rejected_measurements.append({
                        'date': kts_date,
                        'weight': kts.get('measurement', 0),
                        'source': 'unknown',
                        'reason': kts.get('rejection_reason', 'unknown')
                    })
                except Exception as e:
                    logger.debug(f"Error processing rejected measurement: {e}")

    # ========== 1. MAIN WEIGHT TIMELINE ==========

    # Define source styles with different edge colors for each source
    source_styles = {
        'care-team-upload': {'marker': '^', 'edge_color': '#2E7D32', 'label': 'Care Team', 'size': 100},
        'patient-device': {'marker': 's', 'edge_color': '#1565C0', 'label': 'Patient Device', 'size': 80},
        'internal-questionnaire': {'marker': 'D', 'edge_color': '#7B1FA2', 'label': 'Questionnaire', 'size': 120},
        'patient-upload': {'marker': 'v', 'edge_color': '#E65100', 'label': 'Patient Upload', 'size': 80},
        'https://connectivehealth.io': {'marker': 'p', 'edge_color': '#00695C', 'label': 'Connected Device', 'size': 70},
        'https://api.iglucose.com': {'marker': 'h', 'edge_color': '#B71C1C', 'label': 'iGlucose', 'size': 70},
        'unknown': {'marker': 'o', 'edge_color': '#616161', 'label': 'Other', 'size': 60}
    }

    # Plot baseline establishment periods (can be multiple due to gaps)
    baseline_history = data.get('baseline_history', [])
    if baseline_history:
        for i, baseline_entry in enumerate(baseline_history):
            try:
                trigger_date = datetime.fromisoformat(baseline_entry['trigger_date'])
                gap_days = baseline_entry.get('gap_days', 0)

                # Shade baseline establishment period (7 days from trigger)
                baseline_end = trigger_date + timedelta(days=7)
                ax1.axvspan(trigger_date, baseline_end,
                           alpha=0.1, color='#9C27B0',
                           label='Baseline Period' if i == 0 else None, zorder=1)

                # Mark baseline trigger point
                label = f'Baseline {i+1}' if len(baseline_history) > 1 else 'Baseline Start'
                if gap_days > 0:
                    label += f' (after {gap_days}d gap)'

                ax1.axvline(x=trigger_date, color='#9C27B0',
                           linestyle=':', linewidth=1.5, alpha=0.6,
                           label=label, zorder=2)
            except Exception as e:
                logger.debug(f"Error plotting baseline {i}: {e}")

    # If no baseline history but we have a baseline, show legacy visualization
    elif data.get('baseline_established') and data.get('signup_date'):
        try:
            first_date = datetime.fromisoformat(data['signup_date'].replace('Z', '+00:00'))

            # Mark first reading date
            ax1.axvline(x=first_date, color='#9C27B0',
                       linestyle=':', linewidth=1.5, alpha=0.6,
                       label='First Reading', zorder=2)
        except Exception as e:
            logger.debug(f"Error plotting first reading marker: {e}")

    # Group readings by source for better visualization
    source_groups = {}
    
    if use_all_readings:
        # Group ALL readings (including rejected) by source
        for i in range(len(all_dates)):
            source = all_sources[i]
            if source not in source_groups:
                source_groups[source] = {
                    'dates': [], 'weights': [], 'validated': [], 
                    'rejection_reasons': [], 'confidences': []
                }
            source_groups[source]['dates'].append(all_dates[i])
            source_groups[source]['weights'].append(all_weights[i])
            source_groups[source]['validated'].append(all_validated[i])
            source_groups[source]['rejection_reasons'].append(all_rejection_reasons[i])
            # Map confidence from time_series for validated points
            conf = 0.5
            if all_validated[i]:
                for j, d in enumerate(dates):
                    if abs((d - all_dates[i]).total_seconds()) < 60:
                        conf = confidences[j]
                        break
            source_groups[source]['confidences'].append(conf)
    else:
        # Original grouping for backward compatibility
        for i, ts in enumerate(time_series):
            source = ts.get('source', 'unknown')
            if source not in source_groups:
                source_groups[source] = {'dates': [], 'weights': [], 'confidences': [], 
                                        'validated': [], 'rejection_reasons': []}
            source_groups[source]['dates'].append(dates[i])
            source_groups[source]['weights'].append(weights[i])
            source_groups[source]['confidences'].append(confidences[i])
            source_groups[source]['validated'].append(True)
            source_groups[source]['rejection_reasons'].append(None)

    # Plot data points - rejected first (lower z-order), then accepted
    total_rejected = 0
    
    if use_all_readings:
        # Plot using our validation tracking
        for source, points in source_groups.items():
            style = source_styles.get(source, source_styles['unknown'])
            if points['dates']:
                # Plot rejected points first (lower z-order)
                for j in range(len(points['dates'])):
                    if not points['validated'][j]:
                        # Rejected points: red fill, dark red edge
                        ax1.scatter(points['dates'][j], points['weights'][j],
                                   marker=style['marker'], c='#FF4444',
                                   s=style['size'], alpha=0.6,
                                   edgecolors='darkred', linewidth=2, zorder=20)
                        total_rejected += 1
                
                # Plot validated points (higher z-order)
                for j in range(len(points['dates'])):
                    if points['validated'][j]:
                        # Color based on confidence
                        conf = points['confidences'][j]
                        if conf >= 0.9:
                            color = '#4CAF50'  # Green
                        elif conf >= 0.75:
                            color = '#8BC34A'  # Light green
                        elif conf >= 0.6:
                            color = '#FFC107'  # Yellow
                        else:
                            color = '#FF9800'  # Orange
                        
                        ax1.scatter(points['dates'][j], points['weights'][j],
                                   marker=style['marker'], c=color,
                                   s=style['size'], alpha=0.9,
                                   edgecolors=style['edge_color'], linewidth=1.5, zorder=50)
    else:
        # Original code for backward compatibility
        for source, points in source_groups.items():
            style = source_styles.get(source, source_styles['unknown'])
            if points['dates']:
                # Check which points were rejected
                for j, (d, w) in enumerate(zip(points['dates'], points['weights'])):
                    is_rejected = False
                    for rej in rejected_measurements:
                        if abs((d - rej['date']).total_seconds()) < 1 and abs(w - rej['weight']) < 0.01:
                            is_rejected = True
                            break

                    if is_rejected:
                        # Rejected points: red fill, red edge, lower z-order
                        ax1.scatter(points['dates'][j], points['weights'][j],
                                   marker=style['marker'], c='#FF4444',  # Red fill
                                   s=style['size'], alpha=1,
                                   edgecolors='darkred', linewidth=2, zorder=20)
                        total_rejected += 1

        # Second pass: plot all accepted points with higher z-order
        for source, points in source_groups.items():
            style = source_styles.get(source, source_styles['unknown'])
            if points['dates']:
                # Check which points are accepted
                for j, (d, w) in enumerate(zip(points['dates'], points['weights'])):
                    is_rejected = False
                    for rej in rejected_measurements:
                        if abs((d - rej['date']).total_seconds()) < 1 and abs(w - rej['weight']) < 0.01:
                            is_rejected = True
                            break

                    if not is_rejected:
                        # Accepted points: blue fill, source-specific edge color, higher z-order
                        ax1.scatter(points['dates'][j], points['weights'][j],
                                   marker=style['marker'], c='#2196F3',  # Blue fill
                                   s=style['size'], alpha=1,
                                   edgecolors=style['edge_color'], linewidth=1.5, zorder=50)

    # Add legend entries for each source
    for source, points in source_groups.items():
        style = source_styles.get(source, source_styles['unknown'])
        if points['dates']:
            ax1.scatter([], [], marker=style['marker'], c='#2196F3',
                       s=style['size'], alpha=1,
                       label=f"{style['label']} (n={len(points['dates'])})",
                       edgecolors=style['edge_color'], linewidth=1.5)

    # Add rejected measurements to legend if any exist
    if use_all_readings and total_rejected > 0:
        # Create legend entry showing red filled marker
        ax1.scatter([], [], marker='o', c='#FF4444', s=80,
                   edgecolors='darkred', linewidth=2,
                   label=f"Rejected ({total_rejected})")
    elif rejected_measurements:
        # Original legend for backward compatibility
        ax1.scatter([], [], marker='o', c='#FF4444', s=80,
                   edgecolors='darkred', linewidth=2,
                   label=f"Rejected ({len(rejected_measurements)})")

    # Plot baseline results at establishment points
    baseline_history = data.get('baseline_history', [])
    if baseline_history:
        for i, baseline_entry in enumerate(baseline_history):
            try:
                trigger_date = datetime.fromisoformat(baseline_entry['trigger_date'])
                baseline_weight = baseline_entry['weight']
                baseline_end = trigger_date + timedelta(days=7)

                # Draw short horizontal line during baseline period showing the result
                ax1.plot([trigger_date, baseline_end], [baseline_weight, baseline_weight],
                        color='#9C27B0', linestyle='--', linewidth=2, alpha=1)

                # Add a marker at the end of baseline period
                ax1.scatter(baseline_end, baseline_weight,
                           marker='D', s=100, c='#9C27B0',
                           edgecolors='white', linewidth=2, zorder=60,
                           label=f'Baseline {i+1}: {baseline_weight:.1f} kg' if i == 0 else None)

                # Add text annotation with the baseline value
                ax1.annotate(f'{baseline_weight:.1f} kg',
                            xy=(baseline_end, baseline_weight),
                            xytext=(10, 5), textcoords='offset points',
                            fontsize=9, color='#9C27B0', fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.3',
                                    facecolor='white', edgecolor='#9C27B0', alpha=0.8))
            except Exception as e:
                logger.debug(f"Error plotting baseline result {i}: {e}")

    # Also show the final baseline if it's different from baseline_history
    if data.get('baseline_established') and data.get('baseline_weight'):
        # Check if this is different from what's in baseline_history
        show_final_baseline = True
        if baseline_history:
            # Don't show if it's too close to the last baseline in history
            last_history_weight = baseline_history[-1]['weight']
            if abs(data['baseline_weight'] - last_history_weight) < 1.0:
                show_final_baseline = False

        if show_final_baseline and not baseline_history:
            baseline_weight = data['baseline_weight']
            # Try to find a reasonable date to show it
            if data.get('signup_date'):
                try:
                    start_date = datetime.fromisoformat(data['signup_date'].replace('Z', '+00:00'))
                    end_date = start_date + timedelta(days=7)

                    # Draw short line during baseline period
                    ax1.plot([start_date, end_date], [baseline_weight, baseline_weight],
                            color='#9C27B0', linestyle='--', linewidth=2, alpha=1)

                    # Add marker
                    ax1.scatter(end_date, baseline_weight,
                               marker='D', s=100, c='#9C27B0',
                               edgecolors='white', linewidth=2, zorder=60,
                               label=f'Baseline: {baseline_weight:.1f} kg')

                    # Add text annotation
                    ax1.annotate(f'{baseline_weight:.1f} kg',
                                xy=(end_date, baseline_weight),
                                xytext=(10, 5), textcoords='offset points',
                                fontsize=9, color='#9C27B0', fontweight='bold',
                                bbox=dict(boxstyle='round,pad=0.3',
                                        facecolor='white', edgecolor='#9C27B0', alpha=1))
                except Exception as e:
                    logger.debug(f"Error plotting legacy baseline: {e}")

    # Plot Kalman filter trajectory if available
    if has_kalman:
        # Filter out None values and get corresponding dates
        kalman_data = [(d, kf, ku) for d, kf, ku in zip(dates, kalman_filtered, kalman_uncertainty)
                      if kf is not None]

        if kalman_data:
            k_dates, k_filtered, k_uncertainty = zip(*kalman_data)

            # Main Kalman trajectory
            ax1.plot(k_dates, k_filtered, color='#4CAF50',
                    linewidth=2.5, alpha=1,
                    label='Kalman Filter', zorder=40)

            # Uncertainty envelope (2-sigma)
            upper = [kf + 2*ku for kf, ku in zip(k_filtered, k_uncertainty)]
            lower = [kf - 2*ku for kf, ku in zip(k_filtered, k_uncertainty)]
            ax1.fill_between(k_dates, lower, upper,
                            alpha=0.15, color='#4CAF50',
                            label='95% Confidence', zorder=30)

    # Formatting with improved padding
    if use_all_readings and total_rejected > 0:
        ax1.set_title(f'Weight Trajectory - User {user_id[:12]} ({total_rejected} Rejected Shown in Red)',
                      fontsize=14, fontweight='bold', pad=20)
    else:
        ax1.set_title(f'Weight Trajectory - User {user_id[:12]}',
                      fontsize=14, fontweight='bold', pad=20)
    ax1.set_ylabel('Weight (kg)', fontsize=12, labelpad=10)
    ax1.grid(True, alpha=0.2, linestyle='--')

    # Move legend outside plot area to the right side
    ax1.legend(bbox_to_anchor=(1.02, 0.5), loc='center left', fontsize=9,
               frameon=True, fancybox=True, shadow=True,
               borderaxespad=0, columnspacing=1.5)

    # Format x-axis dates
    format_date_axis(ax1, dates)

    # Set y-axis limits based on ALL data points (including rejected) with extra padding
    if use_all_readings and all_weights:
        # Use all weights to ensure rejected points are visible
        y_min = min(all_weights)
        y_max = max(all_weights)
        y_range = y_max - y_min
        y_padding = max(y_range * 0.15, 5)  # Increased padding to ensure all points visible
        ax1.set_ylim(y_min - y_padding, y_max + y_padding)
    elif weights:
        # Fallback to validated weights only
        y_range = max(weights) - min(weights)
        y_padding = max(y_range * 0.15, 3)
        ax1.set_ylim(min(weights) - y_padding, max(weights) + y_padding)

    # ========== 2. ZOOMED WEIGHT TIMELINE (2025 ACCEPTED DATA) ==========
    
    # Filter for accepted data from 2025-01-01 onwards
    zoom_start_date = datetime(2025, 1, 1)
    
    # Get only validated readings from 2025 onwards
    zoom_dates = []
    zoom_weights = []
    zoom_sources = []
    zoom_confidences = []
    
    # Also get rejected readings from 2025 for display
    zoom_rejected_dates = []
    zoom_rejected_weights = []
    zoom_rejected_sources = []
    zoom_rejected_reasons = []
    
    # Use time_series for validated data (or filter all_readings for validated ones)
    for i, ts in enumerate(time_series):
        ts_date = dates[i]
        if ts_date >= zoom_start_date:
            zoom_dates.append(ts_date)
            zoom_weights.append(weights[i])
            zoom_sources.append(ts.get('source', 'unknown'))
            zoom_confidences.append(confidences[i])
    
    # Get rejected readings if available
    if use_all_readings and rejected_measurements:
        for rej in rejected_measurements:
            if rej['date'] >= zoom_start_date:
                zoom_rejected_dates.append(rej['date'])
                zoom_rejected_weights.append(rej['weight'])
                zoom_rejected_sources.append(rej.get('source', 'unknown'))
                zoom_rejected_reasons.append(rej.get('rejection_reason', 'Unknown'))
    
    if zoom_dates or zoom_rejected_dates:
        # Plot the zoomed view
        ax_zoom.axhline(y=0, color='gray', linestyle='-', alpha=0.1)
        
        # Group by source for the zoomed view
        zoom_source_groups = {}
        for i in range(len(zoom_dates)):
            source = zoom_sources[i]
            if source not in zoom_source_groups:
                zoom_source_groups[source] = {
                    'dates': [], 'weights': [], 'confidences': []
                }
            zoom_source_groups[source]['dates'].append(zoom_dates[i])
            zoom_source_groups[source]['weights'].append(zoom_weights[i])
            zoom_source_groups[source]['confidences'].append(zoom_confidences[i])
        
        # First plot rejected points (lower z-order)
        zoom_rejected_count = 0
        if zoom_rejected_dates:
            for i in range(len(zoom_rejected_dates)):
                source = zoom_rejected_sources[i]
                style = source_styles.get(source, source_styles['unknown'])
                # Rejected points: red fill with dark red edge
                ax_zoom.scatter(zoom_rejected_dates[i], zoom_rejected_weights[i],
                               marker=style['marker'], c='#FF4444',
                               s=style['size'] * 0.8, alpha=1,
                               edgecolors='darkred', linewidth=2, zorder=20)
                zoom_rejected_count += 1
        
        # Plot accepted points by source with confidence colors (higher z-order)
        for source, points in zoom_source_groups.items():
            style = source_styles.get(source, source_styles['unknown'])
            if points['dates']:
                for j in range(len(points['dates'])):
                    conf = points['confidences'][j]
                    if conf >= 0.9:
                        color = '#4CAF50'  # Green
                    elif conf >= 0.75:
                        color = '#8BC34A'  # Light green
                    elif conf >= 0.6:
                        color = '#FFC107'  # Yellow
                    else:
                        color = '#FF9800'  # Orange
                    
                    ax_zoom.scatter(points['dates'][j], points['weights'][j],
                               marker=style['marker'], c=color,
                               s=style['size'] * 0.8, alpha=0.9,
                               edgecolors=style['edge_color'], linewidth=1.5, zorder=50)
        
        # Plot Kalman filtered line if available (for 2025 data)
        if has_kalman:
            zoom_kalman_dates = []
            zoom_kalman_filtered = []
            zoom_kalman_upper = []
            zoom_kalman_lower = []
            
            for i, kf in enumerate(kalman_filtered):
                if kf is not None and dates[i] >= zoom_start_date:
                    zoom_kalman_dates.append(dates[i])
                    zoom_kalman_filtered.append(kf)
                    if kalman_uncertainty[i] is not None:
                        zoom_kalman_upper.append(kf + kalman_uncertainty[i])
                        zoom_kalman_lower.append(kf - kalman_uncertainty[i])
                    else:
                        zoom_kalman_upper.append(kf)
                        zoom_kalman_lower.append(kf)
            
            if zoom_kalman_dates:
                # Plot Kalman filtered line
                ax_zoom.plot(zoom_kalman_dates, zoom_kalman_filtered,
                        color='#4CAF50', linewidth=2, alpha=0.9,
                        label='Kalman Filter', zorder=40)
                
                # Plot uncertainty band
                ax_zoom.fill_between(zoom_kalman_dates, zoom_kalman_lower, zoom_kalman_upper,
                                 color='#4CAF50', alpha=0.1, zorder=30,
                                 label='95% Confidence')
        
        # Set y-axis limits based on ALL data (including rejected) for proper visibility
        all_zoom_weights = zoom_weights + zoom_rejected_weights
        if all_zoom_weights:
            y_min = min(all_zoom_weights)
            y_max = max(all_zoom_weights)
            y_range = y_max - y_min
            y_padding = max(y_range * 0.15, 3)  # 15% padding to ensure rejected points are visible
            ax_zoom.set_ylim(y_min - y_padding, y_max + y_padding)
        
        # Format x-axis for the zoomed view
        ax_zoom.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax_zoom.xaxis.set_major_locator(mdates.MonthLocator())
        ax_zoom.xaxis.set_minor_locator(mdates.WeekdayLocator())
        plt.setp(ax_zoom.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Labels and title for zoomed view
        if zoom_rejected_count > 0:
            ax_zoom.set_title(f'2025 Data (Zoomed) - {len(zoom_weights)} validated, {zoom_rejected_count} rejected',
                             fontsize=14, fontweight='bold', pad=20)
        else:
            ax_zoom.set_title(f'2025 Validated Data (Zoomed) - {len(zoom_weights)} readings',
                             fontsize=14, fontweight='bold', pad=20)
        ax_zoom.set_ylabel('Weight (kg)', fontsize=12, labelpad=10)
        ax_zoom.grid(True, alpha=0.2, linestyle='--')
        
        # Add legend for zoomed view
        handles, labels = [], []
        for source in zoom_source_groups.keys():
            style = source_styles.get(source, source_styles['unknown'])
            count = len(zoom_source_groups[source]['dates'])
            if count > 0:
                handles.append(ax_zoom.scatter([], [], marker=style['marker'], 
                                              c='#4CAF50', s=style['size'] * 0.8,
                                              edgecolors=style['edge_color'], linewidth=1.5))
                labels.append(f"{style['label']} (n={count})")
        
        # Add rejected points to legend if any
        if zoom_rejected_count > 0:
            handles.append(ax_zoom.scatter([], [], marker='o', c='#FF4444', 
                                          s=60, edgecolors='darkred', linewidth=2))
            labels.append(f"Rejected (n={zoom_rejected_count})")
        
        if has_kalman and zoom_kalman_dates:
            # Add Kalman to legend
            handles.append(ax_zoom.plot([], [], color='#4CAF50', linewidth=2)[0])
            labels.append('Kalman Filter')
        
        ax_zoom.legend(handles, labels, bbox_to_anchor=(1.02, 0.5), loc='center left', 
                       frameon=True, fancybox=True, shadow=True, fontsize=9)
        
    else:
        # No 2025 data to show
        ax_zoom.text(0.5, 0.5, 'No validated data from 2025 onwards',
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax_zoom.transAxes, fontsize=12, color='gray')
        ax_zoom.set_title('2025 Validated Data (Zoomed)', fontsize=14, fontweight='bold', pad=20)
        ax_zoom.grid(True, alpha=0.1)
    
    # ========== 3. KALMAN INNOVATION PLOT ==========

    if has_kalman and 'kalman_time_series' in data:
        kalman_ts = data['kalman_time_series']
        if kalman_ts:
            innovations = [k['innovation'] for k in kalman_ts]
            k_dates = [datetime.fromisoformat(k['date'].replace('Z', '+00:00'))
                      for k in kalman_ts]
            accepted = [k.get('measurement_accepted', True) for k in kalman_ts]
            
            # Check for filter reinitializations
            reinit_dates = []
            for k in kalman_ts:
                if k.get('filter_reinitialized', False):
                    try:
                        reinit_date = datetime.fromisoformat(k['date'].replace('Z', '+00:00'))
                        reinit_dates.append(reinit_date)
                    except:
                        pass

            # Color by magnitude and acceptance
            std_inn = np.std(innovations) if len(innovations) > 1 else 1.0

            # First pass: plot rejected points with lower z-order
            for i, (d, inn, acc) in enumerate(zip(k_dates, innovations, accepted)):
                if not acc:
                    # Determine color by magnitude
                    if abs(inn) < std_inn:
                        color = '#4CAF50'  # Green for normal
                    elif abs(inn) < 2*std_inn:
                        color = '#FFA726'  # Orange for elevated
                    else:
                        color = '#F44336'  # Red for high

                    # Rejected: colored dot with red outline, lower z-order
                    ax2.scatter(d, inn, c=color, s=30,
                               marker='o', alpha=1,
                               edgecolors='red', linewidth=2, zorder=10)

            # Second pass: plot accepted points with higher z-order
            for i, (d, inn, acc) in enumerate(zip(k_dates, innovations, accepted)):
                if acc:
                    # Determine color by magnitude
                    if abs(inn) < std_inn:
                        color = '#4CAF50'  # Green for normal
                    elif abs(inn) < 2*std_inn:
                        color = '#FFA726'  # Orange for elevated
                    else:
                        color = '#F44336'  # Red for high

                    # Accepted: normal display, higher z-order
                    ax2.scatter(d, inn, c=color, s=30,
                               marker='o', alpha=1, zorder=20)

            ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
            
            # Mark filter reinitializations
            for reinit_date in reinit_dates:
                ax2.axvline(x=reinit_date, color='purple', linestyle='--', 
                           alpha=0.5, linewidth=1.5, label='Filter Reset' if reinit_date == reinit_dates[0] else None)

            # Add control limits
            ax2.axhline(y=2*std_inn, color='#F44336', linestyle='--',
                       alpha=0.3, linewidth=1, label='±2σ')
            ax2.axhline(y=-2*std_inn, color='#F44336', linestyle='--',
                       alpha=0.3, linewidth=1)

            ax2.fill_between(k_dates, [-2*std_inn]*len(k_dates),
                           [2*std_inn]*len(k_dates),
                           alpha=0.05, color='red')

            # Statistics text
            mean_inn = np.mean(innovations)
            num_rejected = sum(1 for a in accepted if not a)
            stats_text = f'Mean: {mean_inn:.3f} kg\nStd: {std_inn:.3f} kg'
            if num_rejected > 0:
                stats_text += f'\nRejected: {num_rejected}'

            ax2.text(0.02, 0.98, stats_text,
                    transform=ax2.transAxes, fontsize=10,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            # Add legend for markers if there are rejections
            if num_rejected > 0:
                # Show rejected with red outline
                ax2.scatter([], [], marker='o', c='#4CAF50', s=30,
                          edgecolors='red', linewidth=2.5,
                          label='Rejected')
                ax2.scatter([], [], marker='o', c='#4CAF50', s=30,
                          label='Normal')
                ax2.scatter([], [], marker='o', c='#FFA726', s=30,
                           label='Elevated')
                # Move legend to avoid data overlap
                ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left',
                          fontsize=9, frameon=True, borderaxespad=0)

            ax2.set_title('Kalman Filter Residuals (Innovation)',
                          fontsize=12, fontweight='bold', pad=15)
            ax2.set_ylabel('Residual (kg)', fontsize=11, labelpad=10)
            ax2.grid(True, alpha=0.2, linestyle='--')
            format_date_axis(ax2, k_dates)
    else:
        ax2.text(0.5, 0.5, 'Kalman filter not available\n(requires baseline establishment)',
                ha='center', va='center', fontsize=11, color='gray')
        ax2.set_title('Kalman Filter Residuals', fontsize=12, fontweight='bold')

    ax2.set_xlabel('Date', fontsize=11)

    # ========== 4. STATISTICS PANEL ==========

    ax3.axis('off')
    stats_text = create_statistics_text(data)
    ax3.text(0.05, 0.95, stats_text, transform=ax3.transAxes,
            fontsize=10, verticalalignment='top',
            fontfamily='monospace')
    ax3.set_title('Key Metrics', fontsize=12, fontweight='bold', loc='left', pad=10)

    # ========== 5. WEIGHT DISTRIBUTION ==========

    if len(weights) > 10:
        ax4.hist(weights, bins=min(20, len(weights)//5),
                color='#2196F3', alpha=0.7, edgecolor='black')

        # Add baseline marker if available
        if data.get('baseline_weight'):
            ax4.axvline(x=data['baseline_weight'], color='#9C27B0',
                       linestyle='--', linewidth=2, label='Baseline')

        # Add current weight marker
        ax4.axvline(x=weights[-1], color='#4CAF50',
                   linestyle='-', linewidth=2, label='Current')

        ax4.set_xlabel('Weight (kg)', fontsize=11, labelpad=8)
        ax4.set_ylabel('Frequency', fontsize=11, labelpad=8)
        ax4.set_title('Weight Distribution', fontsize=12, fontweight='bold', pad=15)
        # Place legend in upper corner to avoid histogram bars
        ax4.legend(loc='upper right', fontsize=9, frameon=True, fancybox=True)
        ax4.grid(True, alpha=0.2, axis='y')
    else:
        ax4.text(0.5, 0.5, 'Insufficient data\nfor distribution',
                ha='center', va='center', fontsize=11, color='gray')
        ax4.set_title('Weight Distribution', fontsize=12, fontweight='bold')
        ax4.axis('off')

    # Overall title
    fig.suptitle('Enhanced Weight Analysis Dashboard',
                fontsize=16, fontweight='bold', y=0.98)

    # Add timestamp
    fig.text(0.99, 0.01, f'Generated: {datetime.now():%Y-%m-%d %H:%M}',
            ha='right', va='bottom', fontsize=8, alpha=0.5)

    # Save figure with adjusted layout for legends
    plt.tight_layout(rect=[0, 0.02, 0.92, 0.96])  # Leave more space on right for legends
    output_file = output_dir / f"{user_id}.png"
    plt.savefig(output_file, dpi=100, bbox_inches='tight', facecolor='white')
    plt.close()

    logger.info(f"Created enhanced dashboard for user {user_id[:8]}...")


def format_date_axis(ax, dates: List[datetime]):
    """Format date axis based on time span."""
    if not dates:
        return

    total_days = (dates[-1] - dates[0]).days if len(dates) > 1 else 1

    if total_days <= 30:
        ax.xaxis.set_major_locator(mdates.WeekdayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    elif total_days <= 90:
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    elif total_days <= 180:
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    elif total_days <= 365:
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    else:
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1,4,7,10]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=9)


def create_statistics_text(data: Dict[str, Any]) -> str:
    """Create formatted statistics text for the panel."""
    lines = []

    # Basic statistics
    lines.append("═" * 35)
    lines.append("BASELINE METRICS")
    lines.append("─" * 35)

    # Check for multiple baselines (gap-triggered)
    baseline_history = data.get('baseline_history', [])
    if baseline_history:
        lines.append(f"Baselines:   {len(baseline_history)} established")
        # Show most recent baseline
        latest = baseline_history[-1]
        lines.append(f"Latest:      {latest['weight']:.1f} kg")
        lines.append(f"Confidence:  {latest.get('confidence', 'N/A').upper()}")
        lines.append(f"Readings:    {latest.get('readings_used', 0)}")
        if latest.get('gap_days', 0) > 0:
            lines.append(f"Gap Trigger: {latest['gap_days']} days")
    elif data.get('baseline_established'):
        lines.append(f"Status:      ✓ Established")
        lines.append(f"Weight:      {data.get('baseline_weight', 'N/A'):.1f} kg")
        lines.append(f"Confidence:  {data.get('baseline_confidence', 'N/A').upper()}")
        lines.append(f"Readings:    {data.get('baseline_readings_count', 0)}")
        lines.append(f"Outliers:    {data.get('baseline_outliers_removed', 0)} removed")
        if data.get('baseline_std'):
            lines.append(f"Std Dev:     {data.get('baseline_std', 0):.2f} kg")
        if data.get('baseline_mad'):
            lines.append(f"MAD:         {data.get('baseline_mad', 0):.2f} kg")
    else:
        lines.append(f"Status:      ✗ Not Established")
        if data.get('baseline_error'):
            lines.append(f"Reason:      {data['baseline_error'][:25]}")

    lines.append("")
    lines.append("═" * 35)
    lines.append("DATA QUALITY")
    lines.append("─" * 35)

    if data.get('total_readings'):
        lines.append(f"Total:       {data['total_readings']} readings")
        lines.append(f"Outliers:    {data.get('outliers', 0)} ({data.get('outliers', 0)/data['total_readings']*100:.1f}%)")
        lines.append(f"Confidence:  {data.get('average_confidence', 0):.2f} avg")
        lines.append(f"Date Range:  {data.get('days_spanned', 0)} days")

    if data.get('weight_mean'):
        lines.append("")
        lines.append("═" * 35)
        lines.append("WEIGHT STATISTICS")
        lines.append("─" * 35)
        lines.append(f"Current:     {data.get('time_series', [{}])[-1].get('weight', 'N/A'):.1f} kg")
        lines.append(f"Mean:        {data['weight_mean']:.1f} kg")
        lines.append(f"Std Dev:     {data.get('weight_std', 0):.2f} kg")
        lines.append(f"Range:       {data.get('weight_range', 0):.1f} kg")

        if data.get('baseline_weight'):
            current = data.get('time_series', [{}])[-1].get('weight', 0)
            baseline = data['baseline_weight']
            change = current - baseline
            change_pct = (change / baseline) * 100 if baseline > 0 else 0
            lines.append(f"Δ Baseline:  {change:+.1f} kg ({change_pct:+.1f}%)")

    if data.get('kalman_filter_initialized'):
        lines.append("")
        lines.append("═" * 35)
        lines.append("KALMAN FILTER")
        lines.append("─" * 35)
        lines.append(f"Status:      ✓ Active")
        if data.get('kalman_summary', {}).get('final_filtered_weight'):
            lines.append(f"Filtered:    {data['kalman_summary']['final_filtered_weight']:.1f} kg")
        if data.get('kalman_summary', {}).get('mean_trend_kg_per_day') is not None:
            trend = data['kalman_summary']['mean_trend_kg_per_day']
            lines.append(f"Trend:       {trend:+.3f} kg/day")
            lines.append(f"             {trend*7:+.2f} kg/week")

    # Add validation metrics if available
    if data.get('kalman_summary', {}).get('validation'):
        validation = data['kalman_summary']['validation']
        metrics = validation.get('metrics', {})
        lines.append("")
        lines.append("═" * 35)
        lines.append("VALIDATION GATE")
        lines.append("─" * 35)
        lines.append(f"Rejected:    {validation.get('total_rejected', 0)} measurements")
        lines.append(f"Accept Rate: {metrics.get('acceptance_rate', 0)*100:.1f}%")
        lines.append(f"Health:      {metrics.get('health_score', 0):.2f}")

        # Show most common rejection reason if any
        common_rejection = metrics.get('most_common_rejection')
        if common_rejection and isinstance(common_rejection, (list, tuple)) and common_rejection[1] > 0:
            reason = common_rejection[0].replace('_', ' ').title()
            count = common_rejection[1]
            lines.append(f"Top Reason:  {reason} ({count})")

    return '\n'.join(lines)
