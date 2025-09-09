"""
Simple Visualization Dashboard for Weight Data
Creates a multi-panel dashboard for each user
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from pathlib import Path
import numpy as np


def create_dashboard(user_id: str, results: list, output_dir: str):
    """
    Create a dashboard visualization for a single user.
    
    Args:
        user_id: User identifier
        results: List of processed weight measurements
        output_dir: Directory to save the dashboard
    """
    if len(results) < 10:
        return
    
    valid_results = [r for r in results if r and r.get('accepted')]
    if len(valid_results) < 5:
        return
    
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f'Weight Analysis Dashboard - User {user_id[:8]}', fontsize=14, fontweight='bold')
    
    timestamps = [r['timestamp'] for r in valid_results]
    if isinstance(timestamps[0], str):
        timestamps = [datetime.fromisoformat(t.replace('Z', '+00:00')) if isinstance(t, str) else t 
                     for t in timestamps]
    
    raw_weights = [r['raw_weight'] for r in valid_results]
    filtered_weights = [r.get('filtered_weight', r['raw_weight']) for r in valid_results]
    confidences = [r.get('confidence', 0.5) for r in valid_results]
    trends = [r.get('trend', 0) for r in valid_results]
    
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(timestamps, raw_weights, 'o-', alpha=0.3, label='Raw', markersize=4)
    ax1.plot(timestamps, filtered_weights, '-', linewidth=2, label='Filtered', color='red')
    
    if filtered_weights:
        baseline = np.median(filtered_weights[:min(10, len(filtered_weights))])
        ax1.axhline(baseline, color='green', linestyle='--', alpha=0.5, label=f'Baseline: {baseline:.1f}kg')
    
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Weight (kg)')
    ax1.set_title('Weight Trajectory')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    ax2 = plt.subplot(2, 3, 2)
    ax2.hist(filtered_weights, bins=20, edgecolor='black', alpha=0.7)
    ax2.axvline(np.mean(filtered_weights), color='red', linestyle='--', label=f'Mean: {np.mean(filtered_weights):.1f}')
    ax2.axvline(np.median(filtered_weights), color='green', linestyle='--', label=f'Median: {np.median(filtered_weights):.1f}')
    ax2.set_xlabel('Weight (kg)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Weight Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3 = plt.subplot(2, 3, 3)
    colors = ['red' if c < 0.5 else 'yellow' if c < 0.8 else 'green' for c in confidences]
    ax3.scatter(timestamps, confidences, c=colors, alpha=0.6, s=30)
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Confidence')
    ax3.set_title('Measurement Confidence')
    ax3.set_ylim([0, 1.05])
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
    
    ax4 = plt.subplot(2, 3, 4)
    if len(trends) > 1:
        weekly_trends = [t * 7 for t in trends]
        ax4.plot(timestamps, weekly_trends, '-', linewidth=1.5, color='blue')
        ax4.axhline(0, color='black', linestyle='-', alpha=0.3)
        ax4.fill_between(timestamps, 0, weekly_trends, alpha=0.3)
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Weekly Trend (kg/week)')
        ax4.set_title('Weight Change Trend')
        ax4.grid(True, alpha=0.3)
        ax4.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
    
    ax5 = plt.subplot(2, 3, 5)
    if len(filtered_weights) > 7:
        moving_avg = np.convolve(filtered_weights, np.ones(7)/7, mode='valid')
        ma_timestamps = timestamps[3:-3] if len(timestamps) > 7 else timestamps
        ax5.plot(timestamps, filtered_weights, 'o-', alpha=0.3, markersize=3, label='Daily')
        ax5.plot(ma_timestamps, moving_avg, '-', linewidth=2, color='purple', label='7-day MA')
        ax5.set_xlabel('Date')
        ax5.set_ylabel('Weight (kg)')
        ax5.set_title('Moving Average')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        ax5.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45)
    
    ax6 = plt.subplot(2, 3, 6)
    stats_text = f"""Statistics Summary
    
Total Measurements: {len(results)}
Accepted: {len(valid_results)} ({100*len(valid_results)/len(results):.1f}%)
Rejected: {len(results) - len(valid_results)}

Weight Range: {min(filtered_weights):.1f} - {max(filtered_weights):.1f} kg
Current Weight: {filtered_weights[-1]:.1f} kg
Average Weight: {np.mean(filtered_weights):.1f} kg
Std Deviation: {np.std(filtered_weights):.2f} kg

Current Trend: {trends[-1]*7:.2f} kg/week
Avg Confidence: {np.mean(confidences):.2f}
    """
    ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace')
    ax6.axis('off')
    
    plt.tight_layout()
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    filename = output_path / f"dashboard_{user_id}.png"
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close()
    
    return str(filename)