"""
Kalman Filter Processing Evaluation Dashboard
Focuses on evaluating processor quality and Kalman filter performance
"""

from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np


def create_dashboard(user_id: str, results: list, output_dir: str, viz_config: dict):
    """
    Create a processor evaluation dashboard for a single user.

    Args:
        user_id: User identifier
        results: List of processed weight measurements
        output_dir: Directory to save the dashboard
        viz_config: Visualization configuration dictionary
    """
    all_results = [r for r in results if r]
    valid_results = [r for r in all_results if r.get("accepted")]
    rejected_results = [r for r in all_results if not r.get("accepted")]

    if not all_results:
        return

    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(
        f"Kalman Filter Processing Evaluation - User {user_id[:8]}",
        fontsize=16,
        fontweight="bold",
    )

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

    ax1 = plt.subplot(3, 4, (1, 4))

    if valid_results:
        ax1.plot(valid_timestamps, filtered_weights, '-', linewidth=3,
                color='#1565C0', label='Kalman Filtered', zorder=2)
        ax1.scatter(valid_timestamps, valid_raw_weights, alpha=0.7, s=50,
                   color='#2E7D32', label='Raw Accepted', zorder=5, edgecolors='white', linewidth=0.5)

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
        ax1.scatter(rejected_timestamps, rejected_raw_weights,
                   marker='x', color='#D32F2F', s=80, alpha=0.9, linewidth=2.5,
                   label=f'Rejected ({len(rejected_results)})', zorder=6)

    if filtered_weights:
        baseline = np.median(filtered_weights[: min(10, len(filtered_weights))])
        ax1.axhline(baseline, color='#F57C00', linestyle='--', alpha=0.6, linewidth=2,
                   label=f'Baseline: {baseline:.1f}kg')

    ax1.set_xlabel("Date", fontsize=11)
    ax1.set_ylabel("Weight (kg)", fontsize=11)
    ax1.set_title("Kalman Filter Output vs Raw Data", fontsize=12, fontweight='bold')
    ax1.legend(loc="best", ncol=3)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    if valid_results:
        innovations = [r.get("innovation", 0) for r in valid_results]
        normalized_innovations = [r.get("normalized_innovation", 0) for r in valid_results]

        ax2 = plt.subplot(3, 4, 5)
        ax2.plot(valid_timestamps, innovations, 'o-', markersize=5, alpha=0.8, color='#0288D1', linewidth=1.5)
        ax2.axhline(0, color='black', linestyle='-', alpha=0.5, linewidth=1)
        ax2.fill_between(valid_timestamps, 0, innovations, alpha=0.4, color='#81D4FA')
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Innovation (kg)")
        ax2.set_title("Kalman Innovation\n(Measurement - Prediction)")
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

        ax3 = plt.subplot(3, 4, 6)
        colors = ['#2E7D32' if ni <= 2 else '#FFA726' if ni <= 3 else '#D32F2F'
                 for ni in normalized_innovations]
        ax3.scatter(valid_timestamps, normalized_innovations, c=colors, alpha=0.8, s=40, edgecolors='white', linewidth=0.5)
        ax3.axhline(2, color='#2E7D32', linestyle='--', alpha=0.6, linewidth=1.5, label='Good (≤2σ)')
        ax3.axhline(3, color='#FF6F00', linestyle='--', alpha=0.6, linewidth=1.5, label='Warning (3σ)')
        ax3.axhline(5, color='#D32F2F', linestyle='--', alpha=0.6, linewidth=1.5, label='Extreme (5σ)')
        ax3.set_xlabel("Date")
        ax3.set_ylabel("Normalized Innovation (σ)")
        ax3.set_title("Normalized Innovation\n(Statistical Significance)")
        ax3.legend(loc='best', fontsize=8)
        ax3.grid(True, alpha=0.3)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)

        confidences = [r.get("confidence", 0.5) for r in valid_results]
        ax4 = plt.subplot(3, 4, 7)
        ax4.plot(valid_timestamps, confidences, 'o-', markersize=5, alpha=0.8, color='#6A1B9A', linewidth=1.5)
        ax4.fill_between(valid_timestamps, 0, confidences, alpha=0.4, color='#CE93D8')
        ax4.axhline(0.95, color='#2E7D32', linestyle='--', alpha=0.6, linewidth=1.5, label='High (>0.95)')
        ax4.axhline(0.7, color='#FF6F00', linestyle='--', alpha=0.6, linewidth=1.5, label='Medium (>0.7)')
        ax4.axhline(0.5, color='#D32F2F', linestyle='--', alpha=0.6, linewidth=1.5, label='Low (<0.5)')
        ax4.set_xlabel("Date")
        ax4.set_ylabel("Confidence")
        ax4.set_title("Measurement Confidence\n(Based on Innovation)")
        ax4.set_ylim([0, 1.05])
        ax4.legend(loc='best', fontsize=8)
        ax4.grid(True, alpha=0.3)
        ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)

        trends = [r.get("trend", 0) for r in valid_results]
        weekly_trends = [t * 7 for t in trends]
        ax5 = plt.subplot(3, 4, 8)
        ax5.plot(valid_timestamps, weekly_trends, '-', linewidth=2, color='#1565C0')
        ax5.axhline(0, color='black', linestyle='-', alpha=0.5, linewidth=1)
        ax5.fill_between(valid_timestamps, 0, weekly_trends, alpha=0.4, color='#90CAF9')
        ax5.set_xlabel("Date")
        ax5.set_ylabel("Weekly Trend (kg/week)")
        ax5.set_title("Kalman Trend Component\n(Velocity State)")
        ax5.grid(True, alpha=0.3)
        ax5.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45)

    ax6 = plt.subplot(3, 4, 9)
    if valid_results and len(filtered_weights) > 1:
        residuals = [r - f for r, f in zip(valid_raw_weights, filtered_weights)]
        ax6.hist(residuals, bins=20, edgecolor='#1A237E', alpha=0.8, color='#5C6BC0', linewidth=1.2)
        ax6.axvline(0, color='#D32F2F', linestyle='--', linewidth=2, label=f'Mean: {np.mean(residuals):.2f}')
        ax6.axvline(np.median(residuals), color='#2E7D32', linestyle='--', linewidth=2, label=f'Median: {np.median(residuals):.2f}')
        ax6.set_xlabel("Residual (Raw - Filtered) kg")
        ax6.set_ylabel("Frequency")
        ax6.set_title("Residual Distribution\n(Should be ~Normal)")
        ax6.legend()
        ax6.grid(True, alpha=0.3)

    ax7 = plt.subplot(3, 4, 10)
    if valid_results and len(normalized_innovations) > 1:
        ax7.hist(normalized_innovations, bins=20, edgecolor='#4A148C', alpha=0.8, color='#9C27B0', linewidth=1.2)
        ax7.axvline(1, color='#2E7D32', linestyle='--', linewidth=1.5, label='1σ')
        ax7.axvline(2, color='#FF6F00', linestyle='--', linewidth=1.5, label='2σ')
        ax7.axvline(3, color='#D32F2F', linestyle='--', linewidth=1.5, label='3σ')
        ax7.set_xlabel("Normalized Innovation (σ)")
        ax7.set_ylabel("Frequency")
        ax7.set_title("Innovation Distribution\n(Should peak at ~1σ)")
        ax7.legend()
        ax7.grid(True, alpha=0.3)

    ax8 = plt.subplot(3, 4, 11)
    if valid_results and len(filtered_weights) > 2:
        diffs = np.diff(filtered_weights)
        time_diffs = [(valid_timestamps[i+1] - valid_timestamps[i]).days
                     for i in range(len(valid_timestamps)-1)]
        daily_changes = [d/td if td > 0 else 0 for d, td in zip(diffs, time_diffs)]
        ax8.hist(daily_changes, bins=20, edgecolor='#1B5E20', alpha=0.8, color='#4CAF50', linewidth=1.2)
        ax8.axvline(0, color='black', linestyle='-', linewidth=2, alpha=0.7)
        ax8.set_xlabel("Daily Change (kg/day)")
        ax8.set_ylabel("Frequency")
        ax8.set_title("Daily Change Distribution\n(Filtered Weights)")
        ax8.grid(True, alpha=0.3)

    ax9 = plt.subplot(3, 4, 12)
    stats_text = f"""Processing Statistics
{'='*25}
Total Measurements: {len(all_results)}
Accepted: {len(valid_results)} ({100*len(valid_results)/len(all_results) if all_results else 0:.1f}%)
Rejected: {len(rejected_results)} ({100*len(rejected_results)/len(all_results) if all_results else 0:.1f}%)

Filter Performance
{'='*25}"""

    if valid_results and filtered_weights:
        stats_text += f"""
Mean Innovation: {np.mean([r.get('innovation', 0) for r in valid_results]):.3f} kg
Std Innovation: {np.std([r.get('innovation', 0) for r in valid_results]):.3f} kg
Mean Norm. Innov: {np.mean([r.get('normalized_innovation', 0) for r in valid_results]):.2f}σ
Max Norm. Innov: {max([r.get('normalized_innovation', 0) for r in valid_results]):.2f}σ

Weight Statistics
{'='*25}
Current Weight: {filtered_weights[-1]:.1f} kg
Mean Weight: {np.mean(filtered_weights):.1f} kg
Std Deviation: {np.std(filtered_weights):.2f} kg
Range: {min(filtered_weights):.1f} - {max(filtered_weights):.1f} kg

Trend Analysis
{'='*25}
Current Trend: {trends[-1]*7 if valid_results else 0:.3f} kg/week
Mean Confidence: {np.mean(confidences) if valid_results else 0:.2f}
"""

    rejection_reasons = {}
    for r in rejected_results:
        reason = r.get('reason', 'Unknown')
        rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1

    # if rejection_reasons:
    #     stats_text += f"\nRejection Reasons\n{'='*25}\n"
    #     for reason, count in sorted(rejection_reasons.items(), key=lambda x: x[1], reverse=True):
    #         stats_text += f"{reason[:20]}: {count}\n"

    ax9.text(0.05, 0.95, stats_text, transform=ax9.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace')
    ax9.axis('off')

    plt.tight_layout()

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    filename = output_path / f"{user_id}.png"
    plt.savefig(filename, dpi=viz_config["dashboard_dpi"], bbox_inches="tight")
    plt.close()

    return str(filename)

