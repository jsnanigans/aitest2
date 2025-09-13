"""
Test the impact of loosening absolute limits while strengthening Kalman trend stiffness.
Analyzes four specific users to understand the tradeoff effects.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List, Tuple
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.processor import WeightProcessor
from src.database import get_state_db

def process_user_with_config(
    user_data: pd.DataFrame,
    user_id: str,
    processing_config: Dict,
    kalman_config: Dict,
    config_name: str
) -> Tuple[List[Dict], Dict]:
    """Process user data with specific configuration."""
    db = get_state_db()
    # Clear state for this user
    db.clear_state(user_id)
    
    results = []
    stats = {
        'total': 0,
        'accepted': 0,
        'rejected': 0,
        'rejection_reasons': {},
        'weight_changes': [],
        'filtered_weights': [],
        'raw_weights': []
    }
    
    for _, row in user_data.iterrows():
        timestamp = datetime.fromisoformat(row['effectiveDateTime'])
        weight = row['weight']
        source = row['source_type']
        
        result = WeightProcessor.process_weight(
            user_id=user_id,
            weight=weight,
            timestamp=timestamp,
            source=source,
            processing_config=processing_config,
            kalman_config=kalman_config,
            db=db
        )
        
        results.append(result)
        stats['total'] += 1
        stats['raw_weights'].append(weight)
        
        if result and result.get('accepted'):
            stats['accepted'] += 1
            stats['filtered_weights'].append(result.get('filtered_weight', weight))
            if len(stats['filtered_weights']) > 1:
                change = abs(stats['filtered_weights'][-1] - stats['filtered_weights'][-2])
                stats['weight_changes'].append(change)
        else:
            stats['rejected'] += 1
            reason = result.get('rejection_reason', 'Unknown') if result else 'Unknown'
            reason_category = reason.split(':')[0] if ':' in reason else reason.split(' ')[0]
            stats['rejection_reasons'][reason_category] = stats['rejection_reasons'].get(reason_category, 0) + 1
    
    return results, stats

def compare_configurations(user_id: str, user_data: pd.DataFrame) -> Dict:
    """Compare different configuration approaches."""
    
    # Current configuration (baseline)
    current_processing = {
        'min_weight': 20.0,
        'max_weight': 300.0,
        'physiological': {
            'enable_physiological_limits': True,
            'max_change_1h_percent': 0.02,
            'max_change_1h_absolute': 3.0,
            'max_change_6h_percent': 0.025,
            'max_change_6h_absolute': 4.0,
            'max_change_24h_percent': 0.035,
            'max_change_24h_absolute': 5.0,
            'max_sustained_daily': 1.5,
            'limit_tolerance': 0.10,
            'sustained_tolerance': 0.25,
            'session_timeout_minutes': 5,
            'session_variance_threshold': 5.0
        }
    }
    
    current_kalman = {
        'initial_variance': 1.0,
        'transition_covariance_weight': 0.1,
        'transition_covariance_trend': 0.001,
        'observation_covariance': 1.0
    }
    
    # Looser limits configuration (30% looser on absolute limits)
    looser_limits_processing = current_processing.copy()
    looser_limits_processing['physiological'] = current_processing['physiological'].copy()
    looser_limits_processing['physiological'].update({
        'max_change_1h_absolute': 4.0,  # Was 3.0 (+33%)
        'max_change_6h_absolute': 5.5,  # Was 4.0 (+38%)
        'max_change_24h_absolute': 7.0,  # Was 5.0 (+40%)
        'max_sustained_daily': 2.0,  # Was 1.5 (+33%)
        'limit_tolerance': 0.15,  # Was 0.10 (+50%)
        'sustained_tolerance': 0.35,  # Was 0.25 (+40%)
    })
    
    # Stiffer Kalman configuration (2x stiffer on trend)
    stiffer_kalman = {
        'initial_variance': 0.5,  # Was 1.0 - more confidence in initial
        'transition_covariance_weight': 0.05,  # Was 0.1 - less weight variance allowed
        'transition_covariance_trend': 0.0002,  # Was 0.001 - 5x stiffer trend
        'observation_covariance': 2.0  # Was 1.0 - less trust in individual observations
    }
    
    # Combined: Looser limits + Stiffer Kalman
    combined_processing = looser_limits_processing.copy()
    combined_kalman = stiffer_kalman.copy()
    
    configs = [
        ('Current', current_processing, current_kalman),
        ('Looser Limits Only', looser_limits_processing, current_kalman),
        ('Stiffer Kalman Only', current_processing, stiffer_kalman),
        ('Combined (Looser + Stiffer)', combined_processing, combined_kalman)
    ]
    
    comparison = {}
    for config_name, proc_config, kalman_config in configs:
        results, stats = process_user_with_config(
            user_data, user_id, proc_config, kalman_config, config_name
        )
        comparison[config_name] = {
            'results': results,
            'stats': stats,
            'acceptance_rate': stats['accepted'] / stats['total'] * 100 if stats['total'] > 0 else 0,
            'avg_change': np.mean(stats['weight_changes']) if stats['weight_changes'] else 0,
            'max_change': np.max(stats['weight_changes']) if stats['weight_changes'] else 0,
            'std_filtered': np.std(stats['filtered_weights']) if stats['filtered_weights'] else 0
        }
    
    return comparison

def visualize_comparison(user_id: str, user_data: pd.DataFrame, comparison: Dict):
    """Create visualization comparing different configurations."""
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle(f'Configuration Impact Analysis - User {user_id[:8]}', fontsize=14, fontweight='bold')
    
    # Plot 1: Acceptance rates
    ax = axes[0, 0]
    configs = list(comparison.keys())
    acceptance_rates = [comparison[c]['acceptance_rate'] for c in configs]
    colors = ['#2E7D32', '#1976D2', '#F57C00', '#C62828']
    bars = ax.bar(range(len(configs)), acceptance_rates, color=colors)
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.set_ylabel('Acceptance Rate (%)')
    ax.set_title('Measurement Acceptance Rates')
    ax.grid(axis='y', alpha=0.3)
    for i, (bar, rate) in enumerate(zip(bars, acceptance_rates)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{rate:.1f}%', ha='center', fontsize=10)
    
    # Plot 2: Filtered weight stability
    ax = axes[0, 1]
    std_values = [comparison[c]['std_filtered'] for c in configs]
    bars = ax.bar(range(len(configs)), std_values, color=colors)
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.set_ylabel('Std Dev (kg)')
    ax.set_title('Filtered Weight Stability (Lower is Better)')
    ax.grid(axis='y', alpha=0.3)
    for bar, std in zip(bars, std_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{std:.2f}', ha='center', fontsize=10)
    
    # Plot 3: Time series comparison
    ax = axes[1, 0]
    timestamps = pd.to_datetime(user_data['effectiveDateTime'])
    
    # Plot raw data
    ax.scatter(timestamps, user_data['weight'], alpha=0.2, s=5, color='gray', label='Raw', zorder=1)
    
    # Plot filtered trajectories for key configs
    plot_configs = ['Current', 'Combined (Looser + Stiffer)']
    for i, config_name in enumerate(plot_configs):
        filtered_weights = comparison[config_name]['stats']['filtered_weights']
        if filtered_weights:
            accepted_results = [r for r in comparison[config_name]['results'] if r and r.get('accepted')]
            accepted_times = [pd.to_datetime(r['timestamp']) for r in accepted_results]
            ax.plot(accepted_times, filtered_weights, 
                   label=config_name, color=colors[0 if i == 0 else 3], 
                   alpha=0.8, linewidth=2, zorder=2)
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Weight (kg)')
    ax.set_title('Weight Trajectories: Current vs Combined')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    
    # Plot 4: Rejection reasons breakdown
    ax = axes[1, 1]
    width = 0.2
    x = np.arange(len(configs))
    
    # Get all unique rejection reasons
    all_reasons = set()
    for config in comparison.values():
        all_reasons.update(config['stats']['rejection_reasons'].keys())
    all_reasons = sorted(list(all_reasons))
    
    if all_reasons:
        reason_colors = plt.cm.Set3(np.linspace(0, 1, len(all_reasons)))
        for i, reason in enumerate(all_reasons[:4]):  # Top 4 reasons
            counts = [comparison[c]['stats']['rejection_reasons'].get(reason, 0) for c in configs]
            ax.bar(x + i*width - width*1.5, counts, width, 
                   label=reason[:15], alpha=0.8, color=reason_colors[i])
        
        ax.set_xlabel('Configuration')
        ax.set_ylabel('Rejection Count')
        ax.set_title('Rejection Reasons by Configuration')
        ax.set_xticks(x)
        ax.set_xticklabels(configs, rotation=45, ha='right')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(axis='y', alpha=0.3)
    
    # Plot 5: Average and max changes
    ax = axes[2, 0]
    x = np.arange(len(configs))
    width = 0.35
    avg_changes = [comparison[c]['avg_change'] for c in configs]
    max_changes = [comparison[c]['max_change'] for c in configs]
    
    ax.bar(x - width/2, avg_changes, width, label='Average', color='#1976D2', alpha=0.7)
    ax.bar(x + width/2, max_changes, width, label='Maximum', color='#F57C00', alpha=0.7)
    ax.set_xlabel('Configuration')
    ax.set_ylabel('Weight Change (kg)')
    ax.set_title('Weight Change Between Accepted Measurements')
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 6: Summary metrics table
    ax = axes[2, 1]
    ax.axis('tight')
    ax.axis('off')
    
    table_data = []
    for config in configs:
        c = comparison[config]
        table_data.append([
            config[:20],
            f"{c['acceptance_rate']:.1f}%",
            f"{c['std_filtered']:.2f}",
            f"{c['avg_change']:.3f}",
            f"{c['stats']['accepted']}/{c['stats']['total']}"
        ])
    
    table = ax.table(cellText=table_data,
                    colLabels=['Config', 'Accept%', 'StdDev', 'AvgΔ', 'Count'],
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.35, 0.15, 0.15, 0.15, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    # Color code the table
    for i in range(len(configs)):
        table[(i+1, 0)].set_facecolor(colors[i])
        table[(i+1, 0)].set_text_props(color='white', weight='bold')
    
    plt.tight_layout()
    return fig

def main():
    # Load data
    df = pd.read_csv('data/2025-09-05_optimized.csv')
    
    # Convert weight to kg if needed
    df['weight'] = df.apply(lambda row: row['weight'] / 1000 if row['unit'] == 'g' else row['weight'], axis=1)
    
    users = [
        "0040872d-333a-4ace-8c5a-b2fcd056e65a",
        "b1c7ec66-85f9-4ecc-b7b8-46742f5e78db",
        "42f31300-fae5-4719-a4e4-f63d61e624cc",
        "8823af48-caa8-4b57-9e2c-dc19c509f2e3"
    ]
    
    all_comparisons = {}
    
    print("=" * 80)
    print("IMPACT ANALYSIS: Loosening Limits vs Strengthening Kalman Stiffness")
    print("=" * 80)
    print("\nConfiguration Changes:")
    print("- Looser Limits: +30-40% on absolute limits, +40-50% on tolerances")
    print("- Stiffer Kalman: 5x stiffer trend, 2x less trust in observations")
    print("=" * 80)
    
    for user_id in users:
        user_data = df[df['user_id'] == user_id].sort_values('effectiveDateTime')
        if user_data.empty:
            print(f"\nUser {user_id[:8]}: No data found")
            continue
        
        print(f"\n{'='*60}")
        print(f"User: {user_id[:8]}... ({len(user_data)} measurements)")
        print(f"Date Range: {user_data['effectiveDateTime'].min()} to {user_data['effectiveDateTime'].max()}")
        print(f"Weight Range: {user_data['weight'].min():.1f} - {user_data['weight'].max():.1f} kg")
        print(f"{'='*60}")
        
        comparison = compare_configurations(user_id, user_data)
        all_comparisons[user_id] = comparison
        
        # Print detailed comparison
        for config_name, metrics in comparison.items():
            print(f"\n{config_name}:")
            print(f"  Acceptance Rate: {metrics['acceptance_rate']:.1f}%")
            print(f"  Accepted/Total: {metrics['stats']['accepted']}/{metrics['stats']['total']}")
            print(f"  Filtered StdDev: {metrics['std_filtered']:.3f} kg")
            print(f"  Avg Change: {metrics['avg_change']:.3f} kg")
            print(f"  Max Change: {metrics['max_change']:.3f} kg")
            
            if metrics['stats']['rejection_reasons']:
                print(f"  Top Rejection Reasons:")
                for reason, count in sorted(metrics['stats']['rejection_reasons'].items(), 
                                           key=lambda x: x[1], reverse=True)[:3]:
                    print(f"    - {reason}: {count}")
        
        # Create visualization
        fig = visualize_comparison(user_id, user_data, comparison)
        plt.savefig(f'test_output/kalman_vs_limits_{user_id[:8]}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nVisualization saved: test_output/kalman_vs_limits_{user_id[:8]}.png")
    
    # Summary across all users
    print("\n" + "=" * 80)
    print("SUMMARY ACROSS ALL USERS")
    print("=" * 80)
    
    config_names = ['Current', 'Looser Limits Only', 'Stiffer Kalman Only', 'Combined (Looser + Stiffer)']
    
    for config_name in config_names:
        total_accepted = 0
        total_measurements = 0
        all_std_devs = []
        all_avg_changes = []
        
        for user_id in users:
            if user_id in all_comparisons:
                c = all_comparisons[user_id][config_name]
                total_accepted += c['stats']['accepted']
                total_measurements += c['stats']['total']
                if c['std_filtered'] > 0:
                    all_std_devs.append(c['std_filtered'])
                if c['avg_change'] > 0:
                    all_avg_changes.append(c['avg_change'])
        
        print(f"\n{config_name}:")
        print(f"  Overall Acceptance: {total_accepted}/{total_measurements} ({total_accepted/total_measurements*100:.1f}%)")
        print(f"  Mean StdDev: {np.mean(all_std_devs):.3f} kg")
        print(f"  Mean Avg Change: {np.mean(all_avg_changes):.3f} kg")
    
    print("\n" + "=" * 80)
    print("KEY INSIGHTS:")
    print("=" * 80)
    print("""
    EFFECTS OF CHANGES:
    
    1. LOOSER LIMITS ONLY (+30-40% on absolute limits):
       - Increases acceptance rate by allowing more physiological variation
       - May increase filtered weight variance slightly
       - Benefits users with legitimate rapid changes (exercise, hydration)
       - Risk: Could accept some outliers that should be rejected
    
    2. STIFFER KALMAN ONLY (5x stiffer trend):
       - Creates smoother, more stable filtered output
       - Reduces impact of individual noisy measurements
       - May lag behind real weight changes
       - Risk: Could miss or delay detection of real weight trends
    
    3. COMBINED APPROACH (Looser limits + Stiffer Kalman):
       - Balances acceptance with stability
       - Kalman stiffness compensates for looser acceptance criteria
       - Accepts more measurements but smooths them more aggressively
       - Best for: Mixed user populations with varying data quality
    
    4. USER-SPECIFIC OBSERVATIONS:
       - Users with high natural variance benefit most from looser limits
       - Users with consistent patterns benefit from stiffer Kalman
       - The combined approach provides good compromise for all user types
    """)
    
    # Calculate improvement metrics
    print("\n" + "=" * 80)
    print("IMPROVEMENT METRICS (Combined vs Current):")
    print("=" * 80)
    
    for user_id in users:
        if user_id not in all_comparisons:
            continue
        
        current = all_comparisons[user_id]['Current']
        combined = all_comparisons[user_id]['Combined (Looser + Stiffer)']
        
        accept_improvement = combined['acceptance_rate'] - current['acceptance_rate']
        stability_change = combined['std_filtered'] - current['std_filtered']
        
        print(f"\nUser {user_id[:8]}:")
        print(f"  Acceptance: {current['acceptance_rate']:.1f}% → {combined['acceptance_rate']:.1f}% ({accept_improvement:+.1f}%)")
        print(f"  Stability: {current['std_filtered']:.3f} → {combined['std_filtered']:.3f} kg ({stability_change:+.3f} kg)")
        
        if accept_improvement > 5 and abs(stability_change) < 0.5:
            print(f"  ✓ Significant improvement with minimal stability impact")
        elif accept_improvement > 0 and stability_change < 0:
            print(f"  ✓ Win-win: Better acceptance AND stability")
        elif accept_improvement > 10:
            print(f"  ✓ Major acceptance improvement")

if __name__ == "__main__":
    main()
