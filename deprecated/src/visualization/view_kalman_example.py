#!/usr/bin/env python3
"""Quick visualization to demonstrate Kalman filter effect."""

import json
from pathlib import Path

def show_kalman_example():
    # Load the most recent baseline results
    output_dir = Path('output')
    baseline_files = list(output_dir.glob('baseline_results_*.json'))
    
    if not baseline_files:
        print("No baseline results found")
        return
    
    results_file = max(baseline_files, key=lambda p: p.stat().st_mtime)
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    # Find users with Kalman filters and good data
    kalman_users = []
    for user_id, user_data in data.items():
        if (user_data.get('kalman_filter_initialized') and 
            user_data.get('total_readings', 0) >= 20 and
            'kalman_summary' in user_data):
            kalman_users.append((user_id, user_data))
    
    if not kalman_users:
        print("No users with Kalman filters found")
        return
    
    # Sort by number of readings
    kalman_users.sort(key=lambda x: x[1]['total_readings'], reverse=True)
    
    print("=" * 70)
    print("KALMAN FILTER VISUALIZATION SUMMARY")
    print("=" * 70)
    print(f"\nFound {len(kalman_users)} users with Kalman filters")
    print("\nTop 3 users with most data:\n")
    
    for user_id, user_data in kalman_users[:3]:
        print(f"User {user_id[:8]}...")
        print(f"  Total readings: {user_data['total_readings']}")
        print(f"  Baseline weight: {user_data.get('baseline_weight', 'N/A'):.1f} kg")
        
        ks = user_data['kalman_summary']
        print(f"  Kalman final weight: {ks.get('final_filtered_weight', 'N/A'):.1f} kg")
        print(f"  Mean uncertainty: ±{ks.get('mean_uncertainty', 0):.2f} kg")
        print(f"  Kalman outliers: {ks.get('kalman_outliers', 0)} ({ks.get('kalman_outlier_rate', 0)*100:.1f}%)")
        
        # Show comparison for last few readings
        if 'time_series' in user_data:
            ts_with_kalman = [ts for ts in user_data['time_series'] if ts.get('kalman_filtered') is not None]
            if len(ts_with_kalman) >= 5:
                print(f"\n  Last 5 measurements vs Kalman filtered:")
                print(f"  {'Date':<12} {'Measured':>10} {'Filtered':>10} {'Diff':>8}")
                print(f"  {'-'*40}")
                for ts in ts_with_kalman[-5:]:
                    date = ts['date'].split('T')[0]
                    measured = ts['weight']
                    filtered = ts['kalman_filtered']
                    diff = filtered - measured
                    print(f"  {date:<12} {measured:>10.2f} {filtered:>10.2f} {diff:>+8.2f}")
        
        print()
    
    print("\n" + "=" * 70)
    print("VISUALIZATION FILES CREATED")
    print("=" * 70)
    print(f"\nCheck the output/graphs_debug/ directory for:")
    print("  • Individual user graphs with Kalman filter overlay (green line)")
    print("  • Kalman uncertainty bands (±2σ in light green)")
    print("  • Baseline reference lines (purple dashed)")
    print("\nThe Kalman filter line shows:")
    print("  • Smoother weight trajectory (filters out noise)")
    print("  • Uncertainty bands that grow during data gaps")
    print("  • Better trend detection than raw measurements")

if __name__ == "__main__":
    show_kalman_example()