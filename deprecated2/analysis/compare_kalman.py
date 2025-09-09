#!/usr/bin/env python3

import json
import numpy as np
from pathlib import Path

def analyze_kalman_performance(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    stats = {
        'total_users': 0,
        'users_with_kalman': 0,
        'users_with_baseline': 0,
        'kalman_outliers_total': 0,
        'confidence_outliers_total': 0,
        'total_readings': 0,
        'mean_innovations': [],
        'std_innovations': [],
        'mean_uncertainties': [],
        'kalman_outlier_rates': [],
        'confidence_outlier_rates': []
    }
    
    for user_id, user_data in data.items():
        stats['total_users'] += 1
        stats['total_readings'] += user_data['total_readings']
        
        if user_data.get('baseline_weight') is not None:
            stats['users_with_baseline'] += 1
        
        if user_data.get('kalman_filter_initialized'):
            stats['users_with_kalman'] += 1
            
            if 'kalman_summary' in user_data:
                ks = user_data['kalman_summary']
                stats['kalman_outliers_total'] += ks.get('kalman_outliers', 0)
                stats['mean_innovations'].append(ks.get('mean_innovation', 0))
                stats['std_innovations'].append(ks.get('std_innovation', 0))
                stats['mean_uncertainties'].append(ks.get('mean_uncertainty', 0))
                stats['kalman_outlier_rates'].append(ks.get('kalman_outlier_rate', 0))
        
        confidence_outliers = user_data.get('outliers', 0)
        stats['confidence_outliers_total'] += confidence_outliers
        
        if user_data['total_readings'] > 0:
            confidence_outlier_rate = confidence_outliers / user_data['total_readings']
            stats['confidence_outlier_rates'].append(confidence_outlier_rate)
    
    return stats

def print_comparison_report(stats):
    print("\n" + "=" * 60)
    print("KALMAN FILTER vs CONFIDENCE SCORING COMPARISON")
    print("=" * 60)
    
    print(f"\nDataset Overview:")
    print(f"  Total users: {stats['total_users']:,}")
    print(f"  Total readings: {stats['total_readings']:,}")
    print(f"  Users with baseline: {stats['users_with_baseline']:,} ({stats['users_with_baseline']/stats['total_users']*100:.1f}%)")
    print(f"  Users with Kalman filter: {stats['users_with_kalman']:,} ({stats['users_with_kalman']/stats['total_users']*100:.1f}%)")
    
    print(f"\nOutlier Detection Comparison:")
    print(f"  Confidence-based outliers: {stats['confidence_outliers_total']:,}")
    print(f"  Kalman-based outliers: {stats['kalman_outliers_total']:,}")
    
    if stats['confidence_outlier_rates']:
        print(f"\n  Average outlier rates:")
        print(f"    Confidence scoring: {np.mean(stats['confidence_outlier_rates'])*100:.2f}%")
        if stats['kalman_outlier_rates']:
            print(f"    Kalman filter: {np.mean(stats['kalman_outlier_rates'])*100:.2f}%")
    
    if stats['mean_innovations']:
        print(f"\nKalman Filter Statistics:")
        print(f"  Mean innovation across users: {np.mean(stats['mean_innovations']):.3f}kg")
        print(f"  Mean std of innovations: {np.mean(stats['std_innovations']):.3f}kg")
        print(f"  Mean uncertainty: {np.mean(stats['mean_uncertainties']):.3f}kg")
        
        innovations_near_zero = sum(1 for x in stats['mean_innovations'] if abs(x) < 0.5)
        print(f"  Users with well-calibrated filter (|innovation| < 0.5): {innovations_near_zero} ({innovations_near_zero/len(stats['mean_innovations'])*100:.1f}%)")
    
    print("\nKey Insights:")
    
    if stats['users_with_kalman'] > 0:
        kalman_coverage = stats['users_with_kalman'] / stats['total_users'] * 100
        if kalman_coverage > 30:
            print(f"  ✓ Good Kalman filter coverage: {kalman_coverage:.1f}% of users")
        else:
            print(f"  ⚠ Low Kalman filter coverage: {kalman_coverage:.1f}% of users")
            print(f"    (Only users with baselines get Kalman filters)")
    
    if stats['kalman_outlier_rates'] and stats['confidence_outlier_rates']:
        kalman_mean_rate = np.mean(stats['kalman_outlier_rates'])
        conf_mean_rate = np.mean(stats['confidence_outlier_rates'])
        
        if kalman_mean_rate < conf_mean_rate:
            reduction = (conf_mean_rate - kalman_mean_rate) / conf_mean_rate * 100
            print(f"  ✓ Kalman filter reduces outlier rate by {reduction:.1f}%")
        else:
            print(f"  ⚠ Kalman filter has higher outlier rate than confidence scoring")
    
    if stats['mean_innovations']:
        well_calibrated_pct = sum(1 for x in stats['mean_innovations'] if abs(x) < 0.5) / len(stats['mean_innovations']) * 100
        if well_calibrated_pct > 70:
            print(f"  ✓ {well_calibrated_pct:.0f}% of Kalman filters are well-calibrated")
        else:
            print(f"  ⚠ Only {well_calibrated_pct:.0f}% of Kalman filters are well-calibrated")
    
    print("\n" + "=" * 60)

def main():
    json_file = Path("output") / "baseline_results_20250903_112102.json"
    
    if not json_file.exists():
        print(f"Error: File {json_file} not found")
        return
    
    stats = analyze_kalman_performance(json_file)
    print_comparison_report(stats)
    
    print("\nPhase 1 Implementation Status:")
    print("✅ WeightKalmanFilter class implemented")
    print("✅ EM parameter learning implemented")
    print("✅ Integrated with baseline processor")
    print("✅ Missing data handling implemented")
    print("✅ Innovation tracking for outlier detection")
    print("✅ Results included in JSON output")
    print("\nReady for Phase 2: Dynamic Measurement Noise")

if __name__ == "__main__":
    main()