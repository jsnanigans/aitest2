#!/usr/bin/env python3
"""
Analysis of rejection patterns to identify configuration issues
"""

import json
import glob
from pathlib import Path
from collections import defaultdict
import statistics

def analyze_rejections():
    results_dir = Path("output/results")
    
    # Aggregate statistics
    total_users = 0
    initialized_users = 0
    total_readings = 0
    total_accepted = 0
    total_rejected = 0
    
    # Outlier type breakdown
    outlier_counts = defaultdict(int)
    
    # Per-user statistics
    user_stats = []
    
    # Process all user files
    for user_file in glob.glob(str(results_dir / "user_*.json")):
        with open(user_file) as f:
            data = json.load(f)
        
        total_users += 1
        
        if not data.get('initialized', False):
            continue
            
        initialized_users += 1
        stats = data['stats']
        
        # Aggregate counts
        total_readings += stats['total_readings']
        total_accepted += stats['accepted_readings']
        total_rejected += stats['rejected_readings']
        
        # Count outlier types
        for outlier_type, count in stats.get('outliers_by_type', {}).items():
            outlier_counts[outlier_type] += count
        
        # Track per-user acceptance rate
        if stats['total_readings'] > 0:
            acceptance_rate = stats['accepted_readings'] / stats['total_readings']
            user_stats.append({
                'user_id': data['user_id'][:8],
                'total': stats['total_readings'],
                'accepted': stats['accepted_readings'],
                'rejected': stats['rejected_readings'],
                'acceptance_rate': acceptance_rate,
                'baseline_confidence': data['baseline'].get('confidence', 'unknown')
            })
    
    # Print analysis
    print("=" * 60)
    print("REJECTION ANALYSIS REPORT")
    print("=" * 60)
    
    print(f"\n📊 OVERVIEW:")
    print(f"  Total users processed: {total_users}")
    print(f"  Users initialized: {initialized_users} ({initialized_users/total_users:.1%})")
    print(f"  Total readings: {total_readings}")
    print(f"  Accepted: {total_accepted} ({total_accepted/total_readings:.1%})")
    print(f"  Rejected: {total_rejected} ({total_rejected/total_readings:.1%})")
    
    print(f"\n❌ REJECTION BREAKDOWN:")
    for outlier_type, count in sorted(outlier_counts.items(), key=lambda x: x[1], reverse=True):
        pct = count / total_rejected * 100 if total_rejected > 0 else 0
        print(f"  {outlier_type}: {count} ({pct:.1f}%)")
    
    # Analyze per-user patterns
    if user_stats:
        acceptance_rates = [u['acceptance_rate'] for u in user_stats]
        
        print(f"\n👤 PER-USER STATISTICS:")
        print(f"  Average acceptance rate: {statistics.mean(acceptance_rates):.1%}")
        print(f"  Median acceptance rate: {statistics.median(acceptance_rates):.1%}")
        print(f"  Std dev: {statistics.stdev(acceptance_rates) if len(acceptance_rates) > 1 else 0:.1%}")
        
        # Show worst performers
        worst_users = sorted(user_stats, key=lambda x: x['acceptance_rate'])[:5]
        print(f"\n  Users with lowest acceptance rates:")
        for u in worst_users:
            print(f"    {u['user_id']}: {u['acceptance_rate']:.1%} ({u['accepted']}/{u['total']}) - baseline: {u['baseline_confidence']}")
        
        # Show by baseline confidence
        by_confidence = defaultdict(list)
        for u in user_stats:
            by_confidence[u['baseline_confidence']].append(u['acceptance_rate'])
        
        print(f"\n  Acceptance rate by baseline confidence:")
        for conf, rates in by_confidence.items():
            if rates:
                print(f"    {conf}: avg={statistics.mean(rates):.1%}, n={len(rates)}")
    
    print(f"\n⚠️  KEY ISSUES IDENTIFIED:")
    
    issues = []
    
    # Check initialization rate
    init_rate = initialized_users / total_users if total_users > 0 else 0
    if init_rate < 0.5:
        issues.append(f"Only {init_rate:.1%} of users initialized (should be >90%)")
    
    # Check overall acceptance rate
    if total_readings > 0:
        acc_rate = total_accepted / total_readings
        if acc_rate < 0.8:
            issues.append(f"Acceptance rate {acc_rate:.1%} is too low (target: 87%+)")
    
    # Check ARIMA outliers
    arima_outliers = outlier_counts.get('additive_outlier', 0)
    if total_rejected > 0:
        arima_pct = arima_outliers / total_rejected
        if arima_pct > 0.5:
            issues.append(f"ARIMA detecting {arima_pct:.0%} of rejections as outliers (too sensitive)")
    
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
    
    print(f"\n💡 RECOMMENDATIONS:")
    print("  1. Increase ARIMA residual_threshold from 3.0 to 4.0-5.0")
    print("  2. Improve baseline establishment (reduce min_readings requirement)")
    print("  3. Implement data deduplication for same-date readings")
    print("  4. Add debug logging to track rejection points in pipeline")
    print("  5. Consider relaxing Layer1 MAD threshold from 3.0 to 4.0")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    analyze_rejections()