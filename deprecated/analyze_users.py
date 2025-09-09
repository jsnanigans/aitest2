#!/usr/bin/env python3

import json
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timedelta
import numpy as np

def analyze_user(user_id, readings):
    readings_sorted = sorted(readings, key=lambda x: x['date'])
    
    weights = [r['weight'] for r in readings_sorted]
    dates = [datetime.fromisoformat(r['date'].replace(' ', 'T')) for r in readings_sorted]
    sources = [r['source'] for r in readings_sorted]
    
    # Calculate time gaps
    gaps = []
    for i in range(1, len(dates)):
        gap_days = (dates[i] - dates[i-1]).days
        if gap_days > 1:
            gaps.append((dates[i-1].isoformat(), dates[i].isoformat(), gap_days))
    
    # Detect outliers (simple method: > 2 std from mean)
    mean_weight = np.mean(weights)
    std_weight = np.std(weights)
    outliers = []
    for i, w in enumerate(weights):
        z_score = abs((w - mean_weight) / std_weight) if std_weight > 0 else 0
        if z_score > 2:
            outliers.append({
                'date': readings_sorted[i]['date'],
                'weight': w,
                'z_score': z_score,
                'deviation_pct': ((w - mean_weight) / mean_weight * 100)
            })
    
    # Check for duplicate/near-duplicate weights on same day
    same_day_weights = defaultdict(list)
    for r in readings_sorted:
        day = r['date'][:10]
        same_day_weights[day].append(r['weight'])
    
    duplicate_days = {day: weights for day, weights in same_day_weights.items() if len(weights) > 1}
    
    # Detect rapid weight changes
    rapid_changes = []
    for i in range(1, len(readings_sorted)):
        time_diff = (dates[i] - dates[i-1]).total_seconds() / 3600  # hours
        weight_diff = abs(weights[i] - weights[i-1])
        if time_diff < 24 and weight_diff > 5:  # More than 5kg change in 24 hours
            rapid_changes.append({
                'from_date': readings_sorted[i-1]['date'],
                'to_date': readings_sorted[i]['date'],
                'from_weight': weights[i-1],
                'to_weight': weights[i],
                'change_kg': weight_diff,
                'hours': time_diff
            })
    
    return {
        'user_id': user_id,
        'total_readings': len(readings),
        'date_range': f"{dates[0].date()} to {dates[-1].date()}",
        'weight_range': f"{min(weights):.1f} - {max(weights):.1f} kg",
        'mean_weight': f"{mean_weight:.1f} kg",
        'std_deviation': f"{std_weight:.1f} kg",
        'sources': list(set(sources)),
        'gaps_over_30_days': len([g for g in gaps if g[2] > 30]),
        'total_gaps': len(gaps),
        'outliers_count': len(outliers),
        'outliers': outliers[:5],  # Show first 5
        'duplicate_days_count': len(duplicate_days),
        'duplicate_days': dict(list(duplicate_days.items())[:5]),  # Show first 5
        'rapid_changes_count': len(rapid_changes),
        'rapid_changes': rapid_changes[:5],  # Show first 5
    }

def main():
    # Read the CSV file
    csv_path = './2025-09-05_optimized.csv'
    
    target_users = [
        '0040872d-333a-4ace-8c5a-b2fcd056e65a',  # madness
        '0675ed39-53be-480b-baa6-fa53fc33709f',  # line frew, 3 person
        '055b0c48-d5b4-44cb-8772-48154999e6c3',  # outliers-same
    ]
    
    user_data = defaultdict(list)
    
    with open(csv_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 5:
                user_id = parts[0]
                if user_id in target_users:
                    user_data[user_id].append({
                        'date': parts[1],
                        'source': parts[2],
                        'weight': float(parts[3]),
                        'unit': parts[4]
                    })
    
    # Analyze each user
    results = {}
    for user_id in target_users:
        if user_id in user_data:
            analysis = analyze_user(user_id, user_data[user_id])
            results[user_id] = analysis
            
            print(f"\n{'='*60}")
            print(f"User: {user_id}")
            if user_id == '0040872d-333a-4ace-8c5a-b2fcd056e65a':
                print("Tag: MADNESS CASE")
            elif user_id == '0675ed39-53be-480b-baa6-fa53fc33709f':
                print("Tag: LINE FREW, 3 PERSON")
            elif user_id == '055b0c48-d5b4-44cb-8772-48154999e6c3':
                print("Tag: OUTLIERS-SAME")
            
            print(f"{'='*60}")
            print(f"Total readings: {analysis['total_readings']}")
            print(f"Date range: {analysis['date_range']}")
            print(f"Weight range: {analysis['weight_range']}")
            print(f"Mean weight: {analysis['mean_weight']}")
            print(f"Std deviation: {analysis['std_deviation']}")
            print(f"Data sources: {', '.join(analysis['sources'])}")
            print(f"\nData Quality Issues:")
            print(f"  - Gaps > 30 days: {analysis['gaps_over_30_days']}")
            print(f"  - Statistical outliers: {analysis['outliers_count']}")
            print(f"  - Days with multiple readings: {analysis['duplicate_days_count']}")
            print(f"  - Rapid weight changes (>5kg in 24h): {analysis['rapid_changes_count']}")
            
            if analysis['outliers']:
                print(f"\n  Sample outliers:")
                for o in analysis['outliers'][:3]:
                    print(f"    {o['date']}: {o['weight']:.1f}kg (z-score: {o['z_score']:.1f}, {o['deviation_pct']:.1f}% from mean)")
            
            if analysis['rapid_changes']:
                print(f"\n  Sample rapid changes:")
                for rc in analysis['rapid_changes'][:3]:
                    print(f"    {rc['from_date'][:10]} → {rc['to_date'][:10]}: {rc['from_weight']:.1f}kg → {rc['to_weight']:.1f}kg ({rc['change_kg']:.1f}kg in {rc['hours']:.1f}h)")
    
    # Save detailed results
    with open('output/challenging_users_analysis.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n\nDetailed analysis saved to output/challenging_users_analysis.json")

if __name__ == "__main__":
    main()