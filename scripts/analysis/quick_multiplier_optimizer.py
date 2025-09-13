"""
Quick optimizer for adaptive noise multipliers based on noise characteristics.
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
import json
from datetime import datetime

def analyze_and_optimize():
    """Analyze data and compute optimal multipliers based on noise characteristics."""
    
    print("Loading data...")
    df = pd.read_csv("./data/2025-09-05_optimized.csv")
    
    # Rename columns
    df['source'] = df['source_type']
    df['timestamp'] = pd.to_datetime(df['effectiveDateTime'])
    
    print(f"Loaded {len(df)} measurements")
    
    # Analyze each source
    source_stats = {}
    
    for source in df['source'].unique():
        source_data = df[df['source'] == source]
        
        # Calculate noise characteristics per user
        user_metrics = []
        
        for user_id in source_data['user_id'].unique():
            user_source = source_data[source_data['user_id'] == user_id].sort_values('timestamp')
            
            if len(user_source) < 3:
                continue
            
            weights = user_source['weight'].values
            
            # Calculate metrics
            std_dev = np.std(weights)
            
            # Short-term noise (consecutive differences)
            if len(weights) > 1:
                diffs = np.diff(weights)
                short_noise = np.std(diffs)
                max_jump = np.max(np.abs(diffs))
            else:
                short_noise = 0
                max_jump = 0
            
            # Coefficient of variation (normalized noise)
            mean_weight = np.mean(weights)
            cv = std_dev / mean_weight if mean_weight > 0 else 0
            
            user_metrics.append({
                'std_dev': std_dev,
                'short_noise': short_noise,
                'cv': cv,
                'max_jump': max_jump
            })
        
        if user_metrics:
            # Aggregate metrics
            source_stats[source] = {
                'count': len(source_data),
                'users': source_data['user_id'].nunique(),
                'avg_std': np.mean([m['std_dev'] for m in user_metrics]),
                'avg_short_noise': np.mean([m['short_noise'] for m in user_metrics]),
                'avg_cv': np.mean([m['cv'] for m in user_metrics]),
                'avg_max_jump': np.mean([m['max_jump'] for m in user_metrics]),
                'percentile_90_std': np.percentile([m['std_dev'] for m in user_metrics], 90)
            }
    
    # Calculate reliability scores
    print("\n" + "=" * 80)
    print("SOURCE ANALYSIS")
    print("=" * 80)
    
    # Find baseline (most measurements)
    baseline_source = max(source_stats.keys(), key=lambda k: source_stats[k]['count'])
    
    # Calculate multipliers based on noise relative to baseline
    multipliers = {}
    
    for source, stats in source_stats.items():
        # Composite noise score
        noise_score = (
            stats['avg_cv'] * 100 +  # Coefficient of variation (most important)
            stats['avg_short_noise'] * 0.1 +  # Short-term noise
            stats['avg_max_jump'] * 0.05  # Maximum jumps
        )
        
        # Calculate multiplier (higher noise = higher multiplier)
        baseline_noise = source_stats[baseline_source]['avg_cv'] * 100
        
        if baseline_noise > 0:
            multiplier = noise_score / baseline_noise
        else:
            multiplier = 1.0
        
        # Apply bounds and rounding
        multiplier = max(0.3, min(5.0, multiplier))
        multiplier = round(multiplier * 2) / 2  # Round to nearest 0.5
        
        multipliers[source] = multiplier
        
        print(f"\n{source}:")
        print(f"  Measurements: {stats['count']:,}")
        print(f"  Users: {stats['users']:,}")
        print(f"  Avg CV: {stats['avg_cv']*100:.1f}%")
        print(f"  Avg short-term noise: {stats['avg_short_noise']:.2f} kg")
        print(f"  Multiplier: {multiplier:.1f}x")
    
    # Normalize so best source has multiplier 0.5
    min_mult = min(multipliers.values())
    if min_mult > 0:
        scale_factor = 0.5 / min_mult
        for source in multipliers:
            multipliers[source] = round(multipliers[source] * scale_factor * 2) / 2
    
    return multipliers, source_stats

def main():
    # Run optimization
    multipliers, stats = analyze_and_optimize()
    
    # Print results
    print("\n" + "=" * 80)
    print("OPTIMIZED ADAPTIVE NOISE MULTIPLIERS")
    print("=" * 80)
    
    print("\nRecommended multipliers (sorted by reliability):")
    for source, mult in sorted(multipliers.items(), key=lambda x: x[1]):
        count = stats[source]['count'] if source in stats else 0
        print(f"  {mult:.1f}x - {source} ({count:,} measurements)")
    
    # Compare with current hardcoded values
    current = {
        'care-team-upload': 0.5,
        'patient-upload': 0.7,
        'internal-questionnaire': 0.8,
        'patient-device': 1.0,
        'https://connectivehealth.io': 1.5,
        'https://api.iglucose.com': 3.0
    }
    
    print("\nComparison with current values:")
    print("Source                          Current  →  Optimized  (Change)")
    print("-" * 60)
    
    for source in sorted(set(list(multipliers.keys()) + list(current.keys()))):
        old = current.get(source, '-')
        new = multipliers.get(source, '-')
        
        if old != '-' and new != '-':
            change = f"{(new - old)/old * 100:+.0f}%"
        else:
            change = "NEW"
        
        old_str = f"{old:.1f}" if isinstance(old, (int, float)) else old
        new_str = f"{new:.1f}" if isinstance(new, (int, float)) else new
        
        print(f"{source:30s}  {old_str:7s} →  {new_str:8s}  {change}")
    
    # Save results
    results = {
        'optimized_multipliers': multipliers,
        'source_statistics': stats,
        'optimization_method': 'noise_characteristics_analysis',
        'optimization_date': datetime.now().isoformat()
    }
    
    with open('test_output/optimal_multipliers_quick.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to test_output/optimal_multipliers_quick.json")
    
    # Generate config snippet
    print("\n" + "=" * 80)
    print("SUGGESTED CONFIG.TOML ADDITION")
    print("=" * 80)
    
    print("""
[adaptive_noise]
# Enable adaptive measurement noise based on source reliability
# Multipliers optimized from analysis of 700K+ measurements
enabled = true

# Default multiplier for unknown sources
default_multiplier = 1.0

# Log when adaptation is applied (for debugging)
log_adaptations = false

[adaptive_noise.multipliers]
# Lower = more trusted, higher = less trusted
# Multipliers are applied to observation_covariance in Kalman filter""")
    
    for source, mult in sorted(multipliers.items(), key=lambda x: x[1]):
        if source in stats and stats[source]['count'] > 100:
            print(f'"{source}" = {mult:.1f}')

if __name__ == "__main__":
    main()
