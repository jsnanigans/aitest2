"""
Analyze source noise characteristics to determine ideal adaptive multipliers.
"""

import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import json
from typing import Dict, List, Tuple

def load_and_analyze_data(csv_file: str) -> Dict:
    """Load data and analyze noise characteristics per source."""
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    
    # Convert timestamp
    df['timestamp'] = pd.to_datetime(df['effectiveDateTime'])
    df['source'] = df['source_type']  # Rename for consistency
    
    # Group by user and source
    print(f"Total measurements: {len(df)}")
    print(f"Unique users: {df['user_id'].nunique()}")
    print(f"Unique sources: {df['source'].nunique()}")
    
    # Analyze each source
    source_stats = {}
    sources = df['source'].unique()
    
    for source in sources:
        source_data = df[df['source'] == source].copy()
        
        stats = {
            'count': len(source_data),
            'users': source_data['user_id'].nunique(),
            'measurements': []
        }
        
        # For each user, analyze measurement patterns
        user_noise_stats = []
        
        for user_id in source_data['user_id'].unique():
            user_source_data = source_data[source_data['user_id'] == user_id].sort_values('timestamp')
            
            if len(user_source_data) < 3:
                continue
            
            weights = user_source_data['weight'].values
            timestamps = user_source_data['timestamp'].values
            
            # Calculate noise metrics
            # 1. Standard deviation (overall variability)
            std_dev = np.std(weights)
            
            # 2. Short-term noise (difference between consecutive measurements)
            if len(weights) > 1:
                diffs = np.diff(weights)
                short_term_noise = np.std(diffs)
            else:
                short_term_noise = 0
            
            # 3. Outlier rate (measurements > 2 std from mean)
            mean_weight = np.mean(weights)
            if std_dev > 0:
                outliers = np.abs(weights - mean_weight) > 2 * std_dev
                outlier_rate = np.sum(outliers) / len(weights)
            else:
                outlier_rate = 0
            
            # 4. Maximum deviation
            max_deviation = np.max(np.abs(weights - mean_weight)) if len(weights) > 0 else 0
            
            # 5. Measurement frequency (avg time between measurements in days)
            if len(timestamps) > 1:
                time_diffs = np.diff(timestamps) / np.timedelta64(1, 'D')
                avg_frequency = np.mean(time_diffs)
            else:
                avg_frequency = 0
            
            user_noise_stats.append({
                'std_dev': std_dev,
                'short_term_noise': short_term_noise,
                'outlier_rate': outlier_rate,
                'max_deviation': max_deviation,
                'avg_frequency_days': avg_frequency,
                'num_measurements': len(weights)
            })
        
        if user_noise_stats:
            # Aggregate statistics across users
            stats['avg_std_dev'] = np.mean([s['std_dev'] for s in user_noise_stats])
            stats['median_std_dev'] = np.median([s['std_dev'] for s in user_noise_stats])
            stats['avg_short_term_noise'] = np.mean([s['short_term_noise'] for s in user_noise_stats])
            stats['avg_outlier_rate'] = np.mean([s['outlier_rate'] for s in user_noise_stats])
            stats['avg_max_deviation'] = np.mean([s['max_deviation'] for s in user_noise_stats])
            stats['avg_frequency_days'] = np.mean([s['avg_frequency_days'] for s in user_noise_stats])
            
            # Calculate reliability score (lower is better)
            # Combine multiple factors
            reliability_score = (
                stats['avg_std_dev'] * 0.3 +  # Overall variability
                stats['avg_short_term_noise'] * 0.3 +  # Measurement-to-measurement noise
                stats['avg_outlier_rate'] * 100 * 0.2 +  # Outlier frequency
                stats['avg_max_deviation'] * 0.2  # Extreme deviations
            )
            stats['reliability_score'] = reliability_score
        else:
            stats['reliability_score'] = float('inf')
        
        source_stats[source] = stats
    
    return source_stats

def calculate_relative_multipliers(source_stats: Dict) -> Dict:
    """Calculate relative noise multipliers based on reliability scores."""
    
    # Find the most reliable source (lowest score)
    valid_sources = {k: v for k, v in source_stats.items() 
                     if v.get('reliability_score') != float('inf') and v['count'] > 10}
    
    if not valid_sources:
        return {}
    
    best_source = min(valid_sources.keys(), key=lambda k: valid_sources[k]['reliability_score'])
    best_score = valid_sources[best_source]['reliability_score']
    
    multipliers = {}
    
    for source, stats in valid_sources.items():
        # Calculate multiplier relative to best source
        # Higher reliability score = higher multiplier (less trusted)
        if best_score > 0:
            multiplier = stats['reliability_score'] / best_score
            # Clamp to reasonable range
            multiplier = max(0.3, min(5.0, multiplier))
        else:
            multiplier = 1.0
        
        multipliers[source] = round(multiplier, 2)
    
    return multipliers

def print_analysis_report(source_stats: Dict, multipliers: Dict):
    """Print detailed analysis report."""
    print("\n" + "=" * 80)
    print("SOURCE NOISE CHARACTERISTICS ANALYSIS")
    print("=" * 80)
    
    # Sort by reliability score
    sorted_sources = sorted(
        [(k, v) for k, v in source_stats.items() if v.get('reliability_score') != float('inf')],
        key=lambda x: x[1]['reliability_score']
    )
    
    for source, stats in sorted_sources:
        print(f"\n{source}:")
        print(f"  Measurements: {stats['count']:,}")
        print(f"  Users: {stats['users']:,}")
        print(f"  Avg std dev: {stats.get('avg_std_dev', 0):.3f} kg")
        print(f"  Short-term noise: {stats.get('avg_short_term_noise', 0):.3f} kg")
        print(f"  Outlier rate: {stats.get('avg_outlier_rate', 0):.1%}")
        print(f"  Max deviation: {stats.get('avg_max_deviation', 0):.2f} kg")
        print(f"  Measurement frequency: {stats.get('avg_frequency_days', 0):.1f} days")
        print(f"  Reliability score: {stats.get('reliability_score', 0):.3f}")
        if source in multipliers:
            print(f"  Suggested multiplier: {multipliers[source]}")
    
    print("\n" + "-" * 40)
    print("RECOMMENDED MULTIPLIERS:")
    print("-" * 40)
    
    for source, mult in sorted(multipliers.items(), key=lambda x: x[1]):
        print(f"  {source}: {mult}")

def main():
    # Load and analyze data
    csv_file = "./data/2025-09-05_optimized.csv"
    source_stats = load_and_analyze_data(csv_file)
    
    # Calculate multipliers
    multipliers = calculate_relative_multipliers(source_stats)
    
    # Print report
    print_analysis_report(source_stats, multipliers)
    
    # Save results
    results = {
        'source_stats': source_stats,
        'recommended_multipliers': multipliers,
        'analysis_date': datetime.now().isoformat()
    }
    
    with open('test_output/source_noise_analysis.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to test_output/source_noise_analysis.json")

if __name__ == "__main__":
    main()
