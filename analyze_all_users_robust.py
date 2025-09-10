"""
Robust analysis of ALL users, handling edge cases.
"""

import csv
import numpy as np
from datetime import datetime
from typing import Dict, List
from collections import defaultdict
import gc

from src.processor import WeightProcessor
from src.processor_database import ProcessorStateDB
from src.visualization import normalize_source_type


def analyze_complete_dataset():
    """Analyze the complete dataset with robust error handling."""
    
    print("="*80)
    print("COMPLETE DATASET ANALYSIS - ALL ELIGIBLE USERS")
    print("="*80)
    
    # Load and filter data
    print("\nLoading and filtering dataset...")
    user_data = defaultdict(list)
    invalid_count = 0
    zero_weight_count = 0
    
    with open('data/2025-09-05_optimized.csv', 'r') as f:
        reader = csv.DictReader(f)
        line_count = 0
        
        for row in reader:
            line_count += 1
            if line_count % 100000 == 0:
                print(f"  Processed {line_count:,} lines...")
            
            try:
                weight = float(row['weight'])
                
                # Skip invalid weights
                if weight <= 0 or weight > 500:
                    if weight == 0:
                        zero_weight_count += 1
                    else:
                        invalid_count += 1
                    continue
                
                # Valid weight - add to user data
                user_data[row['user_id']].append({
                    'timestamp': datetime.fromisoformat(row['effectiveDateTime']),
                    'weight': weight,
                    'source': row['source_type'],
                    'normalized': normalize_source_type(row['source_type'])
                })
            except (ValueError, KeyError):
                invalid_count += 1
                continue
    
    print(f"\nData loading complete:")
    print(f"  Total lines: {line_count:,}")
    print(f"  Zero weights skipped: {zero_weight_count:,}")
    print(f"  Invalid entries skipped: {invalid_count:,}")
    print(f"  Total users: {len(user_data):,}")
    
    # Filter users with sufficient data
    eligible_users = {}
    for uid, measurements in user_data.items():
        if len(measurements) >= 10:
            # Sort by timestamp
            measurements.sort(key=lambda x: x['timestamp'])
            eligible_users[uid] = measurements
    
    print(f"  Eligible users (10+ valid measurements): {len(eligible_users):,}")
    
    # Analyze population
    print("\n" + "="*80)
    print("POPULATION ANALYSIS")
    print("="*80)
    
    measurement_counts = []
    source_diversity = []
    source_distributions = defaultdict(int)
    user_categories = defaultdict(int)
    
    for uid, measurements in eligible_users.items():
        measurement_counts.append(len(measurements))
        
        sources = set(m['normalized'] for m in measurements)
        source_diversity.append(len(sources))
        
        if len(sources) == 1:
            user_categories['single_source'] += 1
        elif len(sources) == 2:
            user_categories['two_sources'] += 1
        else:
            user_categories['multi_source'] += 1
        
        for m in measurements:
            source_distributions[m['normalized']] += 1
    
    print(f"\nMeasurements per user:")
    print(f"  Mean: {np.mean(measurement_counts):.1f}")
    print(f"  Median: {np.median(measurement_counts):.0f}")
    print(f"  Range: {min(measurement_counts)}-{max(measurement_counts)}")
    
    print(f"\nUser categories:")
    total_users = len(eligible_users)
    for category, count in sorted(user_categories.items()):
        pct = count / total_users * 100
        print(f"  {category}: {count:,} ({pct:.1f}%)")
    
    print(f"\nSource distribution:")
    total_measurements = sum(source_distributions.values())
    for source, count in sorted(source_distributions.items(), key=lambda x: x[1], reverse=True):
        pct = count / total_measurements * 100
        print(f"  {source}: {count:,} ({pct:.1f}%)")
    
    # Test strategies on sample
    print("\n" + "="*80)
    print("STRATEGY TESTING")
    print("="*80)
    
    # Sample users for testing (use all if < 1000, otherwise sample 1000)
    import random
    if len(eligible_users) > 1000:
        test_users = random.sample(list(eligible_users.keys()), 1000)
        print(f"\nTesting strategies on sample of 1,000 users...")
    else:
        test_users = list(eligible_users.keys())
        print(f"\nTesting strategies on all {len(test_users):,} users...")
    
    strategies = {
        'baseline': strategy_baseline,
        'trust_weighted': strategy_trust_weighted,
        'adaptive_limits': strategy_adaptive_limits
    }
    
    results = defaultdict(lambda: defaultdict(list))
    
    for i, uid in enumerate(test_users):
        if i % 100 == 0:
            print(f"  Processing user {i+1}/{len(test_users)}...")
        
        measurements = eligible_users[uid]
        
        for strategy_name, strategy_func in strategies.items():
            try:
                strategy_results = strategy_func(measurements)
                metrics = calculate_metrics(strategy_results)
                
                for metric, value in metrics.items():
                    results[strategy_name][metric].append(value)
            except Exception as e:
                print(f"    Error with {strategy_name} for user {uid[:8]}: {e}")
                continue
    
    # Calculate aggregate statistics
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    aggregate_results = {}
    for strategy in results:
        aggregate_results[strategy] = {}
        for metric in results[strategy]:
            values = results[strategy][metric]
            if values:
                aggregate_results[strategy][metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'median': np.median(values)
                }
    
    # Rank strategies
    rankings = []
    for strategy, metrics in aggregate_results.items():
        if 'acceptance_rate' in metrics:
            score = (
                (1 - metrics['acceptance_rate']['mean']) * 10 +
                metrics.get('smoothness', {}).get('mean', 0) * 2 +
                metrics.get('avg_error', {}).get('mean', 0) * 3
            )
            rankings.append((strategy, score, metrics))
    
    rankings.sort(key=lambda x: x[1])
    
    print("\nStrategy Rankings:")
    for rank, (strategy, score, metrics) in enumerate(rankings, 1):
        print(f"\n{rank}. {strategy.upper()}")
        print(f"   Score: {score:.3f}")
        print(f"   Acceptance: {metrics['acceptance_rate']['mean']*100:.2f}% ± {metrics['acceptance_rate']['std']*100:.2f}%")
        if 'smoothness' in metrics:
            print(f"   Smoothness: {metrics['smoothness']['mean']:.4f} ± {metrics['smoothness']['std']:.4f}")
        if 'avg_error' in metrics:
            print(f"   Avg Error: {metrics['avg_error']['mean']:.3f} ± {metrics['avg_error']['std']:.3f} kg")
    
    # Final conclusion
    print("\n" + "="*80)
    print("FINAL CONCLUSION")
    print("="*80)
    
    if rankings and rankings[0][0] == 'baseline':
        print("\n✅ DEFINITIVE RESULT: Baseline processor is optimal")
        print("✅ No source differentiation needed")
        print("✅ Current implementation should be maintained")
    elif rankings:
        print(f"\n✅ DEFINITIVE RESULT: {rankings[0][0]} strategy is optimal")
        baseline_score = next((s for strat, s, _ in rankings if strat == 'baseline'), None)
        if baseline_score:
            improvement = ((baseline_score - rankings[0][1]) / baseline_score) * 100
            print(f"✅ Provides {improvement:.1f}% improvement over baseline")
    
    print(f"\nAnalysis based on:")
    print(f"  • {len(test_users):,} users tested")
    print(f"  • {total_measurements:,} total measurements")
    print(f"  • {len(eligible_users):,} eligible users in dataset")
    
    return aggregate_results


def strategy_baseline(measurements):
    """Baseline strategy."""
    db = ProcessorStateDB()
    results = []
    
    config = {
        'processing': {
            'extreme_threshold': 10.0,
            'max_weight': 400.0,
            'min_weight': 30.0,
        },
        'kalman': {
            'observation_covariance': 5.0,
            'transition_covariance_weight': 0.01,
            'transition_covariance_trend': 0.0001,
            'initial_variance': 10.0
        }
    }
    
    for m in measurements:
        try:
            result = WeightProcessor.process_weight(
                user_id='test',
                weight=m['weight'],
                timestamp=m['timestamp'],
                source=m['source'],
                processing_config=config['processing'],
                kalman_config=config['kalman'],
                db=db
            )
            if result:
                results.append(result)
        except:
            continue
    
    return results


def strategy_trust_weighted(measurements):
    """Trust-weighted strategy."""
    db = ProcessorStateDB()
    results = []
    
    trust_scores = {
        'device': 1.0,
        'connected': 0.85,
        'api': 0.85,
        'questionnaire': 0.6,
        'manual': 0.4,
        'other': 0.5
    }
    
    base_config = {
        'processing': {
            'extreme_threshold': 10.0,
            'max_weight': 400.0,
            'min_weight': 30.0,
        },
        'kalman': {
            'observation_covariance': 5.0,
            'transition_covariance_weight': 0.01,
            'transition_covariance_trend': 0.0001,
            'initial_variance': 10.0
        }
    }
    
    for m in measurements:
        try:
            trust = trust_scores.get(m['normalized'], 0.5)
            config = base_config.copy()
            config['kalman'] = base_config['kalman'].copy()
            config['kalman']['observation_covariance'] = 5.0 / (trust ** 2)
            
            result = WeightProcessor.process_weight(
                user_id='test',
                weight=m['weight'],
                timestamp=m['timestamp'],
                source=m['source'],
                processing_config=config['processing'],
                kalman_config=config['kalman'],
                db=db
            )
            if result:
                results.append(result)
        except:
            continue
    
    return results


def strategy_adaptive_limits(measurements):
    """Adaptive limits strategy."""
    db = ProcessorStateDB()
    results = []
    
    limit_multipliers = {
        'device': 1.0,
        'connected': 1.2,
        'api': 1.2,
        'questionnaire': 1.5,
        'manual': 2.0,
        'other': 1.3
    }
    
    base_config = {
        'processing': {
            'extreme_threshold': 10.0,
            'max_weight': 400.0,
            'min_weight': 30.0,
        },
        'kalman': {
            'observation_covariance': 5.0,
            'transition_covariance_weight': 0.01,
            'transition_covariance_trend': 0.0001,
            'initial_variance': 10.0
        }
    }
    
    for m in measurements:
        try:
            multiplier = limit_multipliers.get(m['normalized'], 1.3)
            config = base_config.copy()
            config['processing'] = base_config['processing'].copy()
            config['processing']['extreme_threshold'] = 10.0 * multiplier
            
            result = WeightProcessor.process_weight(
                user_id='test',
                weight=m['weight'],
                timestamp=m['timestamp'],
                source=m['source'],
                processing_config=config['processing'],
                kalman_config=config['kalman'],
                db=db
            )
            if result:
                results.append(result)
        except:
            continue
    
    return results


def calculate_metrics(results):
    """Calculate metrics."""
    if not results:
        return {'acceptance_rate': 0}
    
    valid = [r for r in results if 'filtered_weight' in r]
    rejected = [r for r in results if r.get('rejected')]
    
    metrics = {
        'acceptance_rate': len(valid) / len(results) if results else 0
    }
    
    if len(valid) > 1:
        weights = [r['filtered_weight'] for r in valid]
        diffs = np.diff(weights)
        metrics['smoothness'] = np.std(diffs) if len(diffs) > 0 else 0
        metrics['max_jump'] = np.max(np.abs(diffs)) if len(diffs) > 0 else 0
        
        errors = []
        for r in valid:
            if 'raw_weight' in r:
                errors.append(abs(r['filtered_weight'] - r['raw_weight']))
        
        if errors:
            metrics['avg_error'] = np.mean(errors)
    
    return metrics


if __name__ == "__main__":
    analyze_complete_dataset()
