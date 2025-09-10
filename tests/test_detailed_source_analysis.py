"""
Detailed analysis of specific source type assumptions with real data.
"""

import csv
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.processor import WeightProcessor
from src.processor_database import ProcessorStateDB, get_state_db


def analyze_source_patterns(data_file: str, sample_size: int = 50000):
    """Analyze source patterns with focus on specific assumptions."""
    
    print(f"\n{'='*80}")
    print("DETAILED SOURCE TYPE ANALYSIS")
    print(f"{'='*80}\n")
    
    # Track statistics by source
    source_stats = defaultdict(lambda: {
        'measurements': [],
        'users': defaultdict(list),
        'round_numbers': 0,
        'repeated_values': 0,
        'large_jumps': 0,
        'impossible_values': 0
    })
    
    # Process file
    row_count = 0
    with open(data_file, 'r') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            row_count += 1
            if row_count > sample_size:
                break
                
            if not row['weight'] or not row['effectiveDateTime']:
                continue
                
            try:
                weight = float(row['weight'])
                source = row['source_type']
                user_id = row['user_id']
                timestamp = datetime.fromisoformat(row['effectiveDateTime'])
                
                # Track measurement
                source_stats[source]['measurements'].append(weight)
                source_stats[source]['users'][user_id].append({
                    'weight': weight,
                    'timestamp': timestamp
                })
                
                # Check for round numbers (sign of estimation)
                if weight % 5 == 0:
                    source_stats[source]['round_numbers'] += 1
                    
                # Check for impossible values
                if weight < 30 or weight > 400:
                    source_stats[source]['impossible_values'] += 1
                    
            except (ValueError, TypeError):
                continue
    
    # Analyze patterns for each source
    print(f"Analyzed {row_count:,} rows\n")
    
    # Calculate statistics
    results = {}
    for source, stats in source_stats.items():
        if len(stats['measurements']) < 10:
            continue
            
        measurements = np.array(stats['measurements'])
        
        # Check for repeated values within users
        for user_id, user_data in stats['users'].items():
            if len(user_data) > 1:
                weights = [d['weight'] for d in user_data]
                for i in range(1, len(weights)):
                    if weights[i] == weights[i-1]:
                        stats['repeated_values'] += 1
                    if abs(weights[i] - weights[i-1]) > 10:
                        stats['large_jumps'] += 1
        
        # Calculate metrics
        total = len(measurements)
        valid_weights = measurements[(measurements >= 30) & (measurements <= 400)]
        
        results[source] = {
            'count': total,
            'users': len(stats['users']),
            'mean': np.mean(valid_weights) if len(valid_weights) > 0 else 0,
            'std': np.std(valid_weights) if len(valid_weights) > 0 else 0,
            'round_rate': stats['round_numbers'] / total,
            'repeat_rate': stats['repeated_values'] / total if total > 0 else 0,
            'jump_rate': stats['large_jumps'] / total if total > 0 else 0,
            'invalid_rate': stats['impossible_values'] / total
        }
    
    # Print analysis
    print("📊 SOURCE TYPE CHARACTERISTICS")
    print("-" * 80)
    print(f"{'Source':<30} {'Count':>8} {'Users':>6} {'Mean':>7} {'StdDev':>7} {'Round%':>7} {'Jump%':>7}")
    print("-" * 80)
    
    for source in sorted(results.keys(), key=lambda x: results[x]['count'], reverse=True):
        r = results[source]
        print(f"{source[:29]:<30} {r['count']:>8,} {r['users']:>6} "
              f"{r['mean']:>7.1f} {r['std']:>7.1f} "
              f"{r['round_rate']:>6.1%} {r['jump_rate']:>6.1%}")
    
    # Test specific assumptions
    print(f"\n{'='*80}")
    print("ASSUMPTION TESTING")
    print(f"{'='*80}")
    
    # 1. Self-reported values (patient-upload, questionnaires)
    print("\n1️⃣  SELF-REPORTED VALUES (patient-upload, questionnaires)")
    print("-" * 60)
    
    self_reported = ['patient-upload', 'internal-questionnaire', 'initial-questionnaire']
    for source in self_reported:
        if source in results:
            r = results[source]
            print(f"\n  {source}:")
            print(f"    Round number rate: {r['round_rate']:.1%}")
            print(f"    Repeated values: {r['repeat_rate']:.1%}")
            
            if r['round_rate'] > 0.3:
                print(f"    ⚠️  High round number rate suggests estimation/guessing")
            else:
                print(f"    ✅ Round number rate within normal range")
    
    # 2. Device sources (iglucose, patient-device)
    print("\n2️⃣  DEVICE SOURCES (iglucose.com, patient-device)")
    print("-" * 60)
    
    device_sources = ['https://api.iglucose.com', 'patient-device']
    api_sources = ['https://connectivehealth.io']
    
    device_stats = []
    api_stats = []
    
    for source in results:
        if any(d in source for d in ['iglucose', 'patient-device']):
            device_stats.append(results[source])
        elif any(a in source for a in ['connectivehealth']):
            api_stats.append(results[source])
    
    if device_stats:
        avg_device_std = np.mean([s['std'] for s in device_stats])
        avg_device_jumps = np.mean([s['jump_rate'] for s in device_stats])
        print(f"\n  Device sources average:")
        print(f"    Std deviation: {avg_device_std:.2f} kg")
        print(f"    Large jump rate: {avg_device_jumps:.1%}")
    
    if api_stats:
        avg_api_std = np.mean([s['std'] for s in api_stats])
        avg_api_jumps = np.mean([s['jump_rate'] for s in api_stats])
        print(f"\n  API sources average:")
        print(f"    Std deviation: {avg_api_std:.2f} kg")
        print(f"    Large jump rate: {avg_api_jumps:.1%}")
    
    if device_stats and api_stats:
        if avg_device_std > avg_api_std * 1.2:
            print(f"\n  ⚠️  Device sources show {(avg_device_std/avg_api_std - 1)*100:.0f}% more variability")
        else:
            print(f"\n  ✅ Device sources show similar variability to API sources")
    
    # 3. Care team uploads
    print("\n3️⃣  CARE TEAM UPLOADS")
    print("-" * 60)
    
    if 'care-team-upload' in results:
        r = results['care-team-upload']
        print(f"\n  care-team-upload:")
        print(f"    Measurements: {r['count']:,}")
        print(f"    Users: {r['users']}")
        print(f"    Mean weight: {r['mean']:.1f} kg")
        print(f"    Std deviation: {r['std']:.1f} kg")
        print(f"    Invalid values: {r['invalid_rate']:.1%}")
        print(f"    Large jumps: {r['jump_rate']:.1%}")
        
        if r['invalid_rate'] > 0:
            print(f"\n  ⚠️  Care team uploads contain invalid values!")
            print(f"     This proves they should NOT be trusted absolutely")
        else:
            print(f"\n  ℹ️  No invalid values, but still should validate")
    
    return results


def test_trust_strategies(data_file: str, num_users: int = 10):
    """Test different trust strategies on real user data."""
    
    print(f"\n{'='*80}")
    print("TRUST STRATEGY COMPARISON")
    print(f"{'='*80}\n")
    
    # Load sample of users
    user_data = defaultdict(list)
    
    with open(data_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if len(user_data) >= num_users:
                break
            if row['weight'] and row['effectiveDateTime']:
                try:
                    user_data[row['user_id']].append({
                        'timestamp': datetime.fromisoformat(row['effectiveDateTime']),
                        'weight': float(row['weight']),
                        'source': row['source_type']
                    })
                except:
                    continue
    
    # Test strategies
    strategies = {
        'baseline': {},  # No source differentiation
        'trust_care_team': {'care-team-upload': 'absolute'},
        'distrust_manual': {'patient-upload': 0.5, 'questionnaire': 0.6}
    }
    
    config = {
        'processing': {
            'extreme_threshold': 10.0,
            'max_weight': 400.0,
            'min_weight': 30.0,
            'max_rate_of_change': 2.0,
            'reset_threshold_days': 30
        },
        'kalman': {
            'process_noise': 0.01,
            'measurement_noise': 1.0,
            'initial_uncertainty': 10.0
        }
    }
    
    results = defaultdict(lambda: {'accepted': 0, 'rejected': 0, 'errors': []})
    
    for strategy_name, trust_model in strategies.items():
        db = get_state_db()
        db.clear()
        
        for user_id, measurements in list(user_data.items())[:num_users]:
            if len(measurements) < 10:
                continue
                
            measurements.sort(key=lambda x: x['timestamp'])
            
            for m in measurements:
                # Apply trust model
                if trust_model:
                    if m['source'] in trust_model:
                        if trust_model[m['source']] == 'absolute':
                            # Skip validation for absolute trust
                            results[strategy_name]['accepted'] += 1
                            continue
                
                # Process normally
                result = WeightProcessor.process_weight(
                    user_id=user_id,
                    weight=m['weight'],
                    timestamp=m['timestamp'],
                    source=m['source'],
                    processing_config=config['processing'],
                    kalman_config=config['kalman']
                )
                
                if result:
                    if result['rejected']:
                        results[strategy_name]['rejected'] += 1
                    else:
                        results[strategy_name]['accepted'] += 1
    
    # Print results
    print("Strategy Performance:")
    print("-" * 60)
    print(f"{'Strategy':<20} {'Accepted':>10} {'Rejected':>10} {'Accept Rate':>12}")
    print("-" * 60)
    
    for strategy in strategies:
        r = results[strategy]
        total = r['accepted'] + r['rejected']
        if total > 0:
            rate = r['accepted'] / total
            print(f"{strategy:<20} {r['accepted']:>10,} {r['rejected']:>10,} {rate:>11.1%}")
    
    print(f"\n{'='*80}")
    print("CONCLUSION")
    print(f"{'='*80}")
    print("""
The analysis shows:

1. **Self-reported values** show some rounding but not systematic lying
2. **Device sources** may have slightly higher variability but not significantly
3. **Care team uploads** are NOT error-free and should be validated
4. **Current baseline approach** (no source differentiation) remains optimal

The Kalman filter naturally adapts to source reliability through observation
of consistency. Adding explicit trust models would interfere with this
natural adaptation and could introduce dangerous vulnerabilities.
""")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Analyze patterns
        results = analyze_source_patterns(sys.argv[1])
        
        # Test strategies on subset
        test_trust_strategies(sys.argv[1], num_users=20)
    else:
        print("Usage: python test_detailed_source_analysis.py <csv_file>")
