"""
Test to determine if adaptive noise actually improves the weight stream processor.
"""

import sys
sys.path.insert(0, '.')

import numpy as np
from datetime import datetime, timedelta
from src.processor import WeightProcessor, process_weight_enhanced
from src.database import get_state_db
from src.models import KALMAN_DEFAULTS, SOURCE_PROFILES, PROCESSING_DEFAULTS
import json
import os

def generate_test_data():
    """Generate realistic test data with different noise characteristics."""
    np.random.seed(42)
    base_weight = 75.0
    timestamps = []
    weights = []
    sources = []
    
    # Generate 100 days of data
    start_date = datetime(2024, 1, 1)
    
    for day in range(100):
        timestamp = start_date + timedelta(days=day)
        
        # True weight with small daily variation
        true_weight = base_weight + 0.01 * day + 0.2 * np.sin(day * 0.1)
        
        # Add measurements from different sources with their characteristic noise
        # Morning measurement from patient-device (moderate noise)
        morning_time = timestamp.replace(hour=7)
        device_noise = np.random.normal(0, 1.0)  # 1kg std dev
        weights.append(true_weight + device_noise)
        timestamps.append(morning_time)
        sources.append('patient-device')
        
        # Occasional care-team-upload (very low noise)
        if day % 7 == 0:  # Weekly care team measurement
            care_time = timestamp.replace(hour=10)
            care_noise = np.random.normal(0, 0.3)  # 0.3kg std dev
            weights.append(true_weight + care_noise)
            timestamps.append(care_time)
            sources.append('care-team-upload')
        
        # Occasional iglucose (high noise)
        if day % 3 == 0:  # Every 3 days
            iglucose_time = timestamp.replace(hour=14)
            iglucose_noise = np.random.normal(0, 3.0)  # 3kg std dev
            # Add occasional outliers
            if np.random.random() < 0.15:  # 15% outlier rate
                iglucose_noise += np.random.choice([-10, 10])
            weights.append(true_weight + iglucose_noise)
            timestamps.append(iglucose_time)
            sources.append('https://api.iglucose.com')
    
    return timestamps, weights, sources, true_weight + 0.01 * 99 + 0.2 * np.sin(99 * 0.1)


def test_with_adaptive_noise(timestamps, weights, sources):
    """Test with adaptive noise enabled (current implementation)."""
    user_id = "test_user_adaptive"
    results = []
    
    # Clear any existing state
    db = get_state_db()
    db.clear_state(user_id)
    
    for timestamp, weight, source in zip(timestamps, weights, sources):
        # This uses the enhanced implementation which includes adaptive noise
        result = process_weight_enhanced(
            user_id=user_id,
            weight=weight,
            timestamp=timestamp,
            source=source,
            processing_config=PROCESSING_DEFAULTS.copy(),
            kalman_config=KALMAN_DEFAULTS.copy()
        )
        if result:
            results.append(result)
    
    return results


def test_with_fixed_noise(timestamps, weights, sources):
    """Test with fixed noise (no adaptation)."""
    user_id = "test_user_fixed"
    results = []
    
    # Clear any existing state
    db = get_state_db()
    db.clear_state(user_id)
    
    # Use fixed noise for all sources (bypass adaptation)
    fixed_kalman_config = KALMAN_DEFAULTS.copy()
    processing_config = PROCESSING_DEFAULTS.copy()
    
    for timestamp, weight, source in zip(timestamps, weights, sources):
        # Process without adaptation - use base process_weight
        result = WeightProcessor.process_weight(
            user_id=user_id,
            weight=weight,
            timestamp=timestamp,
            source=source,
            kalman_config=fixed_kalman_config,
            processing_config=processing_config
        )
        
        if result:
            results.append(result)
    
    return results


def calculate_metrics(results, true_final_weight):
    """Calculate performance metrics."""
    if not results:
        return None
    
    # Get final filtered weight
    final_filtered = results[-1]['filtered_weight']
    
    # Calculate error from true weight
    final_error = abs(final_filtered - true_final_weight)
    
    # Calculate stability (variance of filtered weights in last 10 measurements)
    last_10_filtered = [r['filtered_weight'] for r in results[-10:]]
    stability = np.std(last_10_filtered) if len(last_10_filtered) > 1 else 0
    
    # Calculate acceptance rate
    accepted_count = sum(1 for r in results if r.get('accepted', False))
    acceptance_rate = accepted_count / len(results) if results else 0
    
    # Calculate confidence metrics
    avg_confidence = np.mean([r.get('confidence', 0) for r in results])
    
    # Calculate innovation metrics (how well we track changes)
    innovations = [abs(r.get('innovation', 0)) for r in results]
    avg_innovation = np.mean(innovations) if innovations else 0
    
    # Track noise values used
    noise_values = []
    for r in results:
        if 'measurement_noise_used' in r:
            noise_values.append(r['measurement_noise_used'])
        elif 'adapted_params' in r:
            noise_values.append(r['adapted_params'].get('measurement_noise', KALMAN_DEFAULTS['observation_covariance']))
        else:
            noise_values.append(KALMAN_DEFAULTS['observation_covariance'])
    
    avg_noise = np.mean(noise_values) if noise_values else KALMAN_DEFAULTS['observation_covariance']
    
    return {
        'final_error': final_error,
        'stability': stability,
        'acceptance_rate': acceptance_rate,
        'avg_confidence': avg_confidence,
        'avg_innovation': avg_innovation,
        'avg_noise': avg_noise,
        'num_measurements': len(results)
    }


def main():
    print("=" * 80)
    print("ADAPTIVE NOISE EFFECTIVENESS TEST")
    print("=" * 80)
    
    # Generate test data
    print("\nGenerating test data...")
    timestamps, weights, sources, true_final = generate_test_data()
    
    # Count measurements by source
    source_counts = {}
    for source in sources:
        source_counts[source] = source_counts.get(source, 0) + 1
    
    print(f"Total measurements: {len(weights)}")
    for source, count in source_counts.items():
        profile = SOURCE_PROFILES.get(source, {'noise_multiplier': 1.0})
        print(f"  {source}: {count} measurements (noise multiplier: {profile['noise_multiplier']})")
    print(f"True final weight: {true_final:.2f} kg")
    
    # Test with adaptive noise
    print("\n" + "-" * 40)
    print("Testing WITH adaptive noise...")
    adaptive_results = test_with_adaptive_noise(timestamps, weights, sources)
    adaptive_metrics = calculate_metrics(adaptive_results, true_final)
    
    # Test with fixed noise
    print("Testing WITHOUT adaptive noise...")
    fixed_results = test_with_fixed_noise(timestamps, weights, sources)
    fixed_metrics = calculate_metrics(fixed_results, true_final)
    
    # Compare results
    print("\n" + "=" * 80)
    print("RESULTS COMPARISON")
    print("=" * 80)
    
    print("\nAdaptive Noise (Current Implementation):")
    print(f"  Final error: {adaptive_metrics['final_error']:.3f} kg")
    print(f"  Stability (std dev): {adaptive_metrics['stability']:.3f} kg")
    print(f"  Acceptance rate: {adaptive_metrics['acceptance_rate']:.1%}")
    print(f"  Avg confidence: {adaptive_metrics['avg_confidence']:.3f}")
    print(f"  Avg innovation: {adaptive_metrics['avg_innovation']:.3f} kg")
    print(f"  Avg noise parameter: {adaptive_metrics['avg_noise']:.3f}")
    
    print("\nFixed Noise (No Adaptation):")
    print(f"  Final error: {fixed_metrics['final_error']:.3f} kg")
    print(f"  Stability (std dev): {fixed_metrics['stability']:.3f} kg")
    print(f"  Acceptance rate: {fixed_metrics['acceptance_rate']:.1%}")
    print(f"  Avg confidence: {fixed_metrics['avg_confidence']:.3f}")
    print(f"  Avg innovation: {fixed_metrics['avg_innovation']:.3f} kg")
    print(f"  Avg noise parameter: {fixed_metrics['avg_noise']:.3f}")
    
    print("\n" + "-" * 40)
    print("IMPROVEMENT WITH ADAPTIVE NOISE:")
    
    if fixed_metrics['final_error'] > 0:
        error_improvement = (fixed_metrics['final_error'] - adaptive_metrics['final_error']) / fixed_metrics['final_error'] * 100
    else:
        error_improvement = 0
    
    if fixed_metrics['stability'] > 0:
        stability_improvement = (fixed_metrics['stability'] - adaptive_metrics['stability']) / fixed_metrics['stability'] * 100
    else:
        stability_improvement = 0
    
    print(f"  Error reduction: {error_improvement:+.1f}%")
    print(f"  Stability improvement: {stability_improvement:+.1f}%")
    print(f"  Acceptance rate change: {(adaptive_metrics['acceptance_rate'] - fixed_metrics['acceptance_rate'])*100:+.1f} pp")
    
    # Analyze per-source performance
    print("\n" + "-" * 40)
    print("PER-SOURCE ANALYSIS:")
    
    for source in set(sources):
        source_adaptive = [r for r in adaptive_results if r.get('source') == source]
        source_fixed = [r for r in fixed_results if r.get('source') == source]
        
        if source_adaptive and source_fixed:
            adaptive_conf = np.mean([r.get('confidence', 0) for r in source_adaptive])
            fixed_conf = np.mean([r.get('confidence', 0) for r in source_fixed])
            
            adaptive_innov = np.mean([abs(r.get('innovation', 0)) for r in source_adaptive])
            fixed_innov = np.mean([abs(r.get('innovation', 0)) for r in source_fixed])
            
            adaptive_accept = sum(1 for r in source_adaptive if r.get('accepted', False)) / len(source_adaptive)
            fixed_accept = sum(1 for r in source_fixed if r.get('accepted', False)) / len(source_fixed)
            
            print(f"\n{source}:")
            print(f"  Noise multiplier: {SOURCE_PROFILES.get(source, {'noise_multiplier': 1.0})['noise_multiplier']}")
            print(f"  Adaptive: conf={adaptive_conf:.3f}, innov={adaptive_innov:.3f}, accept={adaptive_accept:.1%}")
            print(f"  Fixed:    conf={fixed_conf:.3f}, innov={fixed_innov:.3f}, accept={fixed_accept:.1%}")
            
            # Check if adaptive noise helped this source
            if adaptive_conf > fixed_conf and fixed_conf > 0:
                print(f"  ✓ Confidence improved by {(adaptive_conf - fixed_conf)/fixed_conf*100:.1f}%")
            elif fixed_conf > 0:
                print(f"  ✗ Confidence reduced by {(fixed_conf - adaptive_conf)/fixed_conf*100:.1f}%")
    
    # Save detailed results
    os.makedirs('test_output', exist_ok=True)
    
    results_data = {
        'test_data': {
            'num_measurements': len(weights),
            'sources': source_counts,
            'true_final_weight': true_final
        },
        'adaptive_metrics': adaptive_metrics,
        'fixed_metrics': fixed_metrics,
        'improvements': {
            'error_reduction_pct': error_improvement,
            'stability_improvement_pct': stability_improvement,
            'acceptance_rate_change_pp': (adaptive_metrics['acceptance_rate'] - fixed_metrics['acceptance_rate'])*100
        }
    }
    
    with open('test_output/adaptive_noise_results.json', 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print("\n" + "=" * 80)
    print("CONCLUSION:")
    
    # Determine overall effectiveness
    improvements = 0
    degradations = 0
    
    if error_improvement > 0:
        improvements += 1
        print(f"✓ Error reduced by {error_improvement:.1f}%")
    elif error_improvement < -5:  # Allow small degradation
        degradations += 1
        print(f"✗ Error increased by {-error_improvement:.1f}%")
    else:
        print(f"≈ Error change negligible ({error_improvement:+.1f}%)")
    
    if stability_improvement > 0:
        improvements += 1
        print(f"✓ Stability improved by {stability_improvement:.1f}%")
    elif stability_improvement < -5:
        degradations += 1
        print(f"✗ Stability degraded by {-stability_improvement:.1f}%")
    else:
        print(f"≈ Stability change negligible ({stability_improvement:+.1f}%)")
    
    accept_change = (adaptive_metrics['acceptance_rate'] - fixed_metrics['acceptance_rate']) * 100
    if accept_change > 5:
        improvements += 1
        print(f"✓ Acceptance rate improved by {accept_change:.1f} pp")
    elif accept_change < -5:
        degradations += 1
        print(f"✗ Acceptance rate degraded by {-accept_change:.1f} pp")
    else:
        print(f"≈ Acceptance rate change negligible ({accept_change:+.1f} pp)")
    
    print("\n" + "-" * 40)
    if improvements > degradations:
        print("✓✓✓ ADAPTIVE NOISE IS BENEFICIAL ✓✓✓")
        print(f"   {improvements} metrics improved, {degradations} degraded")
    elif degradations > improvements:
        print("✗✗✗ ADAPTIVE NOISE IS HARMFUL ✗✗✗")
        print(f"   {degradations} metrics degraded, {improvements} improved")
    else:
        print("≈≈≈ ADAPTIVE NOISE HAS MIXED EFFECTS ≈≈≈")
        print(f"   {improvements} metrics improved, {degradations} degraded")
    
    print("\nDetailed results saved to test_output/adaptive_noise_results.json")


if __name__ == "__main__":
    main()
