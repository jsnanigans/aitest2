"""
Simple optimization test - compare current vs optimized parameters
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from datetime import datetime, timedelta
from processor import WeightProcessor


def test_parameter_sets():
    """Test different parameter sets to find optimal configuration."""
    
    print("=" * 60)
    print("KALMAN PARAMETER OPTIMIZATION TEST")
    print("=" * 60)
    
    # Generate test data
    base_date = datetime(2024, 1, 1)
    test_cases = []
    
    # Case 1: Stable weight with normal noise
    stable_data = []
    for i in range(60):
        weight = 70.0 + np.random.normal(0, 0.5)
        stable_data.append((weight, base_date + timedelta(days=i), "scale"))
    test_cases.append(("stable", stable_data, 70.0))
    
    # Case 2: Weight loss trend
    loss_data = []
    for i in range(60):
        weight = 85.0 - 0.1 * i + np.random.normal(0, 0.5)
        loss_data.append((weight, base_date + timedelta(days=i), "scale"))
    test_cases.append(("weight_loss", loss_data, 85.0))
    
    # Case 3: Noisy data
    noisy_data = []
    for i in range(60):
        if i % 10 == 0:  # Occasional outlier
            weight = 75.0 + np.random.uniform(3, 5)
        else:
            weight = 75.0 + np.random.normal(0, 0.8)
        noisy_data.append((weight, base_date + timedelta(days=i), "scale"))
    test_cases.append(("noisy", noisy_data, 75.0))
    
    # Parameter sets to test
    param_sets = {
        "current": {
            "processing": {
                "min_init_readings": 10,
                "min_weight": 30.0,
                "max_weight": 400.0,
                "max_daily_change": 0.03,
                "extreme_threshold": 0.30
            },
            "kalman": {
                "initial_variance": 1.0,
                "transition_covariance_weight": 0.5,
                "transition_covariance_trend": 0.01,
                "observation_covariance": 1.0,
                "reset_gap_days": 30
            }
        },
        "optimized_v1": {
            "processing": {
                "min_init_readings": 10,
                "min_weight": 30.0,
                "max_weight": 400.0,
                "max_daily_change": 0.05,
                "extreme_threshold": 0.25
            },
            "kalman": {
                "initial_variance": 1.0,
                "transition_covariance_weight": 0.1,
                "transition_covariance_trend": 0.001,
                "observation_covariance": 2.0,
                "reset_gap_days": 30
            }
        },
        "conservative": {
            "processing": {
                "min_init_readings": 10,
                "min_weight": 30.0,
                "max_weight": 400.0,
                "max_daily_change": 0.05,
                "extreme_threshold": 0.20
            },
            "kalman": {
                "initial_variance": 0.5,
                "transition_covariance_weight": 0.05,
                "transition_covariance_trend": 0.0005,
                "observation_covariance": 1.5,
                "reset_gap_days": 30
            }
        },
        "aggressive": {
            "processing": {
                "min_init_readings": 10,
                "min_weight": 30.0,
                "max_weight": 400.0,
                "max_daily_change": 0.05,
                "extreme_threshold": 0.35
            },
            "kalman": {
                "initial_variance": 2.0,
                "transition_covariance_weight": 0.3,
                "transition_covariance_trend": 0.005,
                "observation_covariance": 2.5,
                "reset_gap_days": 30
            }
        }
    }
    
    results = {}
    
    for param_name, params in param_sets.items():
        print(f"\nTesting parameter set: {param_name}")
        print("-" * 40)
        
        param_results = {
            'acceptance_rates': [],
            'mse_values': [],
            'stability_values': [],
            'outlier_detection': []
        }
        
        for case_name, readings, true_weight in test_cases:
            processor = WeightProcessor(
                f"test_{case_name}",
                params["processing"],
                params["kalman"]
            )
            
            results_list = []
            for weight, timestamp, source in readings:
                result = processor.process_weight(weight, timestamp, source)
                if result:
                    results_list.append(result)
            
            if results_list:
                # Calculate metrics
                accepted = [r for r in results_list if r['accepted']]
                acceptance_rate = len(accepted) / len(results_list) * 100
                param_results['acceptance_rates'].append(acceptance_rate)
                
                if accepted:
                    # MSE from true weight
                    filtered_weights = [r['filtered_weight'] for r in accepted[-20:]]
                    mse = np.mean([(w - true_weight) ** 2 for w in filtered_weights])
                    param_results['mse_values'].append(mse)
                    
                    # Stability (std dev)
                    stability = np.std(filtered_weights)
                    param_results['stability_values'].append(stability)
                    
                    # Outlier detection (for noisy case)
                    if case_name == "noisy":
                        rejected = len(results_list) - len(accepted)
                        param_results['outlier_detection'].append(rejected)
                
                print(f"  {case_name}: {acceptance_rate:.1f}% accepted, "
                      f"MSE={mse:.3f}, σ={stability:.3f}")
        
        # Calculate overall scores
        avg_acceptance = np.mean(param_results['acceptance_rates'])
        avg_mse = np.mean(param_results['mse_values'])
        avg_stability = np.mean(param_results['stability_values'])
        
        # Score (lower is better)
        score = avg_mse * 10 + avg_stability * 5 + abs(avg_acceptance - 95) * 0.1
        
        results[param_name] = {
            'acceptance': avg_acceptance,
            'mse': avg_mse,
            'stability': avg_stability,
            'score': score
        }
    
    # Rank results
    ranked = sorted(results.items(), key=lambda x: x[1]['score'])
    
    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULTS (ranked by score)")
    print("=" * 60)
    
    print("\n%-15s %10s %10s %10s %10s" % (
        "Parameter Set", "Score", "Accept%", "MSE", "Stability"
    ))
    print("-" * 65)
    
    for param_name, metrics in ranked:
        print("%-15s %10.2f %9.1f%% %10.3f %10.3f" % (
            param_name,
            metrics['score'],
            metrics['acceptance'],
            metrics['mse'],
            metrics['stability']
        ))
    
    # Show best parameters
    best_name = ranked[0][0]
    best_params = param_sets[best_name]
    
    print("\n" + "=" * 60)
    print("RECOMMENDED CONFIGURATION")
    print("=" * 60)
    print(f"\nBest parameter set: {best_name}")
    print("\nUpdate config.toml with:")
    print("\n[processing]")
    for key, value in best_params['processing'].items():
        if key in ['extreme_threshold', 'max_daily_change']:
            print(f"{key} = {value}")
    print("\n[kalman]")
    for key, value in best_params['kalman'].items():
        if key != 'reset_gap_days':
            print(f"{key} = {value}")
    
    # Compare with current
    current_score = results['current']['score']
    best_score = ranked[0][1]['score']
    improvement = (1 - best_score/current_score) * 100
    
    print(f"\nImprovement over current: {improvement:.1f}%")


if __name__ == "__main__":
    np.random.seed(42)  # For reproducibility
    test_parameter_sets()