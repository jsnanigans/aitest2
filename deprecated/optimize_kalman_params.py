#!/usr/bin/env python3

import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
from src.filters.custom_kalman_filter import CustomKalmanFilter

def load_benchmark_data() -> Dict:
    """Load problematic test cases from benchmarks"""
    test_cases = {
        '01C39CB085ED4D349B3324D30766C156': {
            'true_baseline': 86.5,
            'readings': [
                (86.5, 0), (86.4, 1), (86.3, 2),
                (60.9, 3), (61.0, 4),  # Outliers
                (86.2, 5), (86.5, 6), (86.4, 7)
            ],
            'outlier_indices': [3, 4]
        },
        '090CF20AAA7F495595A30C0F3FEE34BE': {
            'true_baseline': 118.5,
            'readings': [
                (118.5, 0), (118.3, 1),
                (33.9, 2), (32.2, 3), (34.5, 4),  # Extreme outliers
                (118.3, 5), (118.5, 6), (118.4, 7)
            ],
            'outlier_indices': [2, 3, 4]
        }
    }
    return test_cases

def evaluate_parameters(params: Dict, test_cases: Dict) -> float:
    """Evaluate parameter set performance"""
    total_score = 0
    
    for user_id, case in test_cases.items():
        kf = CustomKalmanFilter(**params)
        
        max_deviation = 0
        recovery_deviations = []
        outliers_detected = 0
        
        for i, (weight, day) in enumerate(case['readings']):
            result = kf.process_measurement(weight, time_delta_days=1.0)
            
            filtered_weight = result['filtered_weight']
            deviation = abs(filtered_weight - case['true_baseline'])
            
            if i in case['outlier_indices']:
                max_deviation = max(max_deviation, deviation)
                if result['normalized_innovation'] > 3.0:
                    outliers_detected += 1
            else:
                recovery_deviations.append(deviation)
        
        outlier_detection_rate = outliers_detected / len(case['outlier_indices'])
        avg_recovery = np.mean(recovery_deviations) if recovery_deviations else 0
        
        case_score = (
            (1.0 / (1.0 + max_deviation)) * 30 +  # Lower max deviation is better
            outlier_detection_rate * 30 +          # Higher detection rate is better
            (1.0 / (1.0 + avg_recovery)) * 40      # Lower recovery deviation is better
        )
        
        total_score += case_score
    
    return total_score / len(test_cases)

def grid_search_optimization():
    """Perform grid search to find optimal parameters"""
    
    test_cases = load_benchmark_data()
    
    # Parameter search space based on benchmark findings
    param_grid = {
        'process_noise_weight': [0.2, 0.3, 0.4, 0.5],
        'max_reasonable_trend': [0.03, 0.04, 0.05, 0.06], 
        'process_noise_trend': [0.01, 0.02, 0.03, 0.04],
        'measurement_noise': [0.8, 1.0, 1.2, 1.5]
    }
    
    best_score = 0
    best_params = None
    results = []
    
    print("Starting parameter optimization...")
    total_combinations = (len(param_grid['process_noise_weight']) * 
                          len(param_grid['max_reasonable_trend']) * 
                          len(param_grid['process_noise_trend']) * 
                          len(param_grid['measurement_noise']))
    
    current = 0
    for pnw in param_grid['process_noise_weight']:
        for mrt in param_grid['max_reasonable_trend']:
            for pnt in param_grid['process_noise_trend']:
                for mn in param_grid['measurement_noise']:
                    current += 1
                    params = {
                        'process_noise_weight': pnw,
                        'max_reasonable_trend': mrt,
                        'process_noise_trend': pnt,
                        'measurement_noise': mn
                    }
                    
                    score = evaluate_parameters(params, test_cases)
                    results.append((score, params))
                    
                    if score > best_score:
                        best_score = score
                        best_params = params
                        print(f"[{current}/{total_combinations}] New best score: {score:.2f}")
                        print(f"  Parameters: {params}")
    
    results.sort(reverse=True, key=lambda x: x[0])
    
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE")
    print("="*60)
    print(f"\nBest Score: {best_score:.2f}/100")
    print("\nOptimal Parameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    
    print("\nTop 5 Parameter Sets:")
    for i, (score, params) in enumerate(results[:5], 1):
        print(f"\n{i}. Score: {score:.2f}")
        for key, value in params.items():
            print(f"   {key}: {value}")
    
    # Test the best parameters on specific cases
    print("\n" + "="*60)
    print("DETAILED VALIDATION")
    print("="*60)
    
    for user_id, case in test_cases.items():
        print(f"\nUser: {user_id}")
        print(f"True baseline: {case['true_baseline']:.1f} kg")
        
        kf = CustomKalmanFilter(**best_params)
        
        for i, (weight, day) in enumerate(case['readings']):
            result = kf.process_measurement(weight, time_delta_days=1.0)
            is_outlier = i in case['outlier_indices']
            
            print(f"  Reading {i+1}: {weight:.1f} kg -> {result['filtered_weight']:.1f} kg "
                  f"(innovation: {result['normalized_innovation']:.2f}σ) "
                  f"{'[OUTLIER]' if is_outlier else ''}")
    
    # Save results
    output_file = f"output/kalman_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump({
            'best_params': best_params,
            'best_score': best_score,
            'top_10_results': [(score, params) for score, params in results[:10]],
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    return best_params

if __name__ == "__main__":
    optimal_params = grid_search_optimization()
    
    print("\n" + "="*60)
    print("RECOMMENDED CHANGES TO CustomKalmanFilter")
    print("="*60)
    print("\nUpdate the default parameters in __init__ to:")
    print(f"  process_noise_weight: {optimal_params['process_noise_weight']}")
    print(f"  max_reasonable_trend: {optimal_params['max_reasonable_trend']}")  
    print(f"  process_noise_trend: {optimal_params['process_noise_trend']}")
    print(f"  measurement_noise: {optimal_params['measurement_noise']}")