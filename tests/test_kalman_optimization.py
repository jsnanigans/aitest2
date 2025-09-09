"""
Test suite for optimizing Kalman filter parameters
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from processor import WeightProcessor


def generate_synthetic_data() -> List[Tuple[str, List[Tuple[float, datetime, str]]]]:
    """Generate synthetic test cases with known ground truth."""
    test_cases = []
    base_date = datetime(2024, 1, 1)
    
    # Case 1: Stable weight with normal noise
    stable_weight = 70.0
    readings = []
    for i in range(90):
        noise = np.random.normal(0, 0.5)  # ±0.5kg typical scale noise
        weight = stable_weight + noise
        readings.append((weight, base_date + timedelta(days=i), "scale"))
    test_cases.append(("stable_user", readings))
    
    # Case 2: Linear weight loss (-0.1 kg/day = -0.7 kg/week)
    start_weight = 85.0
    readings = []
    for i in range(90):
        true_weight = start_weight - 0.1 * i
        noise = np.random.normal(0, 0.5)
        weight = true_weight + noise
        readings.append((weight, base_date + timedelta(days=i), "scale"))
    test_cases.append(("weight_loss_user", readings))
    
    # Case 3: Weight with outliers (meal/hydration spikes)
    base_weight = 75.0
    readings = []
    for i in range(90):
        if i % 7 == 0:  # Weekly outlier (big meal, hydration)
            spike = np.random.uniform(2, 4)  # 2-4kg spike
            weight = base_weight + spike
        else:
            noise = np.random.normal(0, 0.5)
            weight = base_weight + noise
        readings.append((weight, base_date + timedelta(days=i), "scale"))
    test_cases.append(("outlier_user", readings))
    
    # Case 4: Cyclic pattern (weekly variations)
    center_weight = 68.0
    readings = []
    for i in range(90):
        weekly_cycle = 1.5 * np.sin(2 * np.pi * i / 7)  # ±1.5kg weekly
        noise = np.random.normal(0, 0.3)
        weight = center_weight + weekly_cycle + noise
        readings.append((weight, base_date + timedelta(days=i), "scale"))
    test_cases.append(("cyclic_user", readings))
    
    # Case 5: Data with gaps
    base_weight = 72.0
    readings = []
    for i in range(90):
        if 30 <= i < 45:  # 15-day gap
            continue
        noise = np.random.normal(0, 0.5)
        weight = base_weight + noise
        readings.append((weight, base_date + timedelta(days=i), "scale"))
    test_cases.append(("gap_user", readings))
    
    return test_cases


def evaluate_parameters(params: Dict[str, float], test_cases: List) -> Dict[str, float]:
    """Evaluate a parameter set across all test cases."""
    
    metrics = {
        'mse': [],           # Mean squared error
        'outlier_rejection': [],  # Outlier rejection rate
        'trend_accuracy': [],     # Trend estimation accuracy
        'stability': [],          # Stability of estimates
        'recovery_time': []       # Recovery after outliers
    }
    
    for user_id, readings in test_cases:
        # Create processor with test parameters
        processing_config = {
            "min_init_readings": 10,
            "min_weight": 30.0,
            "max_weight": 400.0,
            "max_daily_change": 0.05,
            "extreme_threshold": params.get("extreme_threshold", 0.30)
        }
        
        kalman_config = {
            "initial_variance": params.get("initial_variance", 1.0),
            "transition_covariance_weight": params["transition_covariance_weight"],
            "transition_covariance_trend": params["transition_covariance_trend"],
            "observation_covariance": params["observation_covariance"],
            "reset_gap_days": 30
        }
        
        processor = WeightProcessor(user_id, processing_config, kalman_config)
        
        # Process all readings
        results = []
        for weight, timestamp, source in readings:
            result = processor.process_weight(weight, timestamp, source)
            if result:
                results.append(result)
        
        # Calculate metrics
        if len(results) > 20:
            # MSE for stable estimates
            filtered_weights = [r["filtered_weight"] for r in results if r["accepted"]]
            if filtered_weights:
                true_weight = np.median([w for w, _, _ in readings[:20]])
                mse = np.mean([(w - true_weight) ** 2 for w in filtered_weights[-20:]])
                metrics['mse'].append(mse)
            
            # Outlier rejection rate
            total = len(results)
            rejected = sum(1 for r in results if not r["accepted"])
            metrics['outlier_rejection'].append(rejected / total if total > 0 else 0)
            
            # Stability (variance of filtered weights)
            if len(filtered_weights) > 10:
                stability = np.var(filtered_weights[-10:])
                metrics['stability'].append(stability)
            
            # Trend accuracy (for weight loss case)
            if user_id == "weight_loss_user" and len(results) > 30:
                estimated_trend = np.mean([r["trend"] for r in results[-30:] if r["accepted"]])
                true_trend = -0.1  # We know this from synthetic data
                trend_error = abs(estimated_trend - true_trend)
                metrics['trend_accuracy'].append(trend_error)
    
    # Aggregate metrics
    aggregated = {}
    for key, values in metrics.items():
        if values:
            aggregated[key] = np.mean(values)
        else:
            aggregated[key] = float('inf')
    
    # Combined score (lower is better)
    score = (
        aggregated.get('mse', 0) * 1.0 +
        aggregated.get('trend_accuracy', 0) * 10.0 +
        aggregated.get('stability', 0) * 2.0 +
        abs(aggregated.get('outlier_rejection', 0) - 0.05) * 20.0  # Target 5% rejection
    )
    
    aggregated['combined_score'] = score
    return aggregated


def grid_search_optimization():
    """Perform grid search to find optimal Kalman parameters."""
    
    print("Generating synthetic test data...")
    test_cases = generate_synthetic_data()
    
    # Parameter search space
    param_grid = {
        'transition_covariance_weight': [0.1, 0.3, 0.5, 0.7, 1.0],
        'transition_covariance_trend': [0.001, 0.005, 0.01, 0.02],
        'observation_covariance': [0.5, 0.75, 1.0, 1.5, 2.0],
        'extreme_threshold': [0.2, 0.25, 0.3, 0.35]
    }
    
    best_score = float('inf')
    best_params = None
    all_results = []
    
    total_combinations = (
        len(param_grid['transition_covariance_weight']) *
        len(param_grid['transition_covariance_trend']) *
        len(param_grid['observation_covariance']) *
        len(param_grid['extreme_threshold'])
    )
    
    print(f"Testing {total_combinations} parameter combinations...")
    
    current = 0
    for tcw in param_grid['transition_covariance_weight']:
        for tct in param_grid['transition_covariance_trend']:
            for oc in param_grid['observation_covariance']:
                for et in param_grid['extreme_threshold']:
                    current += 1
                    
                    params = {
                        'transition_covariance_weight': tcw,
                        'transition_covariance_trend': tct,
                        'observation_covariance': oc,
                        'extreme_threshold': et
                    }
                    
                    metrics = evaluate_parameters(params, test_cases)
                    score = metrics['combined_score']
                    
                    all_results.append({
                        'params': params,
                        'metrics': metrics,
                        'score': score
                    })
                    
                    if score < best_score:
                        best_score = score
                        best_params = params
                        print(f"[{current}/{total_combinations}] New best score: {score:.3f}")
                        print(f"  Parameters: {params}")
    
    # Sort results by score
    all_results.sort(key=lambda x: x['score'])
    
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)
    
    print(f"\nBest Score: {best_score:.3f}")
    print("\nOptimal Parameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    
    print("\nMetrics for optimal parameters:")
    best_metrics = all_results[0]['metrics']
    for key, value in best_metrics.items():
        if key != 'combined_score':
            print(f"  {key}: {value:.4f}")
    
    print("\nTop 5 Parameter Sets:")
    for i, result in enumerate(all_results[:5], 1):
        print(f"\n{i}. Score: {result['score']:.3f}")
        for key, value in result['params'].items():
            print(f"   {key}: {value}")
    
    # Test on real data if available
    print("\n" + "=" * 60)
    print("VALIDATION ON EDGE CASES")
    print("=" * 60)
    
    # Create extreme test case
    extreme_readings = []
    base_date = datetime(2024, 1, 1)
    
    # Extreme outlier test
    for i in range(30):
        if i == 15:
            weight = 150.0  # Extreme outlier
        else:
            weight = 75.0 + np.random.normal(0, 0.5)
        extreme_readings.append((weight, base_date + timedelta(days=i), "scale"))
    
    processing_config = {
        "min_init_readings": 10,
        "min_weight": 30.0,
        "max_weight": 400.0,
        "max_daily_change": 0.05,
        "extreme_threshold": best_params["extreme_threshold"]
    }
    
    kalman_config = {
        "initial_variance": 1.0,
        "transition_covariance_weight": best_params["transition_covariance_weight"],
        "transition_covariance_trend": best_params["transition_covariance_trend"],
        "observation_covariance": best_params["observation_covariance"],
        "reset_gap_days": 30
    }
    
    processor = WeightProcessor("extreme_test", processing_config, kalman_config)
    
    print("\nExtreme outlier handling:")
    for i, (weight, timestamp, source) in enumerate(extreme_readings):
        result = processor.process_weight(weight, timestamp, source)
        if result and (i == 14 or i == 15 or i == 16):  # Around outlier
            filtered = result.get('filtered_weight', 0.0)
            if isinstance(filtered, (int, float)):
                print(f"  Day {i}: {weight:.1f}kg -> {filtered:.1f}kg "
                      f"(accepted: {result.get('accepted', 'N/A')})")
            else:
                print(f"  Day {i}: {weight:.1f}kg -> buffering")
    
    # Save results
    output = {
        'optimal_params': best_params,
        'best_score': best_score,
        'metrics': best_metrics,
        'top_10': [
            {
                'params': r['params'],
                'score': r['score'],
                'metrics': {k: v for k, v in r['metrics'].items() if k != 'combined_score'}
            }
            for r in all_results[:10]
        ],
        'timestamp': datetime.now().isoformat()
    }
    
    output_file = f"output/kalman_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    return best_params


def adaptive_parameter_estimation():
    """Estimate parameters adaptively based on user data characteristics."""
    
    print("\n" + "=" * 60)
    print("ADAPTIVE PARAMETER ESTIMATION")
    print("=" * 60)
    
    # Load some real user data to analyze patterns
    test_data = [
        ("noisy_user", 2.5, 0.8),      # High variance, high activity
        ("stable_user", 0.3, 0.1),      # Low variance, low activity  
        ("dieting_user", 1.2, 2.5),     # Medium variance, high trend
        ("athlete_user", 3.0, 0.5),     # High daily variance, stable long-term
    ]
    
    print("\nAdaptive parameter recommendations by user type:")
    
    for user_type, daily_var, trend_mag in test_data:
        # Adapt observation noise to daily variance
        obs_covar = min(2.0, max(0.5, daily_var * 0.4))
        
        # Adapt process noise to expected changes
        trans_covar_weight = min(1.0, max(0.1, daily_var * 0.2))
        
        # Adapt trend noise to activity level
        trans_covar_trend = min(0.02, max(0.001, trend_mag * 0.004))
        
        # Adapt threshold to noise level
        extreme_thresh = min(0.4, max(0.2, 0.25 + daily_var * 0.02))
        
        print(f"\n{user_type}:")
        print(f"  Daily variance: {daily_var:.1f} kg")
        print(f"  Trend magnitude: {trend_mag:.1f} kg/month")
        print(f"  Recommended parameters:")
        print(f"    observation_covariance: {obs_covar:.2f}")
        print(f"    transition_covariance_weight: {trans_covar_weight:.2f}")
        print(f"    transition_covariance_trend: {trans_covar_trend:.3f}")
        print(f"    extreme_threshold: {extreme_thresh:.2f}")


if __name__ == "__main__":
    # Run grid search optimization
    optimal_params = grid_search_optimization()
    
    # Show adaptive parameter estimation
    adaptive_parameter_estimation()
    
    print("\n" + "=" * 60)
    print("RECOMMENDED CONFIG CHANGES")
    print("=" * 60)
    print("\nUpdate config.toml [kalman] section with:")
    print(f"transition_covariance_weight = {optimal_params['transition_covariance_weight']}")
    print(f"transition_covariance_trend = {optimal_params['transition_covariance_trend']}")
    print(f"observation_covariance = {optimal_params['observation_covariance']}")
    print(f"\n[processing]")
    print(f"extreme_threshold = {optimal_params['extreme_threshold']}")