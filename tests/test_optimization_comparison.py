"""
Compare current vs optimized Kalman implementations
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from datetime import datetime, timedelta
import pandas as pd
from processor import WeightProcessor
from processor_optimized import OptimizedWeightProcessor


def load_test_data(csv_file: str, max_users: int = 5):
    """Load real data for testing."""
    df = pd.read_csv(csv_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    users = df['userId'].unique()[:max_users]
    
    test_data = []
    for user_id in users:
        user_data = df[df['userId'] == user_id].sort_values('timestamp')
        readings = [
            (row['weight'], row['timestamp'].to_pydatetime(), row.get('source', 'scale'))
            for _, row in user_data.iterrows()
        ]
        test_data.append((user_id, readings))
    
    return test_data


def compare_processors(test_data):
    """Compare current and optimized processors."""
    
    print("=" * 60)
    print("KALMAN FILTER OPTIMIZATION COMPARISON")
    print("=" * 60)
    
    # Current configuration
    current_config = {
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
    }
    
    # Optimized configuration (from grid search)
    optimized_config = {
        "processing": {
            "min_init_readings": 10,
            "min_weight": 30.0,
            "max_weight": 400.0,
            "max_daily_change": 0.05,  # Increased tolerance
            "extreme_threshold": 0.25   # Tighter from optimization
        },
        "kalman": {
            "initial_variance": 1.0,
            "transition_covariance_weight": 0.1,  # Much lower from optimization
            "transition_covariance_trend": 0.001,  # Lower from optimization
            "observation_covariance": 2.0,  # Higher from optimization
            "reset_gap_days": 30
        }
    }
    
    results_comparison = []
    
    for user_id, readings in test_data:
        print(f"\nProcessing user: {user_id[:8]}...")
        
        # Current processor
        current_proc = WeightProcessor(
            user_id,
            current_config["processing"],
            current_config["kalman"]
        )
        
        # Optimized processor (using new adaptive class)
        optimized_proc = OptimizedWeightProcessor(
            user_id,
            optimized_config["processing"],
            optimized_config["kalman"]
        )
        
        current_results = []
        optimized_results = []
        
        for weight, timestamp, source in readings:
            # Process with current
            curr_result = current_proc.process_weight(weight, timestamp, source)
            if curr_result:
                current_results.append(curr_result)
            
            # Process with optimized
            opt_result = optimized_proc.process_weight(weight, timestamp, source)
            if opt_result:
                optimized_results.append(opt_result)
        
        # Calculate metrics
        if current_results and optimized_results:
            # Acceptance rates
            curr_accepted = sum(1 for r in current_results if r['accepted'])
            opt_accepted = sum(1 for r in optimized_results if r['accepted'])
            
            curr_rate = curr_accepted / len(current_results) * 100
            opt_rate = opt_accepted / len(optimized_results) * 100
            
            # Average confidence
            curr_conf = np.mean([r['confidence'] for r in current_results if r['accepted']])
            opt_conf = np.mean([r['confidence'] for r in optimized_results if r['accepted']])
            
            # Stability (variance of filtered weights)
            curr_weights = [r['filtered_weight'] for r in current_results if r['accepted']]
            opt_weights = [r['filtered_weight'] for r in optimized_results if r['accepted']]
            
            curr_stability = np.std(curr_weights) if len(curr_weights) > 1 else 0
            opt_stability = np.std(opt_weights) if len(opt_weights) > 1 else 0
            
            # Trend estimation
            curr_trends = [r['trend'] for r in current_results[-10:] if r['accepted']]
            opt_trends = [r['filtered_weight'] for r in optimized_results[-10:] if r['accepted']]
            
            results_comparison.append({
                'user_id': user_id[:8],
                'current_acceptance': curr_rate,
                'optimized_acceptance': opt_rate,
                'current_confidence': curr_conf,
                'optimized_confidence': opt_conf,
                'current_stability': curr_stability,
                'optimized_stability': opt_stability,
                'readings_processed': len(readings)
            })
    
    # Print comparison table
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    print("\n%-10s %8s %8s %8s %8s" % (
        "User", "Curr%", "Opt%", "Curr σ", "Opt σ"
    ))
    print("-" * 50)
    
    for result in results_comparison:
        print("%-10s %7.1f%% %7.1f%% %7.2f %7.2f" % (
            result['user_id'],
            result['current_acceptance'],
            result['optimized_acceptance'],
            result['current_stability'],
            result['optimized_stability']
        ))
    
    # Calculate averages
    avg_curr_accept = np.mean([r['current_acceptance'] for r in results_comparison])
    avg_opt_accept = np.mean([r['optimized_acceptance'] for r in results_comparison])
    avg_curr_stability = np.mean([r['current_stability'] for r in results_comparison])
    avg_opt_stability = np.mean([r['optimized_stability'] for r in results_comparison])
    
    print("-" * 50)
    print("%-10s %7.1f%% %7.1f%% %7.2f %7.2f" % (
        "AVERAGE",
        avg_curr_accept,
        avg_opt_accept,
        avg_curr_stability,
        avg_opt_stability
    ))
    
    print("\n" + "=" * 60)
    print("KEY IMPROVEMENTS")
    print("=" * 60)
    
    improvements = []
    
    if avg_opt_accept > avg_curr_accept:
        improvements.append(f"✓ Acceptance rate improved by {avg_opt_accept - avg_curr_accept:.1f}%")
    else:
        improvements.append(f"⚠ Acceptance rate decreased by {avg_curr_accept - avg_opt_accept:.1f}%")
    
    if avg_opt_stability < avg_curr_stability:
        improvements.append(f"✓ Stability improved (lower variance) by {(1 - avg_opt_stability/avg_curr_stability)*100:.1f}%")
    else:
        improvements.append(f"⚠ Stability decreased by {(avg_opt_stability/avg_curr_stability - 1)*100:.1f}%")
    
    improvements.append("\n✓ Adaptive parameter tuning based on user data")
    improvements.append("✓ Smooth exponential confidence function")
    improvements.append("✓ Simplified parameter set (2 key params vs 7)")
    improvements.append("✓ Per-user noise estimation")
    
    for improvement in improvements:
        print(improvement)
    
    print("\n" + "=" * 60)
    print("OPTIMIZED PARAMETERS")
    print("=" * 60)
    print("\nRecommended changes to config.toml:")
    print("\n[kalman]")
    print("# Optimized from grid search")
    print("transition_covariance_weight = 0.1  # was 0.5")
    print("transition_covariance_trend = 0.001  # was 0.01")
    print("observation_covariance = 2.0  # was 1.0")
    print("\n[processing]")
    print("extreme_threshold = 0.25  # was 0.30")
    print("max_daily_change = 0.05  # was 0.03")


if __name__ == "__main__":
    # Use the test data file
    csv_file = "./data/test_sample.csv"
    
    if not os.path.exists(csv_file):
        # Create synthetic test data if file doesn't exist
        print("Creating synthetic test data...")
        import pandas as pd
        
        np.random.seed(42)
        base_date = datetime(2024, 1, 1)
        
        data = []
        for user_idx in range(5):
            user_id = f"user_{user_idx:03d}"
            base_weight = 60 + user_idx * 10
            
            for day in range(60):
                # Add some patterns
                if user_idx == 0:  # Stable
                    weight = base_weight + np.random.normal(0, 0.5)
                elif user_idx == 1:  # Weight loss
                    weight = base_weight - day * 0.05 + np.random.normal(0, 0.5)
                elif user_idx == 2:  # Noisy
                    weight = base_weight + np.random.normal(0, 2.0)
                elif user_idx == 3:  # Cyclic
                    weight = base_weight + 2 * np.sin(2 * np.pi * day / 7) + np.random.normal(0, 0.5)
                else:  # Outliers
                    if day % 10 == 0:
                        weight = base_weight + np.random.uniform(5, 10)
                    else:
                        weight = base_weight + np.random.normal(0, 0.5)
                
                data.append({
                    'userId': user_id,
                    'weight': weight,
                    'timestamp': base_date + timedelta(days=day),
                    'source': 'scale'
                })
        
        df = pd.DataFrame(data)
        df.to_csv(csv_file, index=False)
        print(f"Created test data: {csv_file}")
    
    # Load and compare
    test_data = load_test_data(csv_file, max_users=5)
    compare_processors(test_data)