"""
Adaptive Kalman filter optimization with per-user parameter tuning
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.processor import WeightProcessor


class AdaptiveKalmanOptimizer:
    """
    Adaptive parameter estimation for Kalman filter based on user data characteristics.
    Implements council recommendations for simplified, user-specific tuning.
    """
    
    def __init__(self):
        self.default_params = {
            'transition_covariance_weight': 0.5,
            'transition_covariance_trend': 0.01,
            'observation_covariance': 1.0,
            'extreme_threshold': 0.3
        }
        
    def analyze_user_characteristics(self, readings: List[Tuple[float, datetime]]) -> Dict:
        """Analyze user data to determine characteristics."""
        if len(readings) < 10:
            return {
                'noise_level': 'unknown',
                'trend_type': 'unknown',
                'variance': 1.0,
                'daily_change': 0.01
            }
        
        weights = [w for w, _ in readings]
        timestamps = [t for _, t in readings]
        
        # Calculate daily changes
        daily_changes = []
        for i in range(1, len(weights)):
            days_diff = (timestamps[i] - timestamps[i-1]).days
            if days_diff > 0 and days_diff < 7:  # Reasonable time gap
                daily_change = abs(weights[i] - weights[i-1]) / days_diff
                daily_changes.append(daily_change)
        
        # Estimate noise level using median absolute deviation
        median_weight = np.median(weights)
        mad = np.median([abs(w - median_weight) for w in weights])
        variance = mad * 1.4826  # Convert MAD to std deviation estimate
        
        # Classify noise level
        if variance < 0.5:
            noise_level = 'low'
        elif variance < 1.5:
            noise_level = 'medium'
        else:
            noise_level = 'high'
        
        # Detect trend
        if len(weights) > 20:
            first_third = np.mean(weights[:len(weights)//3])
            last_third = np.mean(weights[-len(weights)//3:])
            trend_magnitude = (last_third - first_third) / len(weights)
            
            if abs(trend_magnitude) < 0.01:
                trend_type = 'stable'
            elif trend_magnitude < -0.03:
                trend_type = 'losing'
            elif trend_magnitude > 0.03:
                trend_type = 'gaining'
            else:
                trend_type = 'moderate'
        else:
            trend_type = 'unknown'
        
        return {
            'noise_level': noise_level,
            'trend_type': trend_type,
            'variance': variance,
            'daily_change': np.median(daily_changes) if daily_changes else 0.01
        }
    
    def optimize_parameters(self, characteristics: Dict) -> Dict:
        """
        Optimize Kalman parameters based on user characteristics.
        Following council advice: simpler is better, smooth confidence functions.
        """
        
        params = self.default_params.copy()
        
        # Adjust observation noise based on data variance
        # Higher variance → higher observation noise (less trust in measurements)
        variance = characteristics['variance']
        params['observation_covariance'] = min(3.0, max(0.5, variance * 1.5))
        
        # Adjust process noise based on daily changes
        # Higher daily variation → higher process noise
        daily_change = characteristics['daily_change']
        params['transition_covariance_weight'] = min(1.0, max(0.05, daily_change * 10))
        
        # Adjust trend noise based on trend type
        trend_map = {
            'stable': 0.001,     # Very low trend noise for stable weight
            'moderate': 0.005,   # Moderate trend noise
            'losing': 0.01,      # Higher for active weight loss
            'gaining': 0.01,     # Higher for weight gain
            'unknown': 0.005     # Default moderate
        }
        params['transition_covariance_trend'] = trend_map.get(
            characteristics['trend_type'], 0.005
        )
        
        # Adjust extreme threshold based on noise level
        # Higher noise → more tolerant threshold
        noise_map = {
            'low': 0.2,       # Tight threshold for clean data
            'medium': 0.25,   # Moderate threshold
            'high': 0.35,     # Looser threshold for noisy data
            'unknown': 0.3    # Default
        }
        params['extreme_threshold'] = noise_map.get(
            characteristics['noise_level'], 0.3
        )
        
        return params
    
    def calculate_smooth_confidence(self, normalized_innovation: float, alpha: float = 0.5) -> float:
        """
        Smooth exponential confidence function as recommended by Knuth.
        confidence = exp(-alpha * normalized_innovation^2)
        """
        return np.exp(-alpha * normalized_innovation ** 2)
    
    def validate_parameters(self, params: Dict, test_data: List) -> Dict:
        """Validate parameters on test data and return performance metrics."""
        
        metrics = {
            'mse': [],
            'max_error': [],
            'outliers_caught': 0,
            'false_rejections': 0,
            'recovery_samples': []
        }
        
        for user_id, readings, expected_outliers in test_data:
            processing_config = {
                "min_init_readings": 10,
                "min_weight": 30.0,
                "max_weight": 400.0,
                "max_daily_change": 0.05,
                "extreme_threshold": params['extreme_threshold']
            }
            
            kalman_config = {
                "initial_variance": 1.0,
                "transition_covariance_weight": params['transition_covariance_weight'],
                "transition_covariance_trend": params['transition_covariance_trend'],
                "observation_covariance": params['observation_covariance'],
                "reset_gap_days": 30
            }
            
            processor = WeightProcessor(user_id, processing_config, kalman_config)
            
            results = []
            for i, (weight, timestamp, source) in enumerate(readings):
                result = processor.process_weight(weight, timestamp, source)
                if result:
                    results.append((i, result))
            
            # Calculate metrics
            for idx, result in results:
                if idx in expected_outliers:
                    if not result['accepted']:
                        metrics['outliers_caught'] += 1
                else:
                    if not result['accepted']:
                        metrics['false_rejections'] += 1
            
            # MSE on accepted readings
            accepted_weights = [r['filtered_weight'] for _, r in results if r['accepted']]
            if accepted_weights:
                true_weights = [w for i, (w, _, _) in enumerate(readings) if i not in expected_outliers]
                if true_weights:
                    mse = np.mean([(a - np.median(true_weights))**2 for a in accepted_weights[-10:]])
                    metrics['mse'].append(mse)
        
        return {
            'avg_mse': np.mean(metrics['mse']) if metrics['mse'] else 0,
            'outlier_detection_rate': metrics['outliers_caught'] / max(1, len(expected_outliers) * len(test_data)),
            'false_rejection_rate': metrics['false_rejections'] / max(1, sum(len(r) for _, r, _ in test_data))
        }


def demonstrate_adaptive_optimization():
    """Demonstrate adaptive parameter optimization with various user types."""
    
    optimizer = AdaptiveKalmanOptimizer()
    base_date = datetime(2024, 1, 1)
    
    # Create test users with different characteristics
    test_users = {
        'stable_user': {
            'description': 'Stable weight, low noise (good scale)',
            'readings': [(70.0 + np.random.normal(0, 0.3), base_date + timedelta(days=i)) 
                        for i in range(60)],
            'expected_params': {
                'observation_covariance': 0.5,  # Low noise
                'transition_covariance_weight': 0.05,  # Very stable
            }
        },
        'dieting_user': {
            'description': 'Active weight loss with moderate noise',
            'readings': [(85.0 - 0.1*i + np.random.normal(0, 0.5), base_date + timedelta(days=i))
                        for i in range(60)],
            'expected_params': {
                'observation_covariance': 0.75,  # Moderate noise
                'transition_covariance_weight': 0.3,  # Active change
            }
        },
        'noisy_user': {
            'description': 'High variance (poor scale or hydration swings)',
            'readings': [(75.0 + np.random.normal(0, 2.0), base_date + timedelta(days=i))
                        for i in range(60)],
            'expected_params': {
                'observation_covariance': 3.0,  # High noise
                'extreme_threshold': 0.35,  # More tolerant
            }
        },
        'athlete_user': {
            'description': 'High daily variance but stable long-term',
            'readings': [(68.0 + 2*np.sin(2*np.pi*i/7) + np.random.normal(0, 0.5), 
                         base_date + timedelta(days=i)) for i in range(60)],
            'expected_params': {
                'observation_covariance': 2.0,  # High variance
                'transition_covariance_weight': 0.5,  # Expect changes
            }
        }
    }
    
    print("=" * 60)
    print("ADAPTIVE KALMAN PARAMETER OPTIMIZATION")
    print("=" * 60)
    
    results = {}
    
    for user_type, user_data in test_users.items():
        print(f"\n{user_type}: {user_data['description']}")
        print("-" * 40)
        
        # Analyze characteristics
        characteristics = optimizer.analyze_user_characteristics(user_data['readings'])
        print(f"Detected characteristics:")
        print(f"  Noise level: {characteristics['noise_level']}")
        print(f"  Trend type: {characteristics['trend_type']}")
        print(f"  Variance: {characteristics['variance']:.2f} kg")
        print(f"  Daily change: {characteristics['daily_change']:.3f} kg/day")
        
        # Optimize parameters
        optimized_params = optimizer.optimize_parameters(characteristics)
        print(f"\nOptimized parameters:")
        for key, value in optimized_params.items():
            expected = user_data['expected_params'].get(key)
            if expected:
                match = "✓" if abs(value - expected) < expected * 0.5 else "⚠"
                print(f"  {key}: {value:.3f} (expected ~{expected:.3f}) {match}")
            else:
                print(f"  {key}: {value:.3f}")
        
        results[user_type] = {
            'characteristics': characteristics,
            'parameters': optimized_params
        }
    
    # Test smooth confidence function
    print("\n" + "=" * 60)
    print("SMOOTH CONFIDENCE FUNCTION (Knuth recommendation)")
    print("=" * 60)
    
    innovations = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
    print("\nNormalized Innovation → Confidence:")
    for inn in innovations:
        old_conf = calculate_old_confidence(inn)
        new_conf = optimizer.calculate_smooth_confidence(inn)
        print(f"  {inn:.1f}σ: old={old_conf:.2f}, smooth={new_conf:.3f}")
    
    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'user_analyses': results,
        'recommendations': {
            'use_adaptive': True,
            'smooth_confidence': True,
            'simplified_params': ['observation_covariance', 'extreme_threshold'],
            'advanced_params': ['transition_covariance_weight', 'transition_covariance_trend']
        }
    }
    
    with open('output/adaptive_kalman_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    print("\n1. SIMPLIFICATION (Butler Lampson):")
    print("   - Reduce to 2 key parameters for most users:")
    print("     • observation_covariance (trust in measurements)")
    print("     • extreme_threshold (outlier tolerance)")
    
    print("\n2. ADAPTIVE TUNING (Martin Kleppmann):")
    print("   - Estimate observation_covariance from initial data variance")
    print("   - Adjust per-user based on scale quality")
    
    print("\n3. SMOOTH CONFIDENCE (Donald Knuth):")
    print("   - Replace step function with exponential decay")
    print("   - confidence = exp(-0.5 * normalized_innovation²)")
    
    print("\n4. MULTI-SCALE APPROACH (Leslie Lamport):")
    print("   - Consider separate filters for:")
    print("     • Daily noise (hydration, meals)")
    print("     • Weekly cycles (workout patterns)")
    print("     • Monthly trends (actual weight change)")


def calculate_old_confidence(normalized_innovation: float) -> float:
    """Old step-function confidence calculation for comparison."""
    if normalized_innovation <= 0.5:
        return 0.99
    elif normalized_innovation <= 1.0:
        return 0.95
    elif normalized_innovation <= 2.0:
        return 0.85
    elif normalized_innovation <= 3.0:
        return 0.70
    elif normalized_innovation <= 4.0:
        return 0.50
    elif normalized_innovation <= 5.0:
        return 0.30
    else:
        return 0.10


if __name__ == "__main__":
    demonstrate_adaptive_optimization()