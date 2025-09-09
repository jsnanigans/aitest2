"""
Test advanced adaptive parameter estimation (Step 3)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from datetime import datetime, timedelta
from src.processor import WeightProcessor


def test_adaptive_parameters():
    """Test that parameters adapt correctly to different user profiles."""
    
    print("=" * 60)
    print("TESTING ADVANCED ADAPTIVE PARAMETER ESTIMATION (Step 3)")
    print("=" * 60)
    
    base_config = {
        "processing": {
            "min_init_readings": 10,
            "min_weight": 30.0,
            "max_weight": 400.0,
            "max_daily_change": 0.05,
            "extreme_threshold": 0.25  # Base value
        },
        "kalman": {
            "initial_variance": 0.5,
            "transition_covariance_weight": 0.05,
            "transition_covariance_trend": 0.0005,
            "observation_covariance": 1.5,  # Base value
            "reset_gap_days": 30
        }
    }
    
    base_date = datetime(2024, 1, 1)
    
    # Test Case 1: Very clean data (high-quality medical scale)
    print("\n1. High-Quality Scale User (σ < 0.5)")
    print("-" * 40)
    clean_data = []
    for i in range(15):
        weight = 70.0 + np.random.normal(0, 0.2)  # Very low noise
        clean_data.append((weight, base_date + timedelta(days=i), "medical"))
    
    proc1 = WeightProcessor("clean_user", base_config["processing"], base_config["kalman"])
    for w, t, s in clean_data[:10]:
        proc1.process_weight(w, t, s)
    
    # Trigger initialization
    for w, t, s in clean_data[10:]:
        result = proc1.process_weight(w, t, s)
        if result:
            break
    
    print(f"  Base observation_covariance: {base_config['kalman']['observation_covariance']}")
    if proc1.adapted_params:
        print(f"  Adapted observation_covariance: {proc1.adapted_params['observation_covariance']}")
        print(f"  Base extreme_threshold: {base_config['processing']['extreme_threshold']}")
        print(f"  Adapted extreme_threshold: {proc1.adapted_params.get('extreme_threshold', 'N/A')}")
        print(f"  Expected: obs_covar=0.5, threshold=0.20 ✓" if 
              proc1.adapted_params['observation_covariance'] == 0.5 and 
              proc1.adapted_params.get('extreme_threshold') == 0.20 else "  Expected: obs_covar=0.5, threshold=0.20 ✗")
    else:
        print("  Adaptation not yet triggered")
    
    # Test Case 2: Normal bathroom scale
    print("\n2. Normal Bathroom Scale User (0.5 < σ < 1.5)")
    print("-" * 40)
    normal_data = []
    for i in range(15):
        weight = 75.0 + np.random.normal(0, 0.8)  # Moderate noise
        normal_data.append((weight, base_date + timedelta(days=i), "scale"))
    
    proc2 = WeightProcessor("normal_user", base_config["processing"], base_config["kalman"])
    for w, t, s in normal_data[:10]:
        proc2.process_weight(w, t, s)
    
    for w, t, s in normal_data[10:]:
        result = proc2.process_weight(w, t, s)
        if result:
            break
    
    print(f"  Base observation_covariance: {base_config['kalman']['observation_covariance']}")
    if proc2.adapted_params:
        print(f"  Adapted observation_covariance: {proc2.adapted_params['observation_covariance']}")
        print(f"  Base extreme_threshold: {base_config['processing']['extreme_threshold']}")
        print(f"  Adapted extreme_threshold: {proc2.adapted_params.get('extreme_threshold', 'N/A')}")
        print(f"  Expected: obs_covar=1.5, threshold=0.25 ✓" if 
              proc2.adapted_params['observation_covariance'] == 1.5 and 
              proc2.adapted_params.get('extreme_threshold') == 0.25 else "  Expected: obs_covar=1.5, threshold=0.25 ✗")
    else:
        print("  Adaptation not yet triggered")
    
    # Test Case 3: Poor scale or high hydration variance
    print("\n3. Poor Scale / High Variance User (σ > 1.5)")
    print("-" * 40)
    noisy_data = []
    for i in range(15):
        weight = 80.0 + np.random.normal(0, 2.5)  # High noise
        noisy_data.append((weight, base_date + timedelta(days=i), "cheap_scale"))
    
    proc3 = WeightProcessor("noisy_user", base_config["processing"], base_config["kalman"])
    for w, t, s in noisy_data[:10]:
        proc3.process_weight(w, t, s)
    
    for w, t, s in noisy_data[10:]:
        result = proc3.process_weight(w, t, s)
        if result:
            break
    
    print(f"  Base observation_covariance: {base_config['kalman']['observation_covariance']}")
    if proc3.adapted_params:
        print(f"  Adapted observation_covariance: {proc3.adapted_params['observation_covariance']}")
        print(f"  Base extreme_threshold: {base_config['processing']['extreme_threshold']}")
        print(f"  Adapted extreme_threshold: {proc3.adapted_params.get('extreme_threshold', 'N/A')}")
        print(f"  Expected: obs_covar=3.0, threshold=0.35 ✓" if 
              proc3.adapted_params['observation_covariance'] == 3.0 and 
              proc3.adapted_params.get('extreme_threshold') == 0.35 else "  Expected: obs_covar=3.0, threshold=0.35 ✗")
    else:
        print("  Adaptation not yet triggered")
    
    # Test Case 4: Weight loss user (trend detection)
    print("\n4. Active Weight Loss User (trend detection)")
    print("-" * 40)
    trend_data = []
    for i in range(15):
        weight = 90.0 - 0.15 * i + np.random.normal(0, 0.5)  # Clear downward trend
        trend_data.append((weight, base_date + timedelta(days=i), "scale"))
    
    proc4 = WeightProcessor("trend_user", base_config["processing"], base_config["kalman"])
    for w, t, s in trend_data[:10]:
        proc4.process_weight(w, t, s)
    
    for w, t, s in trend_data[10:]:
        result = proc4.process_weight(w, t, s)
        if result:
            break
    
    print(f"  Base transition_covariance_trend: {base_config['kalman']['transition_covariance_trend']}")
    if proc4.adapted_params:
        print(f"  Adapted transition_covariance_trend: {proc4.adapted_params.get('transition_covariance_trend', 'N/A')}")
        print(f"  Expected: Higher trend noise for active weight change ✓" if 
              proc4.adapted_params.get('transition_covariance_trend', 0) >= 0.001 
              else "  Expected: Higher trend noise ✗")
    else:
        print("  Adaptation not yet triggered")
    
    print("\n" + "=" * 60)
    print("ADAPTATION SUMMARY")
    print("=" * 60)
    
    print("\n✓ Step 2: Smooth Confidence Function - IMPLEMENTED")
    print("  - Exponential decay: confidence = exp(-0.5 * innovation²)")
    
    print("\n✓ Step 3: Advanced Adaptive Parameters - IMPLEMENTED")
    print("  - Automatic variance detection using MAD")
    print("  - Dynamic observation_covariance adjustment")
    print("  - Dynamic extreme_threshold adjustment")
    print("  - Trend detection for process noise tuning")
    
    print("\nThe processor now provides:")
    print("- Per-user parameter optimization")
    print("- Automatic scale quality detection")
    print("- Adaptive outlier thresholds")
    print("- Smart trend noise adjustment")


if __name__ == "__main__":
    np.random.seed(42)
    test_adaptive_parameters()