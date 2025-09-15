#!/usr/bin/env python3
"""Test script to demonstrate Kalman filter adjustment issues after gaps."""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.kalman import KalmanFilterManager
from src.constants import KALMAN_DEFAULTS

def generate_test_data():
    """Generate weight data with a gap and trend."""
    np.random.seed(42)
    
    # Pre-gap data: declining trend
    pre_gap_days = 30
    pre_gap_weights = []
    pre_gap_times = []
    base_weight = 85.0
    daily_loss = -0.1  # -0.1 kg/day
    
    for i in range(pre_gap_days):
        weight = base_weight + daily_loss * i + np.random.normal(0, 0.5)
        pre_gap_weights.append(weight)
        pre_gap_times.append(datetime(2024, 1, 1) + timedelta(days=i))
    
    # 18-day gap
    gap_days = 18
    
    # Post-gap data: continuing trend
    post_gap_days = 20
    post_gap_weights = []
    post_gap_times = []
    post_gap_base = base_weight + daily_loss * (pre_gap_days + gap_days)
    
    for i in range(post_gap_days):
        weight = post_gap_base + daily_loss * i + np.random.normal(0, 0.5)
        post_gap_weights.append(weight)
        post_gap_times.append(datetime(2024, 1, 1) + timedelta(days=pre_gap_days + gap_days + i))
    
    return (pre_gap_times, pre_gap_weights, post_gap_times, post_gap_weights)

def test_current_behavior():
    """Test current Kalman filter behavior after gap."""
    pre_times, pre_weights, post_times, post_weights = generate_test_data()
    
    # Process pre-gap data
    state = None
    pre_filtered = []
    
    for t, w in zip(pre_times, pre_weights):
        if state is None:
            state = KalmanFilterManager.initialize_immediate(
                w, t, KALMAN_DEFAULTS, KALMAN_DEFAULTS['observation_covariance']
            )
        else:
            state = KalmanFilterManager.update_state(
                state, w, t, 'test', {}, KALMAN_DEFAULTS['observation_covariance']
            )
        
        result = KalmanFilterManager.create_result(state, w, t, 'test', True)
        if result:
            pre_filtered.append(result['filtered_weight'])
    
    # Reset after gap (current behavior)
    state = KalmanFilterManager.reset_state(state, post_weights[0])
    state['last_timestamp'] = post_times[0]
    
    # Process post-gap data
    post_filtered = []
    post_trends = []
    
    for t, w in zip(post_times, post_weights):
        state = KalmanFilterManager.update_state(
            state, w, t, 'test', {}, KALMAN_DEFAULTS['observation_covariance']
        )
        
        result = KalmanFilterManager.create_result(state, w, t, 'test', True)
        if result:
            post_filtered.append(result['filtered_weight'])
            post_trends.append(result['trend'])
    
    return (pre_times, pre_weights, pre_filtered,
            post_times, post_weights, post_filtered, post_trends)

def test_warmup_approach():
    """Test warmup buffer approach."""
    pre_times, pre_weights, post_times, post_weights = generate_test_data()
    
    # Collect warmup measurements
    warmup_size = 3
    warmup_weights = post_weights[:warmup_size]
    warmup_times = post_times[:warmup_size]
    
    # Estimate initial trend from warmup
    if len(warmup_weights) >= 2:
        time_diff = (warmup_times[-1] - warmup_times[0]).total_seconds() / 86400
        weight_diff = warmup_weights[-1] - warmup_weights[0]
        initial_trend = weight_diff / time_diff if time_diff > 0 else 0
    else:
        initial_trend = 0
    
    # Initialize with estimated trend
    initial_weight = np.mean(warmup_weights)
    
    # Modified initialization with trend
    state = {
        'kalman_params': {
            'initial_state_mean': [initial_weight, initial_trend],
            'initial_state_covariance': [[KALMAN_DEFAULTS['initial_variance'], 0], 
                                        [0, 0.01]],  # Higher trend variance
            'transition_covariance': [
                [KALMAN_DEFAULTS['transition_covariance_weight'], 0],
                [0, KALMAN_DEFAULTS['transition_covariance_trend']]
            ],
            'observation_covariance': [[KALMAN_DEFAULTS['observation_covariance']]],
        },
        'last_state': np.array([[initial_weight, initial_trend]]),
        'last_covariance': np.array([[[KALMAN_DEFAULTS['initial_variance'], 0], 
                                     [0, 0.01]]]),
        'last_timestamp': warmup_times[-1],
        'last_raw_weight': initial_weight,
    }
    
    # Process remaining post-gap data
    post_filtered_warmup = []
    post_trends_warmup = []
    
    for t, w in zip(post_times[warmup_size:], post_weights[warmup_size:]):
        state = KalmanFilterManager.update_state(
            state, w, t, 'test', {}, KALMAN_DEFAULTS['observation_covariance']
        )
        
        result = KalmanFilterManager.create_result(state, w, t, 'test', True)
        if result:
            post_filtered_warmup.append(result['filtered_weight'])
            post_trends_warmup.append(result['trend'])
    
    return warmup_weights + post_filtered_warmup, [initial_trend] * warmup_size + post_trends_warmup

def test_adaptive_covariance():
    """Test adaptive covariance approach."""
    pre_times, pre_weights, post_times, post_weights = generate_test_data()
    
    # Reset with higher covariances
    gap_days = 18
    gap_factor = min(gap_days / 30, 3.0)
    
    initial_variance = KALMAN_DEFAULTS['initial_variance'] * (1 + gap_factor)
    trend_variance = 0.001 * (1 + gap_factor * 10)  # Much higher trend variance
    
    state = {
        'kalman_params': {
            'initial_state_mean': [post_weights[0], 0],
            'initial_state_covariance': [[initial_variance, 0], [0, trend_variance]],
            'transition_covariance': [
                [KALMAN_DEFAULTS['transition_covariance_weight'] * 2, 0],
                [0, KALMAN_DEFAULTS['transition_covariance_trend'] * 10]
            ],
            'observation_covariance': [[KALMAN_DEFAULTS['observation_covariance']]],
        },
        'last_state': np.array([[post_weights[0], 0]]),
        'last_covariance': np.array([[[initial_variance, 0], [0, trend_variance]]]),
        'last_timestamp': post_times[0],
        'last_raw_weight': post_weights[0],
    }
    
    # Process post-gap data with decaying adaptation
    post_filtered_adaptive = []
    post_trends_adaptive = []
    
    for i, (t, w) in enumerate(zip(post_times, post_weights)):
        # Gradually reduce process noise
        adaptation_factor = np.exp(-i / 3)
        current_state = state.copy()
        current_state['kalman_params']['transition_covariance'] = [
            [KALMAN_DEFAULTS['transition_covariance_weight'] * (1 + adaptation_factor), 0],
            [0, KALMAN_DEFAULTS['transition_covariance_trend'] * (1 + 10 * adaptation_factor)]
        ]
        
        state = KalmanFilterManager.update_state(
            current_state, w, t, 'test', {}, KALMAN_DEFAULTS['observation_covariance']
        )
        
        result = KalmanFilterManager.create_result(state, w, t, 'test', True)
        if result:
            post_filtered_adaptive.append(result['filtered_weight'])
            post_trends_adaptive.append(result['trend'])
    
    return post_filtered_adaptive, post_trends_adaptive

def plot_comparison():
    """Plot comparison of different approaches."""
    # Get test results
    (pre_times, pre_weights, pre_filtered,
     post_times, post_weights, post_filtered, post_trends) = test_current_behavior()
    
    warmup_filtered, warmup_trends = test_warmup_approach()
    adaptive_filtered, adaptive_trends = test_adaptive_covariance()
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot weights
    ax1 = axes[0]
    ax1.scatter(pre_times, pre_weights, alpha=0.3, label='Pre-gap raw', color='gray')
    ax1.plot(pre_times, pre_filtered, 'b-', label='Pre-gap filtered', linewidth=2)
    
    ax1.scatter(post_times, post_weights, alpha=0.3, color='gray')
    ax1.plot(post_times, post_filtered, 'r-', label='Current (zero trend)', linewidth=2)
    ax1.plot(post_times, warmup_filtered, 'g-', label='Warmup approach', linewidth=2)
    ax1.plot(post_times, adaptive_filtered, 'm-', label='Adaptive covariance', linewidth=2)
    
    # Mark gap
    ax1.axvspan(pre_times[-1], post_times[0], alpha=0.2, color='yellow', label='18-day gap')
    
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Weight (kg)')
    ax1.set_title('Kalman Filter Behavior After Gap - Weight Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot trends
    ax2 = axes[1]
    ax2.plot(post_times, np.array(post_trends) * 7, 'r-', label='Current (weekly)', linewidth=2)
    ax2.plot(post_times, np.array(warmup_trends) * 7, 'g-', label='Warmup (weekly)', linewidth=2)
    ax2.plot(post_times, np.array(adaptive_trends) * 7, 'm-', label='Adaptive (weekly)', linewidth=2)
    
    # Add reference line for actual trend
    actual_trend = -0.1 * 7  # -0.7 kg/week
    ax2.axhline(y=actual_trend, color='black', linestyle='--', alpha=0.5, label=f'Actual trend ({actual_trend:.1f} kg/week)')
    
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Weekly Trend (kg/week)')
    ax2.set_title('Kalman Filter Trend Estimation After Gap')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('docs/images/kalman_gap_adjustment_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved comparison plot to docs/images/kalman_gap_adjustment_comparison.png")
    
    # Print statistics
    print("\n=== Performance Metrics ===")
    
    # Calculate RMSE for each approach
    actual_post = [85.0 - 0.1 * (30 + 18 + i) for i in range(len(post_weights))]
    
    rmse_current = np.sqrt(np.mean((np.array(post_filtered) - actual_post)**2))
    rmse_warmup = np.sqrt(np.mean((np.array(warmup_filtered) - actual_post)**2))
    rmse_adaptive = np.sqrt(np.mean((np.array(adaptive_filtered) - actual_post)**2))
    
    print(f"RMSE - Current approach: {rmse_current:.3f} kg")
    print(f"RMSE - Warmup approach: {rmse_warmup:.3f} kg")
    print(f"RMSE - Adaptive approach: {rmse_adaptive:.3f} kg")
    
    # Trend convergence time (measurements to get within 20% of actual)
    target_trend = -0.1
    tolerance = 0.02  # 20% of 0.1
    
    def find_convergence(trends):
        for i, t in enumerate(trends):
            if abs(t - target_trend) < tolerance:
                return i + 1
        return len(trends)
    
    conv_current = find_convergence(post_trends)
    conv_warmup = find_convergence(warmup_trends)
    conv_adaptive = find_convergence(adaptive_trends)
    
    print(f"\nTrend convergence (measurements):")
    print(f"  Current: {conv_current}")
    print(f"  Warmup: {conv_warmup}")
    print(f"  Adaptive: {conv_adaptive}")

if __name__ == "__main__":
    plot_comparison()
