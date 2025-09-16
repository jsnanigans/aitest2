#!/usr/bin/env python3
"""Improved Kalman filter gap handling with hybrid approach."""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.kalman import KalmanFilterManager
from src.constants import KALMAN_DEFAULTS

class ImprovedKalmanManager:
    """Enhanced Kalman filter manager with better gap handling."""
    
    @staticmethod
    def calculate_adaptive_parameters(gap_days: float) -> Dict[str, float]:
        """Calculate adaptive parameters based on gap duration."""
        # Base parameters
        base_variance = KALMAN_DEFAULTS['initial_variance']
        base_trend_var = 0.001
        base_trans_weight = KALMAN_DEFAULTS['transition_covariance_weight']
        base_trans_trend = KALMAN_DEFAULTS['transition_covariance_trend']
        
        # Calculate gap factor (0 to 3, saturates at 90 days)
        gap_factor = min(gap_days / 30, 3.0)
        
        # Increase uncertainties based on gap
        initial_variance = base_variance * (1 + gap_factor * 2)  # Up to 3x
        trend_variance = base_trend_var * (1 + gap_factor * 20)  # Up to 21x
        
        # Temporarily increase process noise for faster adaptation
        trans_weight = base_trans_weight * (1 + gap_factor * 3)  # Up to 4x
        trans_trend = base_trans_trend * (1 + gap_factor * 50)  # Up to 51x
        
        return {
            'initial_variance': initial_variance,
            'trend_variance': trend_variance,
            'transition_weight': trans_weight,
            'transition_trend': trans_trend,
            'gap_factor': gap_factor
        }
    
    @staticmethod
    def estimate_initial_trend(
        measurements: List[float],
        timestamps: List[datetime],
        pre_gap_trend: Optional[float] = None
    ) -> float:
        """Estimate initial trend from measurements."""
        if len(measurements) < 2:
            return pre_gap_trend * 0.5 if pre_gap_trend else 0.0
        
        # Calculate trend from first and last measurements
        time_diff = (timestamps[-1] - timestamps[0]).total_seconds() / 86400
        if time_diff < 0.1:
            return pre_gap_trend * 0.5 if pre_gap_trend else 0.0
        
        weight_diff = measurements[-1] - measurements[0]
        measured_trend = weight_diff / time_diff
        
        # If we have pre-gap trend, blend it with measured trend
        if pre_gap_trend is not None:
            # Decay pre-gap trend based on gap duration
            gap_days = (timestamps[0] - timestamps[-1]).total_seconds() / 86400
            decay_factor = np.exp(-gap_days / 30)  # Half-life of 30 days
            blended_trend = pre_gap_trend * decay_factor * 0.3 + measured_trend * 0.7
            return blended_trend
        
        return measured_trend
    
    @staticmethod
    def initialize_with_warmup(
        warmup_data: List[Dict[str, Any]],
        gap_days: float,
        kalman_config: dict,
        pre_gap_state: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Initialize Kalman filter using warmup data."""
        if not warmup_data:
            raise ValueError("Warmup data cannot be empty")
        
        # Extract measurements and timestamps
        weights = [d['weight'] for d in warmup_data]
        timestamps = [d['timestamp'] for d in warmup_data]
        
        # Get pre-gap trend if available
        pre_gap_trend = None
        if pre_gap_state and pre_gap_state.get('last_state') is not None:
            _, pre_gap_trend = KalmanFilterManager.get_current_state_values(pre_gap_state)
        
        # Estimate initial values
        initial_weight = np.median(weights)  # More robust than mean
        initial_trend = ImprovedKalmanManager.estimate_initial_trend(
            weights, timestamps, pre_gap_trend
        )
        
        # Get adaptive parameters
        params = ImprovedKalmanManager.calculate_adaptive_parameters(gap_days)
        
        # Create enhanced initial state
        state = {
            'kalman_params': {
                'initial_state_mean': [initial_weight, initial_trend],
                'initial_state_covariance': [
                    [params['initial_variance'], 0],
                    [0, params['trend_variance']]
                ],
                'transition_covariance': [
                    [params['transition_weight'], 0],
                    [0, params['transition_trend']]
                ],
                'observation_covariance': [[kalman_config.get('observation_covariance', 3.49)]],
            },
            'last_state': np.array([[initial_weight, initial_trend]]),
            'last_covariance': np.array([[
                [params['initial_variance'], 0],
                [0, params['trend_variance']]
            ]]),
            'last_timestamp': timestamps[-1],
            'last_raw_weight': weights[-1],
            'gap_adaptation': {
                'gap_days': gap_days,
                'gap_factor': params['gap_factor'],
                'measurements_since_gap': len(warmup_data),
                'warmup_size': len(warmup_data),
                'initial_trend': initial_trend
            }
        }
        
        return state
    
    @staticmethod
    def update_with_adaptation(
        state: Dict[str, Any],
        weight: float,
        timestamp: datetime,
        source: str,
        processing_config: dict,
        observation_covariance: Optional[float] = None
    ) -> Dict[str, Any]:
        """Update state with adaptive parameters that decay over time."""
        # Check if we're in adaptation mode
        if 'gap_adaptation' in state:
            gap_info = state['gap_adaptation']
            measurements_since = gap_info['measurements_since_gap']
            
            # Calculate decay factor (exponential decay over ~10 measurements)
            decay_factor = np.exp(-measurements_since / 5)
            
            if decay_factor > 0.01:  # Still adapting
                # Get base parameters
                base_trans_weight = KALMAN_DEFAULTS['transition_covariance_weight']
                base_trans_trend = KALMAN_DEFAULTS['transition_covariance_trend']
                
                # Apply decaying adaptation
                gap_factor = gap_info['gap_factor']
                adapted_weight = base_trans_weight * (1 + gap_factor * 3 * decay_factor)
                adapted_trend = base_trans_trend * (1 + gap_factor * 50 * decay_factor)
                
                # Update transition covariance
                state['kalman_params']['transition_covariance'] = [
                    [adapted_weight, 0],
                    [0, adapted_trend]
                ]
            else:
                # End adaptation mode
                del state['gap_adaptation']
                # Reset to normal parameters
                state['kalman_params']['transition_covariance'] = [
                    [KALMAN_DEFAULTS['transition_covariance_weight'], 0],
                    [0, KALMAN_DEFAULTS['transition_covariance_trend']]
                ]
            
            # Increment measurement counter
            if 'gap_adaptation' in state:
                state['gap_adaptation']['measurements_since_gap'] += 1
        
        # Perform normal update
        return KalmanFilterManager.update_state(
            state, weight, timestamp, source, processing_config, observation_covariance
        )

class WarmupBuffer:
    """Manages warmup data collection after gaps."""
    
    def __init__(self, target_size: int = 3, max_time_span_days: float = 7.0):
        self.target_size = target_size
        self.max_time_span_days = max_time_span_days
        self.buffer: List[Dict[str, Any]] = []
        self.is_complete = False
    
    def add(self, weight: float, timestamp: datetime, source: str) -> bool:
        """Add measurement to buffer. Returns True if buffer is complete."""
        self.buffer.append({
            'weight': weight,
            'timestamp': timestamp,
            'source': source
        })
        
        # Check if buffer is complete
        if len(self.buffer) >= self.target_size:
            self.is_complete = True
        elif len(self.buffer) >= 2:
            # Check time span
            time_span = (self.buffer[-1]['timestamp'] - self.buffer[0]['timestamp']).total_seconds() / 86400
            if time_span >= self.max_time_span_days:
                self.is_complete = True
        
        return self.is_complete
    
    def get_data(self) -> List[Dict[str, Any]]:
        """Get buffer data."""
        return self.buffer
    
    def clear(self):
        """Clear buffer."""
        self.buffer = []
        self.is_complete = False

def demonstrate_improvement():
    """Demonstrate the improved gap handling."""
    import matplotlib.pyplot as plt
    
    # Generate test data
    np.random.seed(42)
    
    # Scenario: Weight loss with 18-day gap
    timestamps = []
    weights = []
    
    # Pre-gap: 30 days of gradual weight loss
    base_weight = 85.0
    for i in range(30):
        t = datetime(2024, 1, 1) + timedelta(days=i)
        w = base_weight - 0.15 * i + np.random.normal(0, 0.5)
        timestamps.append(t)
        weights.append(w)
    
    # 18-day gap
    gap_start = timestamps[-1]
    gap_end = gap_start + timedelta(days=18)
    
    # Post-gap: Continue weight loss trend
    for i in range(20):
        t = gap_end + timedelta(days=i)
        w = base_weight - 0.15 * (30 + 18 + i) + np.random.normal(0, 0.5)
        timestamps.append(t)
        weights.append(w)
    
    # Process with improved handler
    state = None
    warmup_buffer = WarmupBuffer(target_size=3)
    results = []
    in_warmup = False
    gap_days = 0
    
    for t, w in zip(timestamps, weights):
        if state is None:
            # Initialize
            state = KalmanFilterManager.initialize_immediate(
                w, t, KALMAN_DEFAULTS, KALMAN_DEFAULTS['observation_covariance']
            )
            result = KalmanFilterManager.create_result(state, w, t, 'test', True)
        else:
            # Check for gap
            time_delta = (t - state['last_timestamp']).total_seconds() / 86400
            
            if time_delta > 10 and not in_warmup:  # Gap detected
                gap_days = time_delta
                in_warmup = True
                warmup_buffer.clear()
                print(f"Gap detected: {gap_days:.1f} days")
            
            if in_warmup:
                # Collect warmup data
                if warmup_buffer.add(w, t, 'test'):
                    # Warmup complete, initialize with enhanced method
                    print(f"Warmup complete with {len(warmup_buffer.buffer)} measurements")
                    state = ImprovedKalmanManager.initialize_with_warmup(
                        warmup_buffer.get_data(),
                        gap_days,
                        KALMAN_DEFAULTS,
                        pre_gap_state=state
                    )
                    in_warmup = False
                    
                    # Create results for warmup period
                    for data in warmup_buffer.buffer:
                        result = KalmanFilterManager.create_result(
                            state, data['weight'], data['timestamp'], 'test', True
                        )
                        if result:
                            result['in_warmup'] = True
                            results.append(result)
                    continue
                else:
                    # Still collecting warmup
                    result = {'raw_weight': w, 'timestamp': t, 'in_warmup': True}
            else:
                # Normal processing with adaptation
                state = ImprovedKalmanManager.update_with_adaptation(
                    state, w, t, 'test', {}, KALMAN_DEFAULTS['observation_covariance']
                )
                result = KalmanFilterManager.create_result(state, w, t, 'test', True)
        
        if result:
            results.append(result)
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Extract data
    result_times = [r['timestamp'] for r in results]
    raw_weights = [r['raw_weight'] for r in results]
    filtered_weights = [r.get('filtered_weight', r['raw_weight']) for r in results]
    trends = [r.get('trend', 0) * 7 for r in results]  # Weekly trend
    in_warmup_flags = [r.get('in_warmup', False) for r in results]
    
    # Plot weights
    ax1.scatter(result_times, raw_weights, alpha=0.3, s=20, color='gray', label='Raw weights')
    
    # Separate warmup and normal filtered weights
    normal_times = [t for t, w in zip(result_times, in_warmup_flags) if not w]
    normal_filtered = [f for f, w in zip(filtered_weights, in_warmup_flags) if not w]
    warmup_times = [t for t, w in zip(result_times, in_warmup_flags) if w]
    warmup_weights = [f for f, w in zip(filtered_weights, in_warmup_flags) if w]
    
    if normal_times:
        ax1.plot(normal_times, normal_filtered, 'b-', linewidth=2, label='Filtered (improved)')
    if warmup_times:
        ax1.scatter(warmup_times, warmup_weights, color='orange', s=50, marker='s', label='Warmup phase')
    
    # Mark gap
    ax1.axvspan(gap_start, gap_end, alpha=0.2, color='yellow', label='18-day gap')
    
    # Add true trend line
    true_trend_line = [85.0 - 0.15 * i for i in range(len(timestamps))]
    ax1.plot(timestamps, true_trend_line, 'k--', alpha=0.3, label='True trend')
    
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Weight (kg)')
    ax1.set_title('Improved Kalman Filter with Warmup and Adaptive Parameters')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot trends
    ax2.plot(result_times, trends, 'b-', linewidth=2, label='Estimated trend (weekly)')
    ax2.axhline(y=-0.15*7, color='black', linestyle='--', alpha=0.5, label='True trend (-1.05 kg/week)')
    ax2.axvspan(gap_start, gap_end, alpha=0.2, color='yellow')
    
    # Mark warmup phase
    if warmup_times:
        ax2.axvspan(warmup_times[0], warmup_times[-1], alpha=0.2, color='orange', label='Warmup phase')
    
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Weekly Trend (kg/week)')
    ax2.set_title('Trend Estimation with Improved Gap Handling')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('docs/images/improved_gap_handling.png', dpi=150, bbox_inches='tight')
    print("\nSaved plot to docs/images/improved_gap_handling.png")
    
    # Calculate metrics
    post_gap_results = [r for r in results if r['timestamp'] > gap_end]
    if post_gap_results:
        post_gap_filtered = [r.get('filtered_weight', r['raw_weight']) for r in post_gap_results]
        post_gap_times = [r['timestamp'] for r in post_gap_results]
        
        # Expected weights
        days_from_start = [(t - datetime(2024, 1, 1)).days for t in post_gap_times]
        expected_weights = [85.0 - 0.15 * d for d in days_from_start]
        
        # RMSE
        rmse = np.sqrt(np.mean((np.array(post_gap_filtered) - np.array(expected_weights))**2))
        print(f"\nPost-gap RMSE: {rmse:.3f} kg")
        
        # Trend accuracy
        post_gap_trends = [r.get('trend', 0) for r in post_gap_results]
        trend_errors = [abs(t - (-0.15)) for t in post_gap_trends]
        avg_trend_error = np.mean(trend_errors)
        print(f"Average trend error: {avg_trend_error:.4f} kg/day ({avg_trend_error*7:.3f} kg/week)")
        
        # Convergence time
        for i, error in enumerate(trend_errors):
            if error < 0.03:  # Within 0.03 kg/day
                print(f"Trend converged after {i+1} measurements")
                break

if __name__ == "__main__":
    demonstrate_improvement()
