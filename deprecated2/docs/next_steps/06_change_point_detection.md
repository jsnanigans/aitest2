# Step 06: Add Change Point Detection (RESPERM)

## Priority: MEDIUM - NOT IMPLEMENTED

## Current State
**Missing Feature**: No change point detection algorithm exists:
- Kalman filter has adaptive noise (lines 140-173) but no CPD
- No RESPERM implementation as recommended in framework
- No regime detection or classification
- Filter adapts slowly to major lifestyle changes
- Framework Section 4.4 specifies CPD as essential for regime shifts

## Why This Change?
The framework document (Section 4.4) emphasizes that human weight trajectories have distinct regimes or phases. Change Point Detection (CPD):

1. **Rapid Adaptation**: Quickly identifies when weight dynamics fundamentally change
2. **Regime Recognition**: Distinguishes between maintenance, loss, and gain phases  
3. **Filter Reset**: Triggers appropriate Kalman filter adjustments
4. **Clinical Relevance**: Aligns with medical understanding of weight change phases

## Expected Benefits
- **50-70% faster adaptation** to new weight regimes
- **Better detection** of intervention effectiveness (diet/medication)
- **Reduced lag** in tracking genuine weight changes
- **Improved user insights** about when/why weight patterns changed
- **Automated regime classification** for personalized feedback

## Implementation Guide

### RESPERM Algorithm Implementation
```python
import numpy as np
from scipy import stats

class RESPERM:
    """
    Residuals Permutation-Based Method for change point detection
    Optimized for noisy time series like weight data
    """
    
    def __init__(self, window_size=30, n_permutations=1000, alpha=0.05):
        """
        Args:
            window_size: Size of sliding window for detection
            n_permutations: Number of permutations for significance testing
            alpha: Significance level for change point detection
        """
        self.window_size = window_size
        self.n_permutations = n_permutations
        self.alpha = alpha
        self.change_points = []
        
    def detect_change_points(self, time_series):
        """
        Detect all change points in time series
        
        Args:
            time_series: Array of weight measurements or Kalman innovations
        
        Returns:
            List of (index, p_value, change_type)
        """
        n = len(time_series)
        change_points = []
        
        for i in range(self.window_size, n - self.window_size):
            # Test for change point at position i
            p_value, change_type = self.test_change_point(
                time_series, i
            )
            
            if p_value < self.alpha:
                change_points.append({
                    'index': i,
                    'p_value': p_value,
                    'type': change_type,
                    'confidence': 1 - p_value
                })
        
        # Merge nearby change points
        change_points = self.merge_nearby_changes(change_points)
        
        return change_points
    
    def test_change_point(self, series, split_point):
        """
        Test if there's a significant change at split_point
        """
        # Split series
        before = series[max(0, split_point - self.window_size):split_point]
        after = series[split_point:min(len(series), split_point + self.window_size)]
        
        # Calculate test statistic (difference in means)
        observed_stat = np.mean(after) - np.mean(before)
        
        # Permutation test
        combined = np.concatenate([before, after])
        permuted_stats = []
        
        for _ in range(self.n_permutations):
            np.random.shuffle(combined)
            perm_before = combined[:len(before)]
            perm_after = combined[len(before):]
            perm_stat = np.mean(perm_after) - np.mean(perm_before)
            permuted_stats.append(perm_stat)
        
        # Calculate p-value
        p_value = np.mean(np.abs(permuted_stats) >= np.abs(observed_stat))
        
        # Determine change type
        if observed_stat > 0:
            change_type = 'increase'  # Weight gaining phase
        else:
            change_type = 'decrease'  # Weight losing phase
        
        return p_value, change_type
    
    def merge_nearby_changes(self, change_points, min_distance=7):
        """
        Merge change points that are too close together
        """
        if not change_points:
            return []
        
        merged = [change_points[0]]
        
        for cp in change_points[1:]:
            if cp['index'] - merged[-1]['index'] < min_distance:
                # Keep the more significant one
                if cp['p_value'] < merged[-1]['p_value']:
                    merged[-1] = cp
            else:
                merged.append(cp)
        
        return merged
```

### Bayesian Change Point Detection (Alternative)
```python
class BayesianChangePointDetection:
    """
    Bayesian online changepoint detection
    Better for real-time applications
    """
    
    def __init__(self, hazard_rate=0.01):
        """
        hazard_rate: Prior probability of change at any time point
        """
        self.hazard_rate = hazard_rate
        self.run_length_probs = []
        
    def update(self, new_observation):
        """
        Update change point probabilities with new observation
        """
        # Calculate predictive probability
        pred_probs = self.calculate_predictive_probs(new_observation)
        
        # Update run length distribution
        self.update_run_lengths(pred_probs)
        
        # Calculate change point probability
        change_prob = self.run_length_probs[0]
        
        return change_prob
    
    def calculate_predictive_probs(self, obs):
        """
        Calculate probability of observation under different run lengths
        """
        # Simplified: Assume Gaussian with different means for each regime
        probs = []
        for run_length in range(len(self.run_length_probs)):
            # Calculate likelihood under this run length
            likelihood = stats.norm.pdf(obs, loc=self.regime_mean[run_length])
            probs.append(likelihood)
        return probs
```

### Integration with Kalman Filter
```python
class AdaptiveKalmanWithCPD:
    """
    Kalman filter that adapts to detected change points
    """
    
    def __init__(self):
        self.kf = KalmanFilter()
        self.cpd = RESPERM()
        self.regime_history = []
        self.current_regime = 'maintenance'
        
    def process_with_change_detection(self, measurements):
        """
        Process measurements with automatic regime detection
        """
        # First pass: Standard Kalman filtering
        innovations = []
        for measurement in measurements:
            pred_state, _ = self.kf.predict()
            innovation = measurement - self.kf.H @ pred_state
            innovations.append(innovation)
            self.kf.update(measurement)
        
        # Detect change points in innovation sequence
        change_points = self.cpd.detect_change_points(innovations)
        
        # Second pass: Reprocess with regime changes
        self.kf.reset()
        results = []
        current_cp_idx = 0
        
        for i, measurement in enumerate(measurements):
            # Check if we're at a change point
            if (current_cp_idx < len(change_points) and 
                i == change_points[current_cp_idx]['index']):
                
                # Adapt filter for new regime
                self.adapt_to_regime_change(
                    change_points[current_cp_idx]
                )
                current_cp_idx += 1
            
            # Process measurement with adapted filter
            self.kf.predict()
            self.kf.update(measurement)
            
            results.append({
                'index': i,
                'weight': self.kf.state[0],
                'regime': self.current_regime,
                'uncertainty': np.sqrt(self.kf.P[0, 0])
            })
        
        return results, change_points
    
    def adapt_to_regime_change(self, change_point):
        """
        Adjust Kalman filter parameters for new regime
        """
        change_type = change_point['type']
        confidence = change_point['confidence']
        
        if change_type == 'decrease':
            # Weight loss regime detected
            self.current_regime = 'weight_loss'
            
            # Increase process noise (more variation expected)
            self.kf.Q *= 2.0
            
            # Reset velocity estimate
            self.kf.state[1] = -0.1  # Expect negative velocity
            
            # Increase state uncertainty
            self.kf.P *= (1 + confidence)
            
        elif change_type == 'increase':
            # Weight gain regime detected  
            self.current_regime = 'weight_gain'
            
            # Moderate process noise increase
            self.kf.Q *= 1.5
            
            # Reset velocity estimate
            self.kf.state[1] = 0.05  # Expect positive velocity
            
            # Increase state uncertainty
            self.kf.P *= (1 + confidence * 0.5)
            
        # Log regime change
        self.regime_history.append({
            'timestamp': change_point['index'],
            'from_regime': self.regime_history[-1] if self.regime_history else 'unknown',
            'to_regime': self.current_regime,
            'confidence': confidence
        })
```

### Regime-Specific Models
```python
class RegimeModels:
    """
    Different Kalman parameters for different weight regimes
    """
    
    REGIMES = {
        'maintenance': {
            'q_weight': 0.01,
            'q_velocity': 0.0001,
            'expected_velocity': 0.0,
            'velocity_variance': 0.01
        },
        'weight_loss': {
            'q_weight': 0.02,  # More variation during loss
            'q_velocity': 0.0005,
            'expected_velocity': -0.15,  # ~1 kg/week loss
            'velocity_variance': 0.05
        },
        'weight_gain': {
            'q_weight': 0.015,
            'q_velocity': 0.0003,
            'expected_velocity': 0.1,
            'velocity_variance': 0.03
        },
        'rapid_loss': {
            'q_weight': 0.03,  # High variation
            'q_velocity': 0.001,
            'expected_velocity': -0.3,  # ~2 kg/week
            'velocity_variance': 0.1
        }
    }
    
    @staticmethod
    def get_regime_params(regime_name):
        """
        Get Kalman parameters for specific regime
        """
        return RegimeModels.REGIMES.get(
            regime_name, 
            RegimeModels.REGIMES['maintenance']
        )
    
    @staticmethod
    def classify_regime(velocity, acceleration=None):
        """
        Classify current regime based on weight dynamics
        """
        if abs(velocity) < 0.05:  # Less than 50g/day
            return 'maintenance'
        elif velocity < -0.2:  # More than 200g/day loss
            return 'rapid_loss'
        elif velocity < -0.05:
            return 'weight_loss'
        else:
            return 'weight_gain'
```

### Visualization of Change Points
```python
def visualize_change_points(weights, change_points, regimes):
    """
    Create visualization showing detected change points and regimes
    """
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot weight trajectory with change points
    ax1.plot(weights, 'b-', label='Weight')
    
    for cp in change_points:
        idx = cp['index']
        ax1.axvline(x=idx, color='r', linestyle='--', alpha=0.7)
        ax1.text(idx, weights[idx], f"CP\n{cp['confidence']:.2f}", 
                ha='center', fontsize=8)
    
    # Color code by regime
    regime_colors = {
        'maintenance': 'green',
        'weight_loss': 'blue',
        'weight_gain': 'red',
        'rapid_loss': 'purple'
    }
    
    for i, regime in enumerate(regimes):
        ax1.axvspan(i, i+1, alpha=0.1, 
                   color=regime_colors.get(regime, 'gray'))
    
    ax1.set_ylabel('Weight (kg)')
    ax1.legend()
    ax1.set_title('Weight Trajectory with Change Points')
    
    # Plot regime probabilities over time
    ax2.set_ylabel('Regime')
    ax2.set_xlabel('Time')
    ax2.set_title('Detected Weight Regimes')
    
    return fig
```

## Validation Criteria
- Should detect known intervention starts (diet, medication) within 7 days
- False positive rate < 5% (detecting changes when none exist)
- Computational time < 10ms per detection for real-time use
- Should distinguish between noise and genuine regime changes

## Edge Cases
1. **Startup Period**: Don't detect changes in first 14 days
2. **Multiple Simultaneous Changes**: Prioritize most significant
3. **Gradual Changes**: May need trend-based detection
4. **Cyclic Patterns**: Distinguish from true change points

## Clinical Validation
```python
def validate_against_clinical_events(detected_changes, clinical_events):
    """
    Compare detected changes with known clinical interventions
    """
    matches = []
    for clinical_event in clinical_events:
        # Find nearest detected change
        nearest = min(detected_changes, 
                     key=lambda x: abs(x['index'] - clinical_event['day']))
        
        if abs(nearest['index'] - clinical_event['day']) <= 7:
            matches.append({
                'clinical': clinical_event,
                'detected': nearest,
                'lag': nearest['index'] - clinical_event['day']
            })
    
    detection_rate = len(matches) / len(clinical_events)
    avg_lag = np.mean([m['lag'] for m in matches])
    
    return {
        'detection_rate': detection_rate,
        'average_lag_days': avg_lag,
        'matches': matches
    }
```

## References
- Framework Section 4.4: "Change Point Detection"
- RESPERM: Robbins et al. (2011) - Residual-based CPD for time series
- Adams & MacKay (2007): Bayesian Online Changepoint Detection
- Clinical weight loss patterns and intervention studies