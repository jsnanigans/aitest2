# Step 05: Implement Kalman Smoother for Historical Data

## Priority: MEDIUM - NOT IMPLEMENTED

## Current State
**Missing Feature**: The implementation has NO Kalman smoother functionality:
- Only forward filtering exists (real-time processing)
- No RTS (Rauch-Tung-Striebel) smoother implementation
- No backward pass for historical optimization
- Framework Section 6.1 specifies this as essential for batch processing

## Why This Change?
The framework document (Section 6.1) recommends using a Kalman Smoother for batch processing historical data. The smoother:

1. **Bidirectional Processing**: Uses both past AND future data for each estimate
2. **Optimal Estimation**: Provides the mathematically best possible state estimates
3. **Historical Accuracy**: Critical for establishing clean baselines and trend analysis
4. **Retrospective Analysis**: Enables better understanding of past weight changes

## Expected Benefits
- **30-50% improvement** in historical weight trajectory accuracy
- **Better baseline establishment** using all available information
- **Improved trend detection** for historical pattern analysis
- **Clean visualizations** of past weight history
- **Enhanced change point detection** with smoother trajectories

## Implementation Guide

### Rauch-Tung-Striebel (RTS) Smoother
```python
class KalmanSmoother:
    """
    Implements the Rauch-Tung-Striebel smoother for optimal state estimation
    using all available data (past and future)
    """
    
    def __init__(self, kalman_filter):
        """
        Initialize with a configured Kalman filter
        """
        self.kf = kalman_filter
        
    def smooth(self, measurements):
        """
        Perform forward-backward smoothing on entire time series
        
        Args:
            measurements: List of weight measurements
        
        Returns:
            smoothed_states: Optimal state estimates
            smoothed_covariances: Uncertainty estimates
        """
        n = len(measurements)
        
        # Forward pass: Standard Kalman filtering
        filtered_states = []
        filtered_covariances = []
        predicted_states = []
        predicted_covariances = []
        
        for measurement in measurements:
            # Predict
            x_pred, P_pred = self.kf.predict()
            predicted_states.append(x_pred)
            predicted_covariances.append(P_pred)
            
            # Update
            x_filt, P_filt = self.kf.update(measurement)
            filtered_states.append(x_filt)
            filtered_covariances.append(P_filt)
        
        # Backward pass: Smoothing
        smoothed_states = [None] * n
        smoothed_covariances = [None] * n
        
        # Initialize with last filtered values
        smoothed_states[-1] = filtered_states[-1]
        smoothed_covariances[-1] = filtered_covariances[-1]
        
        # Backward recursion
        for k in range(n - 2, -1, -1):
            # Smoother gain
            C_k = filtered_covariances[k] @ self.kf.F.T @ np.linalg.inv(predicted_covariances[k + 1])
            
            # Smoothed state
            smoothed_states[k] = (
                filtered_states[k] + 
                C_k @ (smoothed_states[k + 1] - predicted_states[k + 1])
            )
            
            # Smoothed covariance
            smoothed_covariances[k] = (
                filtered_covariances[k] + 
                C_k @ (smoothed_covariances[k + 1] - predicted_covariances[k + 1]) @ C_k.T
            )
        
        return smoothed_states, smoothed_covariances
```

### Fixed-Interval Smoothing
```python
class FixedIntervalSmoother:
    """
    Smoother for batch processing with known start and end points
    """
    
    def smooth_interval(self, measurements, start_idx, end_idx):
        """
        Smooth a specific interval of data
        Useful for reprocessing after detecting anomalies
        """
        interval_data = measurements[start_idx:end_idx+1]
        
        # Apply smoothing to interval
        smoothed_states, smoothed_covs = self.smooth(interval_data)
        
        # Calculate confidence bands
        confidence_bands = self.calculate_confidence_bands(
            smoothed_states, 
            smoothed_covs
        )
        
        return smoothed_states, confidence_bands
    
    def calculate_confidence_bands(self, states, covariances, confidence=0.95):
        """
        Calculate confidence intervals for smoothed estimates
        """
        from scipy.stats import norm
        z_score = norm.ppf((1 + confidence) / 2)
        
        bands = []
        for state, cov in zip(states, covariances):
            weight = state[0]
            weight_std = np.sqrt(cov[0, 0])
            
            lower = weight - z_score * weight_std
            upper = weight + z_score * weight_std
            
            bands.append({
                'weight': weight,
                'lower': lower,
                'upper': upper,
                'uncertainty': weight_std
            })
        
        return bands
```

### Online Fixed-Lag Smoothing
```python
class FixedLagSmoother:
    """
    Real-time smoothing with fixed delay
    Balances optimality with timeliness
    """
    
    def __init__(self, lag=7):
        """
        lag: Number of future measurements to wait before smoothing
        """
        self.lag = lag
        self.buffer = []
        self.smoothed_history = []
    
    def add_measurement(self, measurement):
        """
        Add new measurement and smooth if buffer is full
        """
        self.buffer.append(measurement)
        
        if len(self.buffer) > self.lag:
            # Smooth the oldest point using lag future points
            smoothed = self.smooth_with_lag(self.buffer)
            self.smoothed_history.append(smoothed)
            self.buffer.pop(0)
        
        return self.get_current_estimate()
    
    def smooth_with_lag(self, buffer):
        """
        Smooth first point using all points in buffer
        """
        # Apply RTS smoother to buffer
        smoother = KalmanSmoother()
        smoothed_states, _ = smoother.smooth(buffer)
        
        # Return smoothed estimate for first point
        return smoothed_states[0]
```

### Historical Data Reprocessing
```python
class HistoricalProcessor:
    """
    Complete pipeline for processing historical data
    """
    
    def process_user_history(self, user_id, raw_data):
        """
        Full historical processing with smoothing
        """
        # Step 1: Clean data with outlier detection
        cleaned_data = self.apply_outlier_detection(raw_data)
        
        # Step 2: Initialize Kalman filter with robust baseline
        baseline_params = self.calculate_robust_baseline(cleaned_data[:14])
        kf = self.initialize_kalman_filter(baseline_params)
        
        # Step 3: Apply forward-backward smoothing
        smoother = KalmanSmoother(kf)
        smoothed_states, smoothed_covs = smoother.smooth(cleaned_data)
        
        # Step 4: Extract weight trajectory
        weight_trajectory = [state[0] for state in smoothed_states]
        velocity_trajectory = [state[1] for state in smoothed_states]
        
        # Step 5: Calculate quality metrics
        metrics = self.calculate_trajectory_metrics(
            raw_data, 
            cleaned_data, 
            weight_trajectory
        )
        
        return {
            'user_id': user_id,
            'raw_measurements': raw_data,
            'cleaned_measurements': cleaned_data,
            'smoothed_weights': weight_trajectory,
            'weight_velocities': velocity_trajectory,
            'confidence_bands': confidence_bands,
            'metrics': metrics
        }
```

### Comparison: Filter vs Smoother
```python
def compare_filter_smoother(measurements):
    """
    Demonstrate improvement from smoothing
    """
    kf = KalmanFilter()
    
    # Forward filtering only
    filtered_weights = []
    for m in measurements:
        kf.predict()
        kf.update(m)
        filtered_weights.append(kf.state[0])
    
    # Reset and apply smoothing
    kf.reset()
    smoother = KalmanSmoother(kf)
    smoothed_states, _ = smoother.smooth(measurements)
    smoothed_weights = [s[0] for s in smoothed_states]
    
    # Calculate improvement metrics
    if ground_truth_available:
        filter_mse = np.mean((filtered_weights - ground_truth)**2)
        smoother_mse = np.mean((smoothed_weights - ground_truth)**2)
        improvement = (filter_mse - smoother_mse) / filter_mse * 100
        
        print(f"Filter MSE: {filter_mse:.4f}")
        print(f"Smoother MSE: {smoother_mse:.4f}")
        print(f"Improvement: {improvement:.1f}%")
    
    return filtered_weights, smoothed_weights
```

### Integration Points

1. **User Onboarding**: Apply smoother to all historical data
2. **Nightly Batch**: Re-smooth recent data for better estimates
3. **Data Export**: Provide smoothed trajectories for analysis
4. **Visualization**: Show both filtered and smoothed estimates

## Validation Criteria
- Smoothed estimates should have lower variance than filtered
- Smoothed trajectory should be physically plausible (no jumps)
- Computational time < 100ms per 365 measurements
- Memory usage should scale linearly with data size

## Performance Optimization
```python
class OptimizedSmoother:
    """
    Performance-optimized implementation
    """
    
    def smooth_large_dataset(self, measurements, chunk_size=1000):
        """
        Process large datasets in chunks with overlap
        """
        n = len(measurements)
        smoothed = []
        overlap = 50  # Overlap between chunks
        
        for i in range(0, n, chunk_size - overlap):
            chunk_end = min(i + chunk_size, n)
            chunk = measurements[i:chunk_end]
            
            # Smooth chunk
            chunk_smoothed, _ = self.smooth(chunk)
            
            # Add to results (handle overlap)
            if i == 0:
                smoothed.extend(chunk_smoothed)
            else:
                # Blend overlap region
                smoothed.extend(chunk_smoothed[overlap:])
        
        return smoothed
```

## Edge Cases
1. **Missing Data**: Interpolate or skip gaps in forward pass
2. **Single Measurement**: Return filtered estimate (no smoothing possible)
3. **Numerical Instability**: Use SVD for matrix inversion
4. **Real-time Constraint**: Use fixed-lag smoother for online applications

## References
- Framework Section 6.1: "Optimal Trajectory Estimation"
- Rauch, Tung, and Striebel (1965): Maximum likelihood estimates
- Shumway & Stoffer (2017): Time Series Analysis and Its Applications
- Fixed-lag smoothing for real-time applications