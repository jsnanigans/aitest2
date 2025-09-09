# Step 04: ✅ Process Noise Matrix Already Diagonal

## Priority: COMPLETED ✓

## Current State
**Good News!** The process noise matrix Q is already properly diagonal:
- Initial Q matrix is diagonal (lines 63-66)
- Dynamic Q matrix maintains diagonal structure (lines 188-191)
- Separate noise terms for weight and trend as specified
- Time-adaptive scaling is correctly applied

## Framework Compliance
The framework document (Section 4.3, Table "Kalman Filter State-Space Formulation") specifies Q should be:
```
Q = [[q1, 0],
     [0, q2]]
```

**Current Implementation Matches**:
- `q1` = `base_process_noise_weight` (weight uncertainty)
- `q2` = `base_process_noise_trend` (velocity/trend uncertainty)
- Diagonal structure maintained throughout
- Time-adaptive scaling preserves diagonal form

## Advanced Features Already Implemented
Beyond basic requirements, the implementation includes:
1. **Time-adaptive scaling** (lines 159-173)
2. **Source-based trust adjustment** (lines 176-186)
3. **Stability factor based on innovation history** (lines 144-157)
4. **Gap-aware process noise** (handles missing data)

## Implementation Guide

### Correct Process Noise Matrix Structure
```python
class ProcessNoiseModel:
    def __init__(self, weight_noise=0.01, velocity_noise=0.001):
        """
        Initialize diagonal process noise covariance matrix
        
        Args:
            weight_noise: Variance of weight process noise (kg²/day)
            velocity_noise: Variance of velocity process noise (kg²/day³)
        """
        self.q_weight = weight_noise
        self.q_velocity = velocity_noise
    
    def get_Q_matrix(self, dt=1.0):
        """
        Get process noise covariance matrix for given time step
        
        For 2D state [weight, velocity]:
        Q = [[q_weight,    0      ],
             [0,        q_velocity]]
        """
        Q = np.array([
            [self.q_weight * dt, 0],
            [0, self.q_velocity * dt]
        ])
        return Q
    
    def get_Q_discrete(self, dt=1.0):
        """
        Alternative: Discrete-time process noise (more theoretically correct)
        Accounts for coupling between position and velocity noise
        """
        # For constant velocity model with acceleration noise
        q = self.q_velocity  # Acceleration process noise
        
        Q = np.array([
            [q * dt**3 / 3, q * dt**2 / 2],
            [q * dt**2 / 2, q * dt]
        ])
        return Q
```

### Physical Interpretation of Parameters
```python
def calculate_process_noise_params(user_profile):
    """
    Calculate appropriate process noise based on user characteristics
    """
    # Base values (for average adult)
    base_weight_noise = 0.01    # 0.01 kg²/day (100g daily variation)
    base_velocity_noise = 0.0001 # 0.0001 kg²/day³ (slow drift in trend)
    
    # Adjust based on user factors
    factors = {
        'high_activity': 1.5,      # More variation for active users
        'medication': 1.3,         # Medications can cause fluctuations
        'female': 1.2,             # Hormonal cycles add variation
        'dieting': 2.0,            # Active weight loss increases variation
        'maintenance': 0.8,        # Stable phase has less variation
    }
    
    # Apply relevant factors
    weight_noise = base_weight_noise
    velocity_noise = base_velocity_noise
    
    for factor, multiplier in factors.items():
        if user_profile.get(factor, False):
            weight_noise *= multiplier
            velocity_noise *= multiplier
    
    return weight_noise, velocity_noise
```

### Adaptive Process Noise
```python
class AdaptiveProcessNoise:
    """
    Dynamically adjust Q based on innovation sequence
    """
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.innovation_history = []
        self.Q_history = []
    
    def update(self, innovation, current_Q):
        """
        Adapt Q based on innovation statistics
        """
        self.innovation_history.append(innovation)
        
        if len(self.innovation_history) > self.window_size:
            self.innovation_history.pop(0)
        
        if len(self.innovation_history) >= 10:
            # Calculate innovation variance
            innovation_var = np.var(self.innovation_history)
            
            # Expected innovation variance: S = H @ P @ H.T + R
            # If actual >> expected, increase Q
            # If actual << expected, decrease Q
            
            # Simple adaptation rule
            if innovation_var > 2 * expected_var:
                # Innovations larger than expected, increase Q
                new_Q = current_Q * 1.1
            elif innovation_var < 0.5 * expected_var:
                # Innovations smaller than expected, decrease Q
                new_Q = current_Q * 0.9
            else:
                new_Q = current_Q
            
            # Ensure Q remains positive definite
            new_Q = np.maximum(new_Q, 1e-6 * np.eye(2))
            
            return new_Q
        
        return current_Q
```

### Integration with Kalman Filter
```python
class ImprovedKalmanFilter:
    def __init__(self, user_profile=None):
        # State dimension
        self.n = 2  # [weight, velocity]
        
        # Initialize process noise model
        if user_profile:
            q_w, q_v = calculate_process_noise_params(user_profile)
        else:
            q_w, q_v = 0.01, 0.0001  # Default values
        
        self.process_noise_model = ProcessNoiseModel(q_w, q_v)
        
        # Other initialization...
        self.state = np.zeros(self.n)
        self.P = np.eye(self.n) * 100  # Initial high uncertainty
    
    def predict(self, dt=1.0):
        """
        Prediction step with correct Q matrix
        """
        # State transition
        F = np.array([[1, dt], [0, 1]])
        
        # Get diagonal process noise
        Q = self.process_noise_model.get_Q_matrix(dt)
        
        # Predict state
        self.state = F @ self.state
        
        # Predict covariance
        self.P = F @ self.P @ F.T + Q
        
        # Ensure P remains symmetric and positive definite
        self.P = (self.P + self.P.T) / 2
        self.P = np.maximum(self.P, 1e-10 * np.eye(self.n))
        
        return self.state, self.P
```

### Tuning Guidelines
```python
def tune_process_noise(validation_data):
    """
    Grid search for optimal Q parameters
    """
    weight_noise_range = np.logspace(-4, -1, 10)  # 0.0001 to 0.1
    velocity_noise_range = np.logspace(-6, -3, 10)  # 0.000001 to 0.001
    
    best_params = None
    best_score = float('inf')
    
    for q_w in weight_noise_range:
        for q_v in velocity_noise_range:
            # Create filter with these parameters
            kf = ImprovedKalmanFilter()
            kf.process_noise_model = ProcessNoiseModel(q_w, q_v)
            
            # Run on validation data
            mse = evaluate_filter(kf, validation_data)
            
            if mse < best_score:
                best_score = mse
                best_params = (q_w, q_v)
    
    return best_params
```

## Validation Criteria
- Q matrix must be positive semi-definite at all times
- Q matrix must remain diagonal (off-diagonal elements = 0)
- Process noise values should be physiologically plausible:
  - Weight noise: 0.001 - 0.1 kg²/day
  - Velocity noise: 0.00001 - 0.001 kg²/day³
- Filter should not diverge over 1000+ measurements

## Common Pitfalls to Avoid
1. **Non-diagonal Q**: Adding correlation terms without physical justification
2. **Negative values**: Process noise variances must be positive
3. **Unit confusion**: Ensure correct units (kg² not kg)
4. **Time scaling**: Remember to scale Q by time step (dt)

## Monitoring
```python
def monitor_process_noise_health(filter_state):
    """
    Check Q matrix health during operation
    """
    Q = filter_state.Q
    
    checks = {
        'is_diagonal': np.allclose(Q - np.diag(np.diag(Q)), 0),
        'is_positive': np.all(np.diag(Q) > 0),
        'is_symmetric': np.allclose(Q, Q.T),
        'eigenvalues_positive': np.all(np.linalg.eigvals(Q) > 0)
    }
    
    return all(checks.values()), checks
```

## References
- Framework Section 4.3: State-Space Model Components
- Kalman Filter Theory: Process Noise Covariance
- Bar-Shalom et al. (2001): Estimation with Applications to Tracking
- Physiological weight variation studies