# Step 07: Implement ARIMA Outlier Classification

## Priority: MEDIUM - NOT IMPLEMENTED

## Current State
**Missing Feature**: No ARIMA-based outlier classification exists:
- Current outlier detection is basic (normalized innovation only)
- Cannot classify outlier types (AO/IO/LS/TC)
- No time-series modeling beyond Kalman filter
- Missing the sophisticated classification from framework Section 3.2

## Why This Change?
The framework document (Section 3.2) highlights that ARIMA-based detection can classify outliers into four distinct types. This classification:

1. **Richer Information**: Understanding outlier type enables better handling
2. **Targeted Response**: Different outlier types require different actions
3. **Pattern Recognition**: Identifies systematic vs. random errors
4. **User Insights**: Can explain why measurements were flagged

## Expected Benefits
- **Better outlier handling** with type-specific responses
- **Improved accuracy** by distinguishing temporary vs. permanent changes
- **Enhanced diagnostics** for data quality issues
- **Smarter filtering** that adapts to outlier patterns
- **User education** about measurement best practices

## Implementation Guide

### ARIMA Model for Weight Time Series
```python
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from scipy import stats

class ARIMAOutlierDetector:
    """
    ARIMA-based outlier detection with classification
    """
    
    def __init__(self, order=(1, 1, 1), window_size=30):
        """
        Args:
            order: (p, d, q) ARIMA model order
                p: autoregressive order
                d: differencing order  
                q: moving average order
            window_size: Size of sliding window for model fitting
        """
        self.order = order
        self.window_size = window_size
        self.model = None
        self.outlier_history = []
    
    def fit_model(self, time_series):
        """
        Fit ARIMA model to time series window
        """
        try:
            # Fit ARIMA model
            model = ARIMA(time_series, order=self.order)
            self.model = model.fit(disp=False)
            
            # Get model parameters
            self.params = {
                'ar': self.model.arparams,
                'ma': self.model.maparams,
                'sigma': np.sqrt(self.model.sigma2)
            }
            
            return True
        except:
            # Fall back to simpler model if fitting fails
            self.order = (1, 0, 1)
            return False
    
    def classify_outlier(self, observation, index, time_series):
        """
        Classify outlier type based on ARIMA model
        
        Returns:
            (outlier_type, confidence, impact)
        """
        if self.model is None:
            return None, 0, 0
        
        # Get prediction and residual
        forecast = self.model.forecast(steps=1)[0]
        residual = observation - forecast
        
        # Calculate standardized residual
        std_residual = residual / self.params['sigma']
        
        # Classify based on residual patterns
        outlier_type = self.determine_outlier_type(
            std_residual, index, time_series
        )
        
        # Calculate confidence and impact
        confidence = min(abs(std_residual) / 3.0, 1.0)
        impact = self.calculate_impact(outlier_type, std_residual)
        
        return outlier_type, confidence, impact
```

### Four Types of Outliers
```python
class OutlierTypes:
    """
    Four distinct outlier types from ARIMA framework
    """
    
    @staticmethod
    def detect_additive_outlier(residuals, index):
        """
        AO: Single-point anomaly that affects only one observation
        Example: Weighing with heavy clothes once
        """
        if index == 0 or index >= len(residuals) - 1:
            return False, 0
        
        # Large residual at index, normal before and after
        current_large = abs(residuals[index]) > 3
        before_normal = abs(residuals[index - 1]) < 2
        after_normal = abs(residuals[index + 1]) < 2
        
        is_ao = current_large and before_normal and after_normal
        confidence = abs(residuals[index]) / 5.0 if is_ao else 0
        
        return is_ao, confidence
    
    @staticmethod
    def detect_innovational_outlier(residuals, index, ar_params):
        """
        IO: Shock that propagates through AR structure
        Example: Scale calibration error affecting multiple readings
        """
        if index >= len(residuals) - 3:
            return False, 0
        
        # Check if residuals follow AR pattern after shock
        shock = residuals[index]
        expected_propagation = []
        
        for i in range(1, min(4, len(residuals) - index)):
            expected = shock * (ar_params[0] ** i) if len(ar_params) > 0 else 0
            expected_propagation.append(expected)
        
        actual_propagation = residuals[index + 1:index + len(expected_propagation) + 1]
        
        # Calculate correlation between expected and actual
        if len(actual_propagation) == len(expected_propagation):
            correlation = np.corrcoef(expected_propagation, actual_propagation)[0, 1]
            is_io = correlation > 0.7 and abs(shock) > 3
            confidence = correlation if is_io else 0
        else:
            is_io = False
            confidence = 0
        
        return is_io, confidence
    
    @staticmethod  
    def detect_level_shift(residuals, index, window=7):
        """
        LS: Sudden permanent change in mean level
        Example: Starting medication or changing scales
        """
        if index < window or index > len(residuals) - window:
            return False, 0
        
        # Compare means before and after
        before = residuals[index - window:index]
        after = residuals[index:index + window]
        
        mean_before = np.mean(before)
        mean_after = np.mean(after)
        
        # T-test for significant difference
        t_stat, p_value = stats.ttest_ind(before, after)
        
        is_ls = p_value < 0.01 and abs(mean_after - mean_before) > 2
        confidence = 1 - p_value if is_ls else 0
        
        return is_ls, confidence
    
    @staticmethod
    def detect_temporary_change(residuals, index, decay_window=5):
        """
        TC: Initial shock that decays back to normal
        Example: Temporary water retention from high sodium meal
        """
        if index >= len(residuals) - decay_window:
            return False, 0
        
        # Check for exponential decay pattern
        shock = residuals[index]
        if abs(shock) < 3:
            return False, 0
        
        # Examine decay pattern
        decay_observed = []
        for i in range(1, min(decay_window, len(residuals) - index)):
            decay_observed.append(residuals[index + i])
        
        # Check if magnitude decreases over time
        is_decaying = all(
            abs(decay_observed[i]) < abs(decay_observed[i-1]) 
            for i in range(1, len(decay_observed))
        )
        
        # Check if returns to normal
        returns_to_normal = abs(decay_observed[-1]) < 1 if decay_observed else False
        
        is_tc = is_decaying and returns_to_normal
        confidence = 1 - (abs(decay_observed[-1]) / abs(shock)) if is_tc else 0
        
        return is_tc, confidence
```

### Complete Classification Pipeline
```python
class OutlierClassificationPipeline:
    """
    Complete pipeline for detecting and classifying outliers
    """
    
    def __init__(self):
        self.arima_detector = ARIMAOutlierDetector()
        self.classification_stats = {
            'AO': 0, 'IO': 0, 'LS': 0, 'TC': 0, 'normal': 0
        }
    
    def process_time_series(self, weights):
        """
        Process entire time series with outlier classification
        """
        results = []
        
        # Sliding window processing
        for i in range(self.arima_detector.window_size, len(weights)):
            window = weights[i - self.arima_detector.window_size:i]
            
            # Fit ARIMA model to window
            self.arima_detector.fit_model(window)
            
            # Classify current observation
            classification = self.classify_observation(
                weights[i], i, weights
            )
            
            results.append({
                'index': i,
                'weight': weights[i],
                'classification': classification
            })
            
            # Update statistics
            self.classification_stats[classification['type']] += 1
        
        return results
    
    def classify_observation(self, observation, index, full_series):
        """
        Classify single observation
        """
        # Get model residuals
        if self.arima_detector.model:
            residuals = self.arima_detector.model.resid
        else:
            return {'type': 'normal', 'confidence': 0}
        
        # Standardize residuals
        std_residuals = residuals / np.std(residuals)
        
        # Test for each outlier type
        tests = [
            ('AO', OutlierTypes.detect_additive_outlier(std_residuals, -1)),
            ('IO', OutlierTypes.detect_innovational_outlier(
                std_residuals, -1, self.arima_detector.params.get('ar', []))),
            ('LS', OutlierTypes.detect_level_shift(std_residuals, -1)),
            ('TC', OutlierTypes.detect_temporary_change(std_residuals, -1))
        ]
        
        # Find most likely classification
        best_type = 'normal'
        best_confidence = 0
        
        for outlier_type, (is_outlier, confidence) in tests:
            if is_outlier and confidence > best_confidence:
                best_type = outlier_type
                best_confidence = confidence
        
        return {
            'type': best_type,
            'confidence': best_confidence,
            'action': self.get_recommended_action(best_type)
        }
    
    def get_recommended_action(self, outlier_type):
        """
        Get recommended action for each outlier type
        """
        actions = {
            'AO': 'reject_single',      # Reject this measurement only
            'IO': 'investigate_device',  # Check device calibration
            'LS': 'update_baseline',    # Update baseline, possible regime change
            'TC': 'wait_and_monitor',   # Monitor, will self-correct
            'normal': 'accept'           # Accept measurement
        }
        return actions.get(outlier_type, 'review')
```

### Type-Specific Handling
```python
class OutlierHandler:
    """
    Handle different outlier types appropriately
    """
    
    def handle_outlier(self, outlier_info, kalman_filter):
        """
        Take appropriate action based on outlier type
        """
        outlier_type = outlier_info['type']
        confidence = outlier_info['confidence']
        
        if outlier_type == 'AO':
            # Additive outlier: Simply reject
            return self.handle_additive_outlier(outlier_info)
        
        elif outlier_type == 'IO':
            # Innovational: May need to reset filter
            return self.handle_innovational_outlier(
                outlier_info, kalman_filter, confidence)
        
        elif outlier_type == 'LS':
            # Level shift: Trigger change point detection
            return self.handle_level_shift(
                outlier_info, kalman_filter)
        
        elif outlier_type == 'TC':
            # Temporary change: Reduce influence
            return self.handle_temporary_change(
                outlier_info, kalman_filter)
        
        else:
            # Normal observation
            return {'action': 'accept', 'modified': False}
    
    def handle_additive_outlier(self, outlier_info):
        """
        Handle single-point anomaly
        """
        return {
            'action': 'reject',
            'reason': 'Single measurement error detected',
            'suggestion': 'Please re-weigh yourself',
            'modified': False
        }
    
    def handle_innovational_outlier(self, outlier_info, kf, confidence):
        """
        Handle propagating error
        """
        if confidence > 0.8:
            # High confidence: Reset filter
            kf.P *= 2.0  # Increase uncertainty
            return {
                'action': 'accept_with_reset',
                'reason': 'Device calibration issue detected',
                'suggestion': 'Check scale calibration',
                'modified': True
            }
        else:
            # Low confidence: Just flag
            return {
                'action': 'accept_with_flag',
                'reason': 'Unusual pattern detected',
                'modified': False
            }
    
    def handle_level_shift(self, outlier_info, kf):
        """
        Handle permanent change
        """
        # Trigger regime change detection
        return {
            'action': 'trigger_cpd',
            'reason': 'Significant weight change detected',
            'suggestion': 'New baseline may be establishing',
            'modified': True,
            'new_regime': True
        }
    
    def handle_temporary_change(self, outlier_info, kf):
        """
        Handle temporary fluctuation
        """
        # Reduce measurement influence
        kf.R *= 2.0  # Increase measurement noise temporarily
        return {
            'action': 'accept_with_reduced_weight',
            'reason': 'Temporary fluctuation detected',
            'suggestion': 'Weight should normalize in 2-3 days',
            'modified': True
        }
```

### Performance Metrics
```python
def evaluate_classification_accuracy(classified_outliers, ground_truth):
    """
    Evaluate classification performance
    """
    from sklearn.metrics import classification_report, confusion_matrix
    
    y_true = [gt['type'] for gt in ground_truth]
    y_pred = [co['type'] for co in classified_outliers]
    
    # Classification report
    report = classification_report(
        y_true, y_pred, 
        labels=['AO', 'IO', 'LS', 'TC', 'normal']
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Type-specific metrics
    type_metrics = {}
    for outlier_type in ['AO', 'IO', 'LS', 'TC']:
        type_true = [1 if t == outlier_type else 0 for t in y_true]
        type_pred = [1 if t == outlier_type else 0 for t in y_pred]
        
        tp = sum(1 for i in range(len(type_true)) 
                if type_true[i] == 1 and type_pred[i] == 1)
        fp = sum(1 for i in range(len(type_true)) 
                if type_true[i] == 0 and type_pred[i] == 1)
        fn = sum(1 for i in range(len(type_true)) 
                if type_true[i] == 1 and type_pred[i] == 0)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        type_metrics[outlier_type] = {
            'precision': precision,
            'recall': recall,
            'f1': 2 * precision * recall / (precision + recall) 
                  if (precision + recall) > 0 else 0
        }
    
    return {
        'classification_report': report,
        'confusion_matrix': cm,
        'type_metrics': type_metrics
    }
```

## Validation Criteria
- Classification accuracy > 80% on labeled test data
- Type-specific precision > 70% for each outlier type
- Processing time < 50ms per observation
- Model should converge within 30 observations

## Edge Cases
1. **Insufficient History**: Need minimum 30 points for reliable ARIMA
2. **Model Convergence**: Fall back to simpler models if ARIMA fails
3. **Multiple Types**: Observation may match multiple types - use highest confidence
4. **Seasonal Patterns**: May be confused with level shifts

## References
- Framework Section 3.2: "ARIMA-Based Outlier Detection"
- Tsay (1988): "Outliers, Level Shifts, and Variance Changes"
- Chen & Liu (1993): "Joint Estimation of Model Parameters and Outlier Effects"
- Fox (1972): "Outliers in Time Series"