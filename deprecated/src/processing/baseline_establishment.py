#!/usr/bin/env python3

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from datetime import datetime, timedelta
from src.core import get_logger

logger = get_logger(__name__)


class RobustBaselineEstimator:
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        self.min_readings = config.get('baseline_min_readings', 3)
        self.max_readings = config.get('baseline_max_readings', 30)
        self.window_days = config.get('baseline_window_days', 7)
        self.max_window_days = config.get('baseline_max_window_days', 14)
        self.iqr_multiplier = config.get('iqr_multiplier', 1.5)
        self.mad_to_std = 1.4826
        self.max_outlier_ratio = 0.5
        self.min_variance = 0.01
        self.reasonable_weight_range = (30.0, 300.0)
        self.typical_weight_range = (40.0, 200.0)
        
    def establish_baseline(self, readings: List[Dict[str, Any]], 
                          signup_date: Optional[datetime] = None,
                          window_start_date: Optional[datetime] = None) -> Dict[str, Any]:
        
        if not readings:
            return {
                'success': False,
                'error': 'No readings provided',
                'readings_count': 0
            }
        
        filtered_readings = self._filter_baseline_window(readings, signup_date, window_start_date)
        
        if len(filtered_readings) < self.min_readings:
            return {
                'success': False,
                'error': f'Insufficient readings: {len(filtered_readings)} < {self.min_readings}',
                'readings_count': len(filtered_readings),
                'original_count': len(readings)
            }
        
        weights = [r['weight'] for r in filtered_readings]
        
        if not self._validate_weight_range(weights):
            return {
                'success': False,
                'error': 'Weights outside reasonable range',
                'readings_count': len(weights)
            }
        
        cleaned_weights, outlier_info = self._iqr_outlier_removal(weights)
        
        if len(cleaned_weights) < self.min_readings:
            return {
                'success': False,
                'error': f'Too few readings after outlier removal: {len(cleaned_weights)}',
                'readings_count': len(weights),
                'outliers_removed': outlier_info['outliers_removed']
            }
        
        outlier_ratio = outlier_info['outliers_removed'] / len(weights) if weights else 0
        if outlier_ratio > self.max_outlier_ratio:
            logger.warning(f"High outlier ratio: {outlier_ratio:.1%}")
            confidence = 'low'
        elif len(cleaned_weights) >= 10:
            confidence = 'high'
        elif len(cleaned_weights) >= 5:
            confidence = 'medium'
        else:
            confidence = 'low'
        
        baseline_weight = self._calculate_baseline_weight(cleaned_weights)
        
        variance_info = self._calculate_variance(cleaned_weights, baseline_weight)
        
        baseline_dates = [r['date'] for r in filtered_readings 
                         if r['weight'] >= outlier_info['lower_fence'] 
                         and r['weight'] <= outlier_info['upper_fence']]
        
        result = {
            'success': True,
            'baseline_weight': round(baseline_weight, 2),
            'measurement_variance': round(variance_info['variance'], 4),
            'measurement_noise_std': round(variance_info['std'], 3),
            'mad': round(variance_info['mad'], 3),
            'readings_used': len(cleaned_weights),
            'original_count': len(readings),
            'window_count': len(filtered_readings),
            'outliers_removed': outlier_info['outliers_removed'],
            'outlier_ratio': round(outlier_ratio, 3),
            'confidence': confidence,
            'method': 'IQR→Median→MAD',
            'iqr_fences': {
                'lower': round(outlier_info['lower_fence'], 2),
                'upper': round(outlier_info['upper_fence'], 2),
                'q1': round(outlier_info['q1'], 2),
                'q3': round(outlier_info['q3'], 2),
                'iqr': round(outlier_info['iqr'], 2)
            }
        }
        
        if baseline_dates:
            result['baseline_start'] = min(baseline_dates)
            result['baseline_end'] = max(baseline_dates)
        
        if len(cleaned_weights) >= 5:
            result['percentiles'] = {
                'p5': round(np.percentile(cleaned_weights, 5), 2),
                'p25': round(np.percentile(cleaned_weights, 25), 2),
                'p50': round(baseline_weight, 2),
                'p75': round(np.percentile(cleaned_weights, 75), 2),
                'p95': round(np.percentile(cleaned_weights, 95), 2)
            }
        
        return result
    
    def _filter_baseline_window(self, readings: List[Dict[str, Any]], 
                                signup_date: Optional[datetime],
                                window_start_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        
        if window_start_date:
            start_date = window_start_date
        elif signup_date:
            start_date = signup_date
        else:
            sorted_readings = sorted(readings, key=lambda r: r.get('date', datetime.min))
            return sorted_readings[:min(len(sorted_readings), self.max_readings)]
        
        window_end = start_date + timedelta(days=self.window_days)
        max_window_end = start_date + timedelta(days=self.max_window_days)
        
        filtered = []
        for reading in readings:
            reading_date = reading.get('date')
            if isinstance(reading_date, str):
                try:
                    reading_date = datetime.fromisoformat(reading_date.replace('Z', '+00:00'))
                except:
                    continue
            
            if reading_date and start_date <= reading_date <= window_end:
                filtered.append(reading)
                if len(filtered) >= self.max_readings:
                    break
            elif reading_date and window_end < reading_date <= max_window_end and len(filtered) < self.min_readings:
                filtered.append(reading)
        
        return sorted(filtered, key=lambda r: r.get('date', datetime.min))[:self.max_readings]
    
    def _validate_weight_range(self, weights: List[float]) -> bool:
        
        min_weight, max_weight = self.reasonable_weight_range
        
        valid_weights = [w for w in weights if min_weight <= w <= max_weight]
        
        if len(valid_weights) < self.min_readings:
            logger.warning(f"Too few valid weights after range filtering: {len(valid_weights)}")
            return False
        
        return True
    
    def _iqr_outlier_removal(self, weights: List[float]) -> Tuple[List[float], Dict[str, Any]]:
        
        if len(weights) < 4:
            return weights, {
                'outliers_removed': 0,
                'lower_fence': min(weights),
                'upper_fence': max(weights),
                'q1': min(weights),
                'q3': max(weights),
                'iqr': max(weights) - min(weights)
            }
        
        q1 = np.percentile(weights, 25)
        q3 = np.percentile(weights, 75)
        iqr = q3 - q1
        
        lower_fence = q1 - self.iqr_multiplier * iqr
        upper_fence = q3 + self.iqr_multiplier * iqr
        
        lower_fence = max(lower_fence, self.reasonable_weight_range[0])
        upper_fence = min(upper_fence, self.reasonable_weight_range[1])
        
        cleaned = [w for w in weights if lower_fence <= w <= upper_fence]
        outliers_removed = len(weights) - len(cleaned)
        
        if outliers_removed > 0:
            logger.debug(f"Removed {outliers_removed} outliers using IQR method")
            logger.debug(f"Fences: [{lower_fence:.1f}, {upper_fence:.1f}]")
        
        return cleaned, {
            'outliers_removed': outliers_removed,
            'lower_fence': lower_fence,
            'upper_fence': upper_fence,
            'q1': q1,
            'q3': q3,
            'iqr': iqr
        }
    
    def _calculate_baseline_weight(self, weights: List[float]) -> float:
        
        return float(np.median(weights))
    
    def _calculate_variance(self, weights: List[float], baseline: float) -> Dict[str, Any]:
        
        deviations = np.abs(np.array(weights) - baseline)
        mad = float(np.median(deviations))
        
        if mad < 0.001:
            logger.debug("MAD near zero, using minimum variance")
            std_estimate = np.sqrt(self.min_variance)
        else:
            std_estimate = self.mad_to_std * mad
        
        variance = max(std_estimate ** 2, self.min_variance)
        
        sample_std = float(np.std(weights, ddof=1)) if len(weights) > 1 else std_estimate
        
        return {
            'variance': variance,
            'std': std_estimate,
            'mad': mad,
            'sample_std': sample_std,
            'robust_to_sample_ratio': std_estimate / sample_std if sample_std > 0 else 1.0
        }
    
    def validate_baseline_quality(self, baseline_result: Dict[str, Any]) -> Dict[str, Any]:
        
        if not baseline_result.get('success'):
            return {
                'valid': False,
                'reason': baseline_result.get('error', 'Baseline establishment failed'),
                'recommendations': ['Collect more readings', 'Check data quality']
            }
        
        issues = []
        recommendations = []
        
        if baseline_result['outlier_ratio'] > 0.3:
            issues.append('High outlier ratio')
            recommendations.append('Review data collection process')
        
        if baseline_result['readings_used'] < 5:
            issues.append('Few readings')
            recommendations.append('Collect more baseline readings')
        
        weight = baseline_result['baseline_weight']
        if not (self.typical_weight_range[0] <= weight <= self.typical_weight_range[1]):
            issues.append('Atypical weight range')
            recommendations.append('Verify measurement units and calibration')
        
        if baseline_result['measurement_noise_std'] > 2.0:
            issues.append('High measurement variability')
            recommendations.append('Use consistent measurement conditions')
        
        return {
            'valid': len(issues) == 0,
            'quality_score': max(0, 1.0 - len(issues) * 0.25),
            'issues': issues,
            'recommendations': recommendations,
            'confidence': baseline_result.get('confidence', 'unknown')
        }