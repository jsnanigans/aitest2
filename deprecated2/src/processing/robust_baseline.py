"""
Robust Initial Weight Baseline Establishment
Framework Part II - Sections 2.2 and 2.3
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging

from src.core.types import WeightMeasurement, BaselineResult

logger = logging.getLogger(__name__)


class RobustBaselineEstimator:
    """
    Implements framework-specified robust baseline calculation.
    Protocol from Section 2.3: IQR outlier removal → Median → MAD
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        
        # Framework recommends 7-14 days
        self.collection_days = config.get('collection_days', 7)
        self.max_collection_days = config.get('max_collection_days', 14)
        
        # Minimum readings for reliable baseline
        self.min_readings = config.get('min_readings', 3)
        self.max_readings = config.get('max_readings', 30)
        
        # IQR outlier detection parameters
        self.iqr_multiplier = config.get('iqr_multiplier', 1.5)
        
        # MAD to standard deviation conversion (framework: k ≈ 1.4826)
        self.mad_scale = 1.4826
        
        # Trimming parameters for alternative methods
        self.trim_percent = config.get('trim_percent', 10)
        
    def establish_baseline(self, 
                          measurements: List[WeightMeasurement],
                          start_date: Optional[datetime] = None) -> BaselineResult:
        """
        Establish robust baseline following framework protocol.
        
        Args:
            measurements: List of weight measurements
            start_date: Optional start date for baseline window
            
        Returns:
            BaselineResult with baseline weight and variance estimates
        """
        if not measurements:
            return BaselineResult(
                success=False,
                error="No measurements provided"
            )
            
        # Step 1: Filter to baseline window
        filtered = self._filter_baseline_window(measurements, start_date)
        
        if len(filtered) < self.min_readings:
            return BaselineResult(
                success=False,
                error=f"Insufficient readings: {len(filtered)} < {self.min_readings}",
                metadata={'readings_available': len(filtered)}
            )
            
        weights = [m.weight for m in filtered]
        
        # Step 2: IQR outlier removal (framework section 2.3, step 2)
        cleaned_weights, outlier_info = self._iqr_outlier_removal(weights)
        
        if len(cleaned_weights) < self.min_readings:
            # Fallback: Use trimmed mean if IQR too aggressive
            logger.info("IQR removed too many points, using trimmed mean fallback")
            cleaned_weights = self._trimmed_weights(weights, self.trim_percent)
            
        if len(cleaned_weights) < self.min_readings:
            return BaselineResult(
                success=False,
                error=f"Too few readings after cleaning: {len(cleaned_weights)}",
                metadata={'outliers_removed': len(weights) - len(cleaned_weights)}
            )
            
        # Step 3: Calculate baseline weight (median as per framework)
        baseline_weight = np.median(cleaned_weights)
        
        # Step 4: Calculate initial variance estimate
        variance_info = self._calculate_variance(cleaned_weights, baseline_weight)
        
        # Determine confidence based on data quality
        confidence = self._assess_confidence(
            len(cleaned_weights),
            len(weights),
            variance_info['coefficient_of_variation']
        )
        
        return BaselineResult(
            success=True,
            baseline_weight=baseline_weight,
            measurement_variance=variance_info['variance'],
            measurement_noise_std=variance_info['std'],
            confidence=confidence,
            readings_used=len(cleaned_weights),
            method="IQR→Median→MAD",
            metadata={
                'original_count': len(measurements),
                'window_count': len(filtered),
                'outliers_removed': outlier_info['outliers_removed'],
                'mad': variance_info['mad'],
                'iqr_bounds': outlier_info,
                'percentiles': self._calculate_percentiles(cleaned_weights)
            }
        )
        
    def _filter_baseline_window(self, 
                               measurements: List[WeightMeasurement],
                               start_date: Optional[datetime]) -> List[WeightMeasurement]:
        """Filter measurements to baseline collection window."""
        if not start_date:
            # Use first measurement date
            sorted_measurements = sorted(measurements, key=lambda m: m.timestamp)
            if sorted_measurements:
                start_date = sorted_measurements[0].timestamp
            else:
                return measurements[:self.max_readings]
                
        end_date = start_date + timedelta(days=self.collection_days)
        max_end_date = start_date + timedelta(days=self.max_collection_days)
        
        filtered = []
        for m in measurements:
            if m.timestamp and start_date <= m.timestamp <= end_date:
                filtered.append(m)
                if len(filtered) >= self.max_readings:
                    break
            # Extended window if needed
            elif m.timestamp and end_date < m.timestamp <= max_end_date:
                if len(filtered) < self.min_readings:
                    filtered.append(m)
                    
        return filtered
        
    def _iqr_outlier_removal(self, weights: List[float]) -> Tuple[List[float], Dict]:
        """
        IQR-based outlier removal as specified in framework.
        Section 2.3, Step 2: Q1 - 1.5*IQR to Q3 + 1.5*IQR
        """
        q1 = np.percentile(weights, 25)
        q3 = np.percentile(weights, 75)
        iqr = q3 - q1
        
        # Calculate fences
        lower_fence = q1 - self.iqr_multiplier * iqr
        upper_fence = q3 + self.iqr_multiplier * iqr
        
        # Filter outliers
        cleaned = [w for w in weights if lower_fence <= w <= upper_fence]
        
        outlier_info = {
            'q1': q1,
            'q3': q3,
            'iqr': iqr,
            'lower_fence': lower_fence,
            'upper_fence': upper_fence,
            'outliers_removed': len(weights) - len(cleaned)
        }
        
        return cleaned, outlier_info
        
    def _calculate_variance(self, weights: List[float], baseline: float) -> Dict[str, float]:
        """
        Calculate variance using MAD as specified in framework.
        Section 2.3, Step 4: MAD-based robust variance estimation
        """
        # Calculate MAD
        deviations = [abs(w - baseline) for w in weights]
        mad = np.median(deviations)
        
        # Convert to standard deviation (framework: σ = k * MAD, k ≈ 1.4826)
        std = self.mad_scale * mad
        variance = std ** 2
        
        # Ensure minimum variance for numerical stability
        if variance < 0.01:
            variance = 0.01
            std = np.sqrt(variance)
            
        # Calculate coefficient of variation for confidence assessment
        cv = std / baseline if baseline > 0 else 0
        
        return {
            'variance': variance,
            'std': std,
            'mad': mad,
            'coefficient_of_variation': cv
        }
        
    def _trimmed_weights(self, weights: List[float], trim_percent: float) -> List[float]:
        """
        Alternative: Trimmed mean approach (framework section 2.2).
        Remove top and bottom trim_percent of values.
        """
        if len(weights) < 3:
            return weights
            
        sorted_weights = sorted(weights)
        trim_count = max(1, int(len(sorted_weights) * trim_percent / 100))
        
        if trim_count * 2 >= len(sorted_weights):
            return sorted_weights
            
        return sorted_weights[trim_count:-trim_count]
        
    def _winsorize_weights(self, weights: List[float], winsor_percent: float) -> List[float]:
        """
        Alternative: Winsorization approach (framework section 2.2).
        Replace extreme values with percentile values.
        """
        if len(weights) < 3:
            return weights
            
        lower_percentile = winsor_percent
        upper_percentile = 100 - winsor_percent
        
        lower_value = np.percentile(weights, lower_percentile)
        upper_value = np.percentile(weights, upper_percentile)
        
        winsorized = []
        for w in weights:
            if w < lower_value:
                winsorized.append(lower_value)
            elif w > upper_value:
                winsorized.append(upper_value)
            else:
                winsorized.append(w)
                
        return winsorized
        
    def _assess_confidence(self, 
                          readings_used: int,
                          original_count: int,
                          cv: float) -> str:
        """Assess baseline confidence based on data quality."""
        # High confidence criteria - relaxed for real-world data
        if readings_used >= 7 and cv < 0.03:  # 3% variation is normal for human weight
            return 'high'
        # Low confidence criteria
        elif readings_used < 3 or cv > 0.10:  # 10% variation indicates issues
            return 'low'
        # Medium confidence (default)
        else:
            return 'medium'
            
    def _calculate_percentiles(self, weights: List[float]) -> Dict[str, float]:
        """Calculate percentiles for additional context."""
        if len(weights) < 5:
            return {}
            
        return {
            'p5': np.percentile(weights, 5),
            'p25': np.percentile(weights, 25),
            'p50': np.percentile(weights, 50),  # median
            'p75': np.percentile(weights, 75),
            'p95': np.percentile(weights, 95)
        }