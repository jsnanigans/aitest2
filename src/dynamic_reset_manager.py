"""
Dynamic Reset Manager for Weight Processor

Implements intelligent reset strategies based on:
1. Source type (questionnaire vs device)
2. Variance detection
3. Source reliability scoring
4. Statistical change point detection

This module can be integrated with the existing processor to provide
more intelligent state reset decisions.
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional, List
from collections import deque
from scipy import stats

try:
    from .threshold_calculator import ThresholdCalculator
except ImportError:
    from threshold_calculator import ThresholdCalculator


class DynamicResetManager:
    """
    Manages dynamic reset decisions for the weight processor.
    
    Features:
    - Post-questionnaire gap reduction (10 days instead of 30)
    - Variance-based reset for outlier detection
    - Source-adaptive thresholds based on reliability
    - Statistical change point detection
    - Configurable voting mechanism for combined decisions
    """
    
    # Default configuration
    DEFAULT_CONFIG = {
        'questionnaire_gap_days': 10,      # Shorter gap after questionnaire
        'standard_gap_days': 30,           # Standard gap for reset
        'variance_threshold': 0.15,        # 15% variance triggers reset
        'trend_reversal_threshold': 0.5,   # 0.5 kg/day trend reversal
        'change_point_z_score': 3.0,       # Z-score for change point detection
        'cusum_k_factor': 0.5,             # CUSUM allowance factor
        'cusum_h_factor': 4.0,             # CUSUM decision factor
        'min_history_for_changepoint': 5,  # Minimum measurements for change detection
        'max_history_length': 20,          # Maximum history to keep
        'combined_vote_threshold': 2,      # Votes needed for combined reset
        'enable_questionnaire_gap': True,  # Enable questionnaire gap strategy
        'enable_variance_reset': True,     # Enable variance detection
        'enable_reliability_reset': True,  # Enable source reliability
        'enable_changepoint_reset': False, # Disable by default (more aggressive)
    }
    
    # Source reliability gap thresholds (days)
    RELIABILITY_GAPS = {
        'excellent': 45,
        'good': 30,
        'moderate': 20,
        'poor': 15,
        'unknown': 25
    }
    
    # Questionnaire source patterns
    QUESTIONNAIRE_SOURCES = [
        'questionnaire',
        'internal-questionnaire',
        'initial-questionnaire',
        'care-team-upload',  # Often contains questionnaire data
    ]
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the Dynamic Reset Manager.
        
        Args:
            config: Optional configuration dictionary to override defaults
        """
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        self.reset_history = deque(maxlen=100)  # Track recent resets
        
    def should_reset(
        self,
        current_weight: float,
        timestamp: datetime,
        source: str,
        state: Dict,
        method: str = 'auto'
    ) -> Tuple[bool, str, Dict]:
        """
        Determine if state should be reset.
        
        Args:
            current_weight: Current weight measurement (kg)
            timestamp: Current measurement timestamp
            source: Data source identifier
            state: Current processor state
            method: Reset method ('auto', 'questionnaire_gap', 'variance', 
                   'source_reliability', 'change_point', 'combined')
        
        Returns:
            Tuple of (should_reset, reason, metadata)
        """
        
        # Initialize metadata
        metadata = {
            'timestamp': timestamp,
            'source': source,
            'weight': current_weight,
            'method': method
        }
        
        # No previous state - no reset needed
        if not state or not state.get('last_timestamp'):
            return False, "No previous state", metadata
        
        # Calculate time gap
        last_timestamp = state['last_timestamp']
        if isinstance(last_timestamp, str):
            last_timestamp = datetime.fromisoformat(last_timestamp)
        
        gap_days = (timestamp - last_timestamp).total_seconds() / 86400
        metadata['gap_days'] = gap_days
        
        # Choose reset method
        if method == 'auto':
            # Auto-select based on configuration
            if self.config['enable_questionnaire_gap']:
                method = 'questionnaire_gap'
            elif self.config['enable_variance_reset']:
                method = 'variance'
            elif self.config['enable_reliability_reset']:
                method = 'source_reliability'
            else:
                method = 'standard'
        
        # Apply selected method
        if method == 'questionnaire_gap':
            return self._questionnaire_gap_reset(source, gap_days, state, metadata)
        elif method == 'variance':
            return self._variance_reset(current_weight, state, metadata)
        elif method == 'source_reliability':
            return self._reliability_reset(source, gap_days, metadata)
        elif method == 'change_point':
            return self._changepoint_reset(current_weight, state, metadata)
        elif method == 'combined':
            return self._combined_reset(current_weight, timestamp, source, state, gap_days, metadata)
        else:
            # Standard gap-based reset
            if gap_days > self.config['standard_gap_days']:
                reason = f"Standard gap reset ({gap_days:.1f} days)"
                metadata['reset_type'] = 'standard_gap'
                return True, reason, metadata
            return False, "No reset needed", metadata
    
    def _questionnaire_gap_reset(
        self, 
        source: str, 
        gap_days: float, 
        state: Dict,
        metadata: Dict
    ) -> Tuple[bool, str, Dict]:
        """Apply questionnaire-specific gap threshold."""
        
        # Check if last source was questionnaire
        last_source = state.get('last_source', '')
        
        # Check if last measurement was from questionnaire
        is_post_questionnaire = any(
            q in last_source.lower() 
            for q in self.QUESTIONNAIRE_SOURCES
        )
        
        metadata['last_source'] = last_source
        metadata['is_post_questionnaire'] = is_post_questionnaire
        
        if is_post_questionnaire:
            threshold = self.config['questionnaire_gap_days']
            if gap_days > threshold:
                reason = f"Post-questionnaire reset (gap: {gap_days:.1f}/{threshold} days)"
                metadata['reset_type'] = 'questionnaire_gap'
                metadata['threshold_used'] = threshold
                return True, reason, metadata
        else:
            threshold = self.config['standard_gap_days']
            if gap_days > threshold:
                reason = f"Standard gap reset ({gap_days:.1f}/{threshold} days)"
                metadata['reset_type'] = 'standard_gap'
                metadata['threshold_used'] = threshold
                return True, reason, metadata
        
        return False, f"Within threshold ({gap_days:.1f}/{threshold} days)", metadata
    
    def _variance_reset(
        self, 
        current_weight: float, 
        state: Dict,
        metadata: Dict
    ) -> Tuple[bool, str, Dict]:
        """Detect high variance or trend reversals."""
        
        if state.get('last_state') is None:
            return False, "No state history", metadata
        
        # Extract last filtered weight
        last_state = state['last_state']
        if isinstance(last_state, np.ndarray):
            if len(last_state.shape) > 1:
                filtered_weight = last_state[-1][0]
                trend = last_state[-1][1]
            else:
                filtered_weight = last_state[0]
                trend = 0
        else:
            filtered_weight = last_state[0] if isinstance(last_state, (list, tuple)) else last_state
            trend = 0
        
        # Calculate deviation
        deviation = abs(current_weight - filtered_weight)
        deviation_pct = deviation / filtered_weight
        
        metadata['filtered_weight'] = filtered_weight
        metadata['deviation'] = deviation
        metadata['deviation_pct'] = deviation_pct
        metadata['trend'] = trend
        
        # Check variance threshold
        if deviation_pct > self.config['variance_threshold']:
            reason = f"High variance reset (deviation: {deviation_pct:.1%})"
            metadata['reset_type'] = 'variance'
            return True, reason, metadata
        
        # Check trend reversal
        if abs(trend) > self.config['trend_reversal_threshold']:
            weight_change = current_weight - filtered_weight
            if np.sign(weight_change) != np.sign(trend):
                reason = f"Trend reversal (trend: {trend:.2f}, change: {weight_change:.2f})"
                metadata['reset_type'] = 'trend_reversal'
                return True, reason, metadata
        
        return False, "Variance within bounds", metadata
    
    def _reliability_reset(
        self, 
        source: str, 
        gap_days: float,
        metadata: Dict
    ) -> Tuple[bool, str, Dict]:
        """Apply source reliability-based thresholds."""
        
        # Get source reliability
        reliability = ThresholdCalculator.get_source_reliability(source)
        threshold = self.RELIABILITY_GAPS.get(reliability, 30)
        
        metadata['source_reliability'] = reliability
        metadata['reliability_threshold'] = threshold
        
        if gap_days > threshold:
            reason = f"Source-adaptive reset ({reliability}, gap: {gap_days:.1f}/{threshold} days)"
            metadata['reset_type'] = 'source_reliability'
            return True, reason, metadata
        
        return False, f"Within {reliability} threshold ({gap_days:.1f}/{threshold} days)", metadata
    
    def _changepoint_reset(
        self, 
        current_weight: float, 
        state: Dict,
        metadata: Dict
    ) -> Tuple[bool, str, Dict]:
        """Statistical change point detection using CUSUM."""
        
        # Initialize measurement history if needed
        if 'measurement_history' not in state:
            state['measurement_history'] = []
        
        history = state['measurement_history']
        history.append(current_weight)
        
        # Limit history length
        if len(history) > self.config['max_history_length']:
            history = history[-self.config['max_history_length']:]
            state['measurement_history'] = history
        
        metadata['history_length'] = len(history)
        
        # Need minimum history
        if len(history) < self.config['min_history_for_changepoint']:
            return False, "Insufficient history", metadata
        
        # Calculate statistics
        recent_history = history[:-1]  # Exclude current
        mean = np.mean(recent_history)
        std = np.std(recent_history) if len(recent_history) > 1 else 1.0
        
        metadata['history_mean'] = mean
        metadata['history_std'] = std
        
        # Z-score test
        if std > 0:
            z_score = abs((current_weight - mean) / std)
            metadata['z_score'] = z_score
            
            if z_score > self.config['change_point_z_score']:
                reason = f"Change point detected (z-score: {z_score:.2f})"
                metadata['reset_type'] = 'change_point'
                # Reset history after change point
                state['measurement_history'] = [current_weight]
                return True, reason, metadata
        
        # CUSUM test for more gradual changes
        if len(history) >= 10:
            k = self.config['cusum_k_factor'] * std  # Allowance
            h = self.config['cusum_h_factor'] * std  # Threshold
            
            cusum_pos = 0
            cusum_neg = 0
            
            for i in range(1, len(history)):
                diff = history[i] - history[i-1]
                cusum_pos = max(0, cusum_pos + diff - k)
                cusum_neg = max(0, cusum_neg - diff - k)
                
                if cusum_pos > h or cusum_neg > h:
                    reason = f"CUSUM change detected (pos: {cusum_pos:.2f}, neg: {cusum_neg:.2f})"
                    metadata['reset_type'] = 'cusum'
                    metadata['cusum_pos'] = cusum_pos
                    metadata['cusum_neg'] = cusum_neg
                    state['measurement_history'] = [current_weight]
                    return True, reason, metadata
        
        return False, "No change point detected", metadata
    
    def _combined_reset(
        self,
        current_weight: float,
        timestamp: datetime,
        source: str,
        state: Dict,
        gap_days: float,
        metadata: Dict
    ) -> Tuple[bool, str, Dict]:
        """Combine multiple methods with voting."""
        
        votes = []
        reasons = []
        vote_details = {}
        
        # Method 1: Questionnaire gap
        if self.config['enable_questionnaire_gap']:
            reset, reason, _ = self._questionnaire_gap_reset(source, gap_days, state, {})
            if reset:
                votes.append('questionnaire')
                reasons.append(reason)
                vote_details['questionnaire'] = reason
        
        # Method 2: Variance
        if self.config['enable_variance_reset']:
            reset, reason, _ = self._variance_reset(current_weight, state, {})
            if reset:
                votes.append('variance')
                reasons.append(reason)
                vote_details['variance'] = reason
        
        # Method 3: Source reliability
        if self.config['enable_reliability_reset']:
            reset, reason, _ = self._reliability_reset(source, gap_days, {})
            if reset:
                votes.append('reliability')
                reasons.append(reason)
                vote_details['reliability'] = reason
        
        # Method 4: Change point
        if self.config['enable_changepoint_reset']:
            reset, reason, _ = self._changepoint_reset(current_weight, state, {})
            if reset:
                votes.append('changepoint')
                reasons.append(reason)
                vote_details['changepoint'] = reason
        
        metadata['votes'] = votes
        metadata['vote_count'] = len(votes)
        metadata['vote_details'] = vote_details
        metadata['vote_threshold'] = self.config['combined_vote_threshold']
        
        # Check if enough votes for reset
        if len(votes) >= self.config['combined_vote_threshold']:
            reason = f"Combined reset ({len(votes)} triggers: {', '.join(votes)})"
            metadata['reset_type'] = 'combined'
            return True, reason, metadata
        
        return False, f"Insufficient votes ({len(votes)}/{self.config['combined_vote_threshold']})", metadata
    
    def track_reset(
        self,
        timestamp: datetime,
        source: str,
        reason: str,
        metadata: Dict
    ):
        """
        Track reset event for analysis.
        
        Args:
            timestamp: Reset timestamp
            source: Data source that triggered reset
            reason: Reset reason
            metadata: Additional metadata
        """
        self.reset_history.append({
            'timestamp': timestamp,
            'source': source,
            'reason': reason,
            'metadata': metadata
        })
    
    def get_reset_statistics(self) -> Dict:
        """
        Get statistics about recent resets.
        
        Returns:
            Dictionary with reset statistics
        """
        if not self.reset_history:
            return {
                'total_resets': 0,
                'reset_types': {},
                'average_gap_days': 0
            }
        
        reset_types = {}
        total_gap_days = 0
        
        for reset in self.reset_history:
            reset_type = reset.get('metadata', {}).get('reset_type', 'unknown')
            reset_types[reset_type] = reset_types.get(reset_type, 0) + 1
            total_gap_days += reset.get('metadata', {}).get('gap_days', 0)
        
        return {
            'total_resets': len(self.reset_history),
            'reset_types': reset_types,
            'average_gap_days': total_gap_days / len(self.reset_history) if self.reset_history else 0,
            'recent_resets': list(self.reset_history)[-5:]  # Last 5 resets
        }
    
    def update_state_with_source(self, state: Dict, source: str):
        """
        Update state with source information for next reset decision.
        
        Args:
            state: Processor state to update
            source: Current source identifier
        """
        state['last_source'] = source
        
        # Initialize measurement history if using change point detection
        if self.config['enable_changepoint_reset'] and 'measurement_history' not in state:
            state['measurement_history'] = []