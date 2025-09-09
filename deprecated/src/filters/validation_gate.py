from typing import Tuple, Optional, Dict, List
from collections import Counter
import numpy as np
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class ValidationGate:
    
    def __init__(self, gamma: float = 3.0, enable_adaptive: bool = True):
        self.gamma = gamma
        self.base_gamma = gamma
        self.enable_adaptive = enable_adaptive
        
        self.stats = {
            'accepted': 0,
            'rejected': 0,
            'rejection_reasons': [],
            'confidence_scores': []
        }
        
        self.rejection_history = []
        self.innovation_history = []
        
    def validate(self, 
                 measurement: float, 
                 prediction: float, 
                 innovation_covariance: float,
                 user_history: Optional[List[Dict]] = None) -> Tuple[bool, float, Optional[str]]:
        
        if self.enable_adaptive and user_history:
            self.gamma = self._calculate_adaptive_gamma(user_history)
        
        innovation = measurement - prediction
        uncertainty = np.sqrt(innovation_covariance)
        
        if uncertainty <= 0:
            logger.debug("Invalid uncertainty, accepting measurement")
            return True, 0.5, None
            
        normalized_innovation = abs(innovation) / uncertainty
        
        self.innovation_history.append({
            'innovation': innovation,
            'normalized': normalized_innovation,
            'uncertainty': uncertainty
        })
        
        is_valid = normalized_innovation <= self.gamma
        
        confidence = self._calculate_confidence(normalized_innovation)
        
        reason = None
        if not is_valid:
            reason = self._determine_rejection_reason(normalized_innovation)
            self.stats['rejected'] += 1
            self.stats['rejection_reasons'].append(reason)
            self.rejection_history.append({
                'measurement': measurement,
                'prediction': prediction,
                'normalized_innovation': normalized_innovation,
                'reason': reason
            })
        else:
            self.stats['accepted'] += 1
            
        self.stats['confidence_scores'].append(confidence)
        
        return is_valid, confidence, reason
    
    def _calculate_confidence(self, normalized_innovation: float) -> float:
        confidence = 2 * (1 - stats.norm.cdf(normalized_innovation))
        
        return max(0.01, min(1.0, confidence))
    
    def _determine_rejection_reason(self, normalized_innovation: float) -> str:
        if normalized_innovation > 6:
            return "extreme_outlier"
        elif normalized_innovation > 5:
            return "severe_deviation"
        elif normalized_innovation > 4:
            return "major_deviation"
        else:
            return "exceeds_threshold"
    
    def _calculate_adaptive_gamma(self, user_history: List[Dict]) -> float:
        if len(user_history) < 30:
            return self.base_gamma
            
        recent_history = user_history[-30:]
        
        innovations = []
        for h in recent_history:
            if 'normalized_innovation' in h:
                innovations.append(h['normalized_innovation'])
        
        if not innovations:
            return self.base_gamma
            
        innovation_std = np.std(innovations)
        
        # Check for consecutive rejections (filter might be stuck)
        recent_rejections = len([r for r in self.rejection_history[-10:]])
        consecutive_rejections = 0
        if len(self.rejection_history) > 0:
            # Count how many of the last measurements were rejected
            for i in range(min(10, len(self.rejection_history))):
                if i < len(self.rejection_history):
                    consecutive_rejections += 1
                else:
                    break
        
        # If we have many consecutive rejections, be much more permissive
        if consecutive_rejections >= 5:
            logger.debug(f"Detected {consecutive_rejections} consecutive rejections - significantly increasing gamma to allow recovery")
            return min(self.base_gamma + 3.0, 8.0)
        
        # Check for recent high rejection rate (possible filter stuck)
        recent_rejection_rate = self.stats['rejected'] / max(1, self.stats['accepted'] + self.stats['rejected'])
        if recent_rejection_rate > 0.5 and self.stats['rejected'] >= 5:
            # Too many rejections - be more permissive
            logger.debug(f"High rejection rate {recent_rejection_rate:.2%} - increasing gamma")
            return min(self.base_gamma + 2.0, 6.0)
        
        if innovation_std < 0.5:
            return self.base_gamma - 0.5
        elif innovation_std > 2.0:
            return self.base_gamma + 0.5
        else:
            return self.base_gamma
    
    def get_multi_level_feedback(self, 
                                  measurement: float, 
                                  prediction: float,
                                  innovation_covariance: float) -> Tuple[str, str, float]:
        
        uncertainty = np.sqrt(innovation_covariance)
        if uncertainty <= 0:
            return 'error', 'Invalid uncertainty', 0.0
            
        normalized_innovation = abs(measurement - prediction) / uncertainty
        
        levels = [
            (2.0, 'normal', 'Within expected range'),
            (2.5, 'marginal', 'Slightly unusual but acceptable'),
            (3.0, 'suspicious', 'Unusual - please verify'),
            (4.0, 'likely_error', 'Likely measurement error'),
            (float('inf'), 'rejected', 'Measurement rejected - extreme outlier')
        ]
        
        for threshold, status, message in levels:
            if normalized_innovation <= threshold:
                return status, message, normalized_innovation
                
        return 'rejected', 'Extreme outlier', normalized_innovation
    
    def get_metrics(self) -> Dict:
        total = self.stats['accepted'] + self.stats['rejected']
        
        if total == 0:
            return {
                'acceptance_rate': 0.0,
                'rejection_rate': 0.0,
                'average_confidence': 0.0,
                'health_score': 0.0
            }
        
        acceptance_rate = self.stats['accepted'] / total
        rejection_rate = self.stats['rejected'] / total
        
        avg_confidence = (
            np.mean(self.stats['confidence_scores']) 
            if self.stats['confidence_scores'] else 0.0
        )
        
        health_score = 1.0 if 0.95 <= acceptance_rate <= 0.99 else 0.5
        
        reason_counts = Counter(self.stats['rejection_reasons'])
        most_common_reason = (
            reason_counts.most_common(1)[0] 
            if reason_counts else ('none', 0)
        )
        
        return {
            'acceptance_rate': acceptance_rate,
            'rejection_rate': rejection_rate,
            'average_confidence': avg_confidence,
            'health_score': health_score,
            'most_common_rejection': most_common_reason,
            'total_processed': total,
            'total_accepted': self.stats['accepted'],
            'total_rejected': self.stats['rejected']
        }
    
    def should_rebaseline(self) -> bool:
        if len(self.rejection_history) < 3:
            return False
            
        recent_rejections = self.rejection_history[-5:]
        rejection_rate = len(recent_rejections) / 5.0
        
        return rejection_rate > 0.6
    
    def reset(self):
        self.gamma = self.base_gamma
        self.stats = {
            'accepted': 0,
            'rejected': 0,
            'rejection_reasons': [],
            'confidence_scores': []
        }
        self.rejection_history = []
        self.innovation_history = []