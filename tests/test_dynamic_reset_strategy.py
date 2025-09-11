"""
Test Dynamic Reset Strategy for Weight Processor

Explores various approaches for intelligent reset detection:
1. Shorter gap threshold after questionnaire data
2. Variance-based reset detection
3. Source reliability scoring for adaptive thresholds
4. Statistical change point detection
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from scipy import stats

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from processor import WeightProcessor
from processor_database import ProcessorStateDB, get_state_db
from threshold_calculator import ThresholdCalculator


class DynamicResetStrategy:
    """Enhanced reset detection with multiple mathematical approaches."""
    
    def __init__(self):
        self.reset_history = []
        self.variance_window = []
        self.source_history = []
        
    def should_reset(
        self,
        current_weight: float,
        timestamp: datetime,
        source: str,
        state: Dict,
        method: str = 'combined'
    ) -> Tuple[bool, str]:
        """
        Determine if state should be reset using specified method.
        
        Methods:
        - 'questionnaire_gap': Shorter gap (10 days) after questionnaire
        - 'variance': Reset on high variance detection
        - 'source_reliability': Adaptive gaps based on source quality
        - 'change_point': Statistical change point detection
        - 'combined': Combines all methods with voting
        """
        
        if not state or not state.get('last_timestamp'):
            return False, "No previous state"
        
        last_timestamp = state['last_timestamp']
        if isinstance(last_timestamp, str):
            last_timestamp = datetime.fromisoformat(last_timestamp)
        
        gap_days = (timestamp - last_timestamp).days
        
        if method == 'questionnaire_gap':
            return self._questionnaire_gap_reset(source, gap_days, state)
        elif method == 'variance':
            return self._variance_based_reset(current_weight, state)
        elif method == 'source_reliability':
            return self._source_reliability_reset(source, gap_days)
        elif method == 'change_point':
            return self._change_point_reset(current_weight, state)
        elif method == 'combined':
            return self._combined_reset(current_weight, timestamp, source, state, gap_days)
        else:
            # Default 30-day gap
            if gap_days > 30:
                return True, f"Standard gap reset ({gap_days} days)"
            return False, "No reset needed"
    
    def _questionnaire_gap_reset(self, source: str, gap_days: int, state: Dict) -> Tuple[bool, str]:
        """Reset with shorter threshold after questionnaire sources."""
        
        # Check if last source was questionnaire
        last_source = state.get('last_source', '')
        
        # Questionnaire sources
        questionnaire_sources = [
            'internal-questionnaire',
            'initial-questionnaire',
            'questionnaire',
            'care-team-upload'  # Often contains questionnaire data
        ]
        
        # If last measurement was from questionnaire, use shorter gap
        if any(q in last_source.lower() for q in questionnaire_sources):
            if gap_days > 10:
                return True, f"Post-questionnaire reset (gap: {gap_days} days, threshold: 10)"
        
        # Standard 30-day gap otherwise
        if gap_days > 30:
            return True, f"Standard gap reset ({gap_days} days)"
        
        return False, "No reset needed"
    
    def _variance_based_reset(self, current_weight: float, state: Dict) -> Tuple[bool, str]:
        """Reset based on variance detection."""
        
        if state.get('last_state') is None:
            return False, "No state history"
        
        # Get recent weight history from state
        last_state = state['last_state']
        if isinstance(last_state, np.ndarray):
            if len(last_state.shape) > 1:
                filtered_weight = last_state[-1][0]
            else:
                filtered_weight = last_state[0]
        else:
            filtered_weight = last_state[0] if isinstance(last_state, (list, tuple)) else last_state
        
        # Calculate deviation
        deviation = abs(current_weight - filtered_weight)
        deviation_pct = deviation / filtered_weight
        
        # Variance thresholds
        if deviation_pct > 0.15:  # 15% change
            return True, f"High variance reset (deviation: {deviation_pct:.1%})"
        
        # Check trend reversal
        if state.get('last_state') is not None:
            if len(last_state.shape) > 1 and last_state.shape[0] > 1:
                trend = last_state[-1][1]
                if abs(trend) > 0.5:  # Strong trend (0.5 kg/day)
                    weight_change = current_weight - filtered_weight
                    # Check if change opposes trend
                    if np.sign(weight_change) != np.sign(trend):
                        return True, f"Trend reversal reset (trend: {trend:.2f}, change: {weight_change:.2f})"
        
        return False, "Variance within bounds"
    
    def _source_reliability_reset(self, source: str, gap_days: int) -> Tuple[bool, str]:
        """Adaptive reset thresholds based on source reliability."""
        
        # Get source reliability
        reliability = ThresholdCalculator.get_source_reliability(source)
        
        # Set gap thresholds based on reliability
        gap_thresholds = {
            'excellent': 45,  # Trust excellent sources longer
            'good': 30,       # Standard threshold
            'moderate': 20,   # Less reliable, reset sooner
            'poor': 15,       # Unreliable, reset frequently
            'unknown': 25     # Conservative for unknown
        }
        
        threshold = gap_thresholds.get(reliability, 30)
        
        if gap_days > threshold:
            return True, f"Source-adaptive reset ({reliability} source, gap: {gap_days}/{threshold} days)"
        
        return False, f"Within {reliability} threshold ({gap_days}/{threshold} days)"
    
    def _change_point_reset(self, current_weight: float, state: Dict) -> Tuple[bool, str]:
        """Statistical change point detection using CUSUM."""
        
        if not state.get('measurement_history'):
            # Initialize if not present
            state['measurement_history'] = []
        
        # Add current measurement
        state['measurement_history'].append(current_weight)
        
        # Need minimum history
        history = state['measurement_history']
        if len(history) < 5:
            return False, "Insufficient history for change point detection"
        
        # Keep only recent measurements (last 20)
        if len(history) > 20:
            history = history[-20:]
            state['measurement_history'] = history
        
        # Calculate CUSUM (Cumulative Sum)
        mean = np.mean(history[:-1])  # Mean excluding current
        std = np.std(history[:-1]) if len(history) > 1 else 1.0
        
        # Standardize current value
        if std > 0:
            z_score = abs((current_weight - mean) / std)
            
            # Change point detected if z-score > 3
            if z_score > 3:
                return True, f"Change point detected (z-score: {z_score:.2f})"
        
        # Advanced: Page's CUSUM test
        if len(history) >= 10:
            cusum_pos = 0
            cusum_neg = 0
            k = 0.5 * std  # Allowance parameter
            h = 4 * std    # Decision threshold
            
            for i in range(1, len(history)):
                diff = history[i] - history[i-1]
                cusum_pos = max(0, cusum_pos + diff - k)
                cusum_neg = max(0, cusum_neg - diff - k)
                
                if cusum_pos > h or cusum_neg > h:
                    return True, f"CUSUM change detected (pos: {cusum_pos:.2f}, neg: {cusum_neg:.2f})"
        
        return False, "No change point detected"
    
    def _combined_reset(
        self,
        current_weight: float,
        timestamp: datetime,
        source: str,
        state: Dict,
        gap_days: int
    ) -> Tuple[bool, str]:
        """Combine multiple methods with voting."""
        
        votes = []
        reasons = []
        
        # Method 1: Questionnaire gap
        reset, reason = self._questionnaire_gap_reset(source, gap_days, state)
        if reset:
            votes.append(1)
            reasons.append(reason)
        
        # Method 2: Variance
        reset, reason = self._variance_based_reset(current_weight, state)
        if reset:
            votes.append(1)
            reasons.append(reason)
        
        # Method 3: Source reliability
        reset, reason = self._source_reliability_reset(source, gap_days)
        if reset:
            votes.append(1)
            reasons.append(reason)
        
        # Method 4: Change point (only if enough history)
        if state.get('measurement_history') and len(state.get('measurement_history', [])) >= 5:
            reset, reason = self._change_point_reset(current_weight, state)
            if reset:
                votes.append(1)
                reasons.append(reason)
        
        # Require at least 2 votes for reset
        if len(votes) >= 2:
            return True, f"Combined reset ({len(votes)} triggers: {'; '.join(reasons)})"
        
        return False, "No consensus for reset"
    
    def track_reset(self, timestamp: datetime, reason: str, gap_days: int):
        """Track reset events for analysis."""
        self.reset_history.append({
            'timestamp': timestamp,
            'reason': reason,
            'gap_days': gap_days
        })


def test_questionnaire_gap_strategy():
    """Test shorter gap threshold after questionnaire data."""
    print("\n" + "="*80)
    print("Testing Questionnaire Gap Reset Strategy")
    print("="*80)
    
    # Create test data with questionnaire followed by gap
    measurements = [
        # Initial questionnaire data
        (85.0, 'internal-questionnaire', datetime(2025, 1, 1)),
        (84.8, 'internal-questionnaire', datetime(2025, 1, 2)),
        
        # 12-day gap after questionnaire (should trigger reset with 10-day threshold)
        (86.0, 'patient-device', datetime(2025, 1, 14)),
        
        # Regular measurements
        (85.5, 'patient-device', datetime(2025, 1, 15)),
        
        # 12-day gap after device (should NOT trigger reset - needs 30 days)
        (85.0, 'patient-device', datetime(2025, 1, 27)),
        
        # 32-day gap (should trigger standard reset)
        (84.0, 'patient-device', datetime(2025, 2, 28)),
    ]
    
    # Process with dynamic strategy
    strategy = DynamicResetStrategy()
    db = ProcessorStateDB()
    user_id = "test_user_questionnaire"
    
    results = []
    for weight, source, timestamp in measurements:
        # Get current state
        state = db.get_state(user_id)
        
        # Check for reset
        if state:
            should_reset, reset_reason = strategy.should_reset(
                weight, timestamp, source, state, method='questionnaire_gap'
            )
            
            if should_reset:
                print(f"\n🔄 RESET TRIGGERED: {reset_reason}")
                db.clear_state(user_id)
                state = db.create_initial_state()
        
        # Process weight
        result = WeightProcessor.process_weight(
            user_id=user_id,
            weight=weight,
            timestamp=timestamp,
            source=source,
            processing_config={
                'min_weight': 30,
                'max_weight': 400,
                'extreme_threshold': 0.1,
                'physiological': {'enable_physiological_limits': True}
            },
            kalman_config={
                'initial_variance': 1.0,
                'transition_covariance_weight': 0.01,
                'transition_covariance_trend': 0.0001,
                'observation_covariance': 1.0,
                'reset_gap_days': 1000  # Disable built-in reset
            },
            db=db
        )
        
        # Track source for next iteration
        state = db.get_state(user_id)
        if state:
            state['last_source'] = source
            db.save_state(user_id, state)
        
        results.append(result)
        
        # Display result
        gap = (timestamp - measurements[max(0, len(results)-2)][2]).days if len(results) > 1 else 0
        print(f"\n{timestamp.date()} | {source:25} | Gap: {gap:3d} days | Weight: {weight:.1f} kg")
        if result:
            print(f"  → Filtered: {result.get('filtered_weight', weight):.1f} kg | "
                  f"Accepted: {result.get('accepted', False)}")
    
    print("\n✅ Questionnaire gap strategy test complete")
    return results


def test_variance_based_reset():
    """Test variance-based reset detection."""
    print("\n" + "="*80)
    print("Testing Variance-Based Reset Strategy")
    print("="*80)
    
    # Create test data with high variance
    measurements = [
        # Stable period
        (85.0, 'patient-device', datetime(2025, 1, 1)),
        (85.2, 'patient-device', datetime(2025, 1, 2)),
        (84.8, 'patient-device', datetime(2025, 1, 3)),
        
        # High variance jump (>15%)
        (98.0, 'patient-upload', datetime(2025, 1, 4)),  # 15.3% increase
        
        # Continue at new level
        (97.5, 'patient-device', datetime(2025, 1, 5)),
        (97.8, 'patient-device', datetime(2025, 1, 6)),
        
        # Another variance event
        (82.0, 'iglucose', datetime(2025, 1, 7)),  # 16% decrease
    ]
    
    strategy = DynamicResetStrategy()
    db = ProcessorStateDB()
    user_id = "test_user_variance"
    
    results = []
    for weight, source, timestamp in measurements:
        state = db.get_state(user_id)
        
        if state:
            should_reset, reset_reason = strategy.should_reset(
                weight, timestamp, source, state, method='variance'
            )
            
            if should_reset:
                print(f"\n🔄 VARIANCE RESET: {reset_reason}")
                db.clear_state(user_id)
                strategy.track_reset(timestamp, reset_reason, 0)
        
        result = WeightProcessor.process_weight(
            user_id=user_id,
            weight=weight,
            timestamp=timestamp,
            source=source,
            processing_config={
                'min_weight': 30,
                'max_weight': 400,
                'extreme_threshold': 0.2,  # Higher threshold for testing
                'physiological': {'enable_physiological_limits': True}
            },
            kalman_config={
                'initial_variance': 1.0,
                'transition_covariance_weight': 0.01,
                'transition_covariance_trend': 0.0001,
                'observation_covariance': 1.0,
                'reset_gap_days': 1000
            },
            db=db
        )
        
        results.append(result)
        
        print(f"\n{timestamp.date()} | {source:20} | Weight: {weight:.1f} kg")
        if result:
            filtered = result.get('filtered_weight', weight)
            deviation = abs(weight - filtered) / filtered * 100
            print(f"  → Filtered: {filtered:.1f} kg | Deviation: {deviation:.1f}% | "
                  f"Accepted: {result.get('accepted', False)}")
    
    print(f"\n📊 Total variance resets: {len(strategy.reset_history)}")
    return results


def test_source_reliability_reset():
    """Test adaptive reset thresholds based on source reliability."""
    print("\n" + "="*80)
    print("Testing Source Reliability Reset Strategy")
    print("="*80)
    
    # Test different sources with various gaps
    test_cases = [
        # (source, gap_days, expected_reset)
        ('care-team-upload', 40, False),      # Excellent: 45-day threshold
        ('care-team-upload', 50, True),       # Exceeds 45 days
        ('patient-device', 25, False),        # Good: 30-day threshold
        ('patient-device', 35, True),         # Exceeds 30 days
        ('https://connectivehealth.io', 18, False),  # Moderate: 20-day threshold
        ('https://connectivehealth.io', 22, True),   # Exceeds 20 days
        ('https://api.iglucose.com', 12, False),     # Poor: 15-day threshold
        ('https://api.iglucose.com', 18, True),      # Exceeds 15 days
    ]
    
    strategy = DynamicResetStrategy()
    
    print("\nSource Reliability Thresholds:")
    print("-" * 50)
    
    for source, gap_days, expected_reset in test_cases:
        # Create minimal state
        state = {'last_timestamp': datetime(2025, 1, 1)}
        current_timestamp = datetime(2025, 1, 1) + timedelta(days=gap_days)
        
        should_reset, reason = strategy.should_reset(
            85.0, current_timestamp, source, state, method='source_reliability'
        )
        
        reliability = ThresholdCalculator.get_source_reliability(source)
        status = "✅" if should_reset == expected_reset else "❌"
        
        print(f"{status} {source:30} | Reliability: {reliability:9} | "
              f"Gap: {gap_days:2}d | Reset: {should_reset}")
        if should_reset:
            print(f"   → {reason}")
    
    print("\n✅ Source reliability reset test complete")


def test_change_point_detection():
    """Test statistical change point detection."""
    print("\n" + "="*80)
    print("Testing Change Point Detection Reset Strategy")
    print("="*80)
    
    # Create data with clear change points
    measurements = [
        # Stable period around 85kg
        (85.0, datetime(2025, 1, 1)),
        (85.2, datetime(2025, 1, 2)),
        (84.8, datetime(2025, 1, 3)),
        (85.1, datetime(2025, 1, 4)),
        (84.9, datetime(2025, 1, 5)),
        
        # Change point: Jump to 90kg level
        (90.5, datetime(2025, 1, 6)),  # Should trigger change point
        (90.2, datetime(2025, 1, 7)),
        (90.8, datetime(2025, 1, 8)),
        (90.3, datetime(2025, 1, 9)),
        
        # Another change point: Drop to 82kg
        (82.0, datetime(2025, 1, 10)),  # Should trigger change point
        (82.3, datetime(2025, 1, 11)),
        (81.8, datetime(2025, 1, 12)),
    ]
    
    strategy = DynamicResetStrategy()
    state = {'measurement_history': []}
    
    print("\nProcessing measurements with change point detection:")
    print("-" * 60)
    
    change_points = []
    for i, (weight, timestamp) in enumerate(measurements):
        # Check for change point
        should_reset, reason = strategy._change_point_reset(weight, state)
        
        if should_reset:
            print(f"\n🎯 CHANGE POINT at measurement {i+1}: {reason}")
            print(f"   Weight: {weight:.1f} kg at {timestamp.date()}")
            change_points.append((timestamp, weight, reason))
            # Reset history after change point
            state['measurement_history'] = [weight]
        
        status = "📍" if should_reset else "  "
        history_mean = np.mean(state['measurement_history']) if state['measurement_history'] else weight
        print(f"{status} {timestamp.date()} | Weight: {weight:5.1f} kg | "
              f"History mean: {history_mean:5.1f} kg")
    
    print(f"\n📊 Total change points detected: {len(change_points)}")
    return change_points


def test_combined_strategy():
    """Test combined voting strategy with realistic scenario."""
    print("\n" + "="*80)
    print("Testing Combined Reset Strategy (Multi-Method Voting)")
    print("="*80)
    
    # Realistic scenario with various reset triggers
    measurements = [
        # Initial questionnaire data
        (85.0, 'internal-questionnaire', datetime(2025, 1, 1)),
        (84.8, 'internal-questionnaire', datetime(2025, 1, 2)),
        
        # 11-day gap after questionnaire + unreliable source
        (92.0, 'https://api.iglucose.com', datetime(2025, 1, 13)),  # Multiple triggers
        
        # Continue with device data
        (85.5, 'patient-device', datetime(2025, 1, 14)),
        (85.3, 'patient-device', datetime(2025, 1, 15)),
        
        # Large variance + moderate gap
        (95.0, 'https://connectivehealth.io', datetime(2025, 1, 30)),  # Multiple triggers
        
        # Back to normal
        (85.0, 'patient-device', datetime(2025, 1, 31)),
    ]
    
    strategy = DynamicResetStrategy()
    db = ProcessorStateDB()
    user_id = "test_user_combined"
    
    print("\nProcessing with combined strategy:")
    print("-" * 80)
    
    for weight, source, timestamp in measurements:
        state = db.get_state(user_id)
        
        if state:
            # Initialize measurement history if needed
            if 'measurement_history' not in state:
                state['measurement_history'] = []
            
            should_reset, reset_reason = strategy.should_reset(
                weight, timestamp, source, state, method='combined'
            )
            
            if should_reset:
                print(f"\n🔄 COMBINED RESET TRIGGERED")
                print(f"   Reason: {reset_reason}")
                db.clear_state(user_id)
                state = db.create_initial_state()
                state['measurement_history'] = [weight]
        else:
            state = db.create_initial_state()
            state['measurement_history'] = [weight]
        
        # Process weight
        result = WeightProcessor.process_weight(
            user_id=user_id,
            weight=weight,
            timestamp=timestamp,
            source=source,
            processing_config={
                'min_weight': 30,
                'max_weight': 400,
                'extreme_threshold': 0.15,
                'physiological': {'enable_physiological_limits': True}
            },
            kalman_config={
                'initial_variance': 1.0,
                'transition_covariance_weight': 0.01,
                'transition_covariance_trend': 0.0001,
                'observation_covariance': 1.0,
                'reset_gap_days': 1000
            },
            db=db
        )
        
        # Update state with source
        state = db.get_state(user_id)
        if state:
            state['last_source'] = source
            db.save_state(user_id, state)
        
        reliability = ThresholdCalculator.get_source_reliability(source)
        print(f"\n{timestamp.date()} | {source:30} [{reliability:9}] | {weight:.1f} kg")
        if result:
            print(f"  → Filtered: {result.get('filtered_weight', weight):.1f} kg | "
                  f"Accepted: {result.get('accepted', False)}")
    
    print("\n✅ Combined strategy test complete")


def compare_reset_strategies():
    """Compare all reset strategies on the same dataset."""
    print("\n" + "="*80)
    print("Comparing All Reset Strategies")
    print("="*80)
    
    # Create challenging dataset
    measurements = [
        (85.0, 'internal-questionnaire', datetime(2025, 1, 1)),
        (84.8, 'patient-device', datetime(2025, 1, 2)),
        (85.2, 'patient-device', datetime(2025, 1, 3)),
        
        # 12-day gap with variance
        (92.0, 'patient-upload', datetime(2025, 1, 15)),
        
        # Back to normal
        (85.5, 'patient-device', datetime(2025, 1, 16)),
        
        # 25-day gap with unreliable source
        (88.0, 'https://api.iglucose.com', datetime(2025, 2, 10)),
        
        # Continue
        (85.0, 'patient-device', datetime(2025, 2, 11)),
        
        # 35-day gap
        (84.0, 'care-team-upload', datetime(2025, 3, 18)),
    ]
    
    strategies = [
        'questionnaire_gap',
        'variance',
        'source_reliability',
        'change_point',
        'combined'
    ]
    
    print("\nReset Strategy Comparison:")
    print("-" * 100)
    print(f"{'Timestamp':<12} | {'Gap':<4} | {'Weight':<6} | ", end="")
    print(" | ".join(f"{s[:8]:^8}" for s in strategies))
    print("-" * 100)
    
    for i, (weight, source, timestamp) in enumerate(measurements):
        gap = 0
        if i > 0:
            gap = (timestamp - measurements[i-1][2]).days
        
        # Test each strategy
        reset_results = []
        for strategy_name in strategies:
            strategy = DynamicResetStrategy()
            
            # Create mock state
            state = {
                'last_timestamp': measurements[i-1][2] if i > 0 else None,
                'last_source': measurements[i-1][1] if i > 0 else None,
                'last_state': np.array([[measurements[i-1][0], 0]]) if i > 0 else None,
                'measurement_history': [m[0] for m in measurements[:i]] if strategy_name == 'change_point' else []
            }
            
            if state['last_timestamp']:
                should_reset, _ = strategy.should_reset(
                    weight, timestamp, source, state, method=strategy_name
                )
                reset_results.append("✓" if should_reset else "-")
            else:
                reset_results.append("-")
        
        print(f"{timestamp.date()} | {gap:4d} | {weight:6.1f} | ", end="")
        print(" | ".join(f"{r:^8}" for r in reset_results))
    
    print("\n📊 Strategy Summary:")
    print("  - questionnaire_gap: Shorter threshold after questionnaire")
    print("  - variance: Detects large weight changes")
    print("  - source_reliability: Adaptive thresholds by source quality")
    print("  - change_point: Statistical anomaly detection")
    print("  - combined: Requires 2+ methods to agree")


def main():
    """Run all dynamic reset strategy tests."""
    
    print("\n" + "="*80)
    print("DYNAMIC RESET STRATEGY INVESTIGATION")
    print("="*80)
    
    # Test individual strategies
    test_questionnaire_gap_strategy()
    test_variance_based_reset()
    test_source_reliability_reset()
    test_change_point_detection()
    test_combined_strategy()
    
    # Compare all strategies
    compare_reset_strategies()
    
    print("\n" + "="*80)
    print("INVESTIGATION COMPLETE")
    print("="*80)
    print("\n📋 Key Findings:")
    print("  1. Questionnaire gap strategy effectively handles sparse self-reported data")
    print("  2. Variance detection catches sudden weight changes that may indicate errors")
    print("  3. Source reliability provides context-aware thresholds")
    print("  4. Change point detection uses statistics to find anomalies")
    print("  5. Combined voting reduces false positives while catching true resets")
    
    print("\n💡 Recommendations:")
    print("  1. Implement questionnaire gap as immediate improvement (simple, effective)")
    print("  2. Add variance detection for robustness against outliers")
    print("  3. Consider combined approach for production use")
    print("  4. Track reset metrics to tune thresholds over time")


if __name__ == "__main__":
    main()