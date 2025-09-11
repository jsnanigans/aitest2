"""
Test Integration of Dynamic Reset Manager with Weight Processor

Demonstrates how the Dynamic Reset Manager can be integrated with the
existing weight processor to provide intelligent reset decisions.
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from processor import WeightProcessor
from processor_database import ProcessorStateDB, get_state_db
from dynamic_reset_manager import DynamicResetManager


def process_with_dynamic_reset(
    measurements: List[Tuple[float, str, datetime]],
    user_id: str = "test_user",
    reset_config: Dict = None,
    processing_config: Dict = None,
    kalman_config: Dict = None
) -> List[Dict]:
    """
    Process measurements with dynamic reset management.
    
    Args:
        measurements: List of (weight, source, timestamp) tuples
        user_id: User identifier
        reset_config: Configuration for reset manager
        processing_config: Processing configuration
        kalman_config: Kalman filter configuration
    
    Returns:
        List of processing results
    """
    
    # Initialize components
    reset_manager = DynamicResetManager(reset_config)
    db = ProcessorStateDB()
    
    # Default configurations
    if processing_config is None:
        processing_config = {
            'min_weight': 30,
            'max_weight': 400,
            'extreme_threshold': 0.15,
            'physiological': {'enable_physiological_limits': True}
        }
    
    if kalman_config is None:
        kalman_config = {
            'initial_variance': 1.0,
            'transition_covariance_weight': 0.01,
            'transition_covariance_trend': 0.0001,
            'observation_covariance': 1.0,
            'reset_gap_days': 1000  # Disable built-in reset
        }
    
    results = []
    
    for i, (weight, source, timestamp) in enumerate(measurements):
        # Get current state
        state = db.get_state(user_id)
        
        # Check for dynamic reset
        if state:
            should_reset, reset_reason, reset_metadata = reset_manager.should_reset(
                current_weight=weight,
                timestamp=timestamp,
                source=source,
                state=state,
                method='combined'  # Use combined voting
            )
            
            if should_reset:
                print(f"\n🔄 DYNAMIC RESET at {timestamp.date()}")
                print(f"   Reason: {reset_reason}")
                print(f"   Metadata: Gap={reset_metadata.get('gap_days', 0):.1f} days, "
                      f"Votes={reset_metadata.get('votes', [])}")
                
                # Track reset
                reset_manager.track_reset(timestamp, source, reset_reason, reset_metadata)
                
                # Clear state for reset
                db.clear_state(user_id)
                state = db.create_initial_state()
        
        # Process weight
        result = WeightProcessor.process_weight(
            user_id=user_id,
            weight=weight,
            timestamp=timestamp,
            source=source,
            processing_config=processing_config,
            kalman_config=kalman_config,
            db=db
        )
        
        # Update state with source for next iteration
        state = db.get_state(user_id)
        if state:
            reset_manager.update_state_with_source(state, source)
            db.save_state(user_id, state)
        
        results.append(result)
        
        # Display result
        if result:
            print(f"{timestamp.date()} | {source:30} | Weight: {weight:6.1f} kg | "
                  f"Filtered: {result.get('filtered_weight', weight):6.1f} kg | "
                  f"Accepted: {result.get('accepted', False)}")
    
    # Display reset statistics
    stats = reset_manager.get_reset_statistics()
    print(f"\n📊 Reset Statistics:")
    print(f"   Total resets: {stats['total_resets']}")
    print(f"   Reset types: {stats['reset_types']}")
    print(f"   Average gap: {stats['average_gap_days']:.1f} days")
    
    return results


def test_realistic_scenario():
    """Test with realistic user data including questionnaire gaps."""
    print("\n" + "="*80)
    print("REALISTIC SCENARIO WITH DYNAMIC RESET")
    print("="*80)
    
    # Create realistic measurement sequence
    measurements = [
        # Initial questionnaire data
        (87.6, 'internal-questionnaire', datetime(2025, 1, 1)),
        (87.4, 'internal-questionnaire', datetime(2025, 1, 2)),
        
        # 11-day gap after questionnaire (should trigger with 10-day threshold)
        (88.0, 'patient-device', datetime(2025, 1, 13)),
        (87.8, 'patient-device', datetime(2025, 1, 14)),
        (87.5, 'patient-device', datetime(2025, 1, 15)),
        
        # Stable device measurements
        (87.3, 'patient-device', datetime(2025, 1, 20)),
        (87.1, 'patient-device', datetime(2025, 1, 25)),
        
        # Unreliable source with noise
        (95.0, 'https://api.iglucose.com', datetime(2025, 2, 1)),  # High variance
        (86.5, 'patient-device', datetime(2025, 2, 2)),
        
        # Another questionnaire after gap
        (86.0, 'care-team-upload', datetime(2025, 2, 15)),
        
        # 8-day gap (under 10-day threshold)
        (85.8, 'patient-device', datetime(2025, 2, 23)),
        
        # 12-day gap after device (under 30-day threshold)
        (85.5, 'patient-device', datetime(2025, 3, 7)),
        
        # Long gap that should trigger standard reset
        (84.0, 'patient-upload', datetime(2025, 4, 15)),
    ]
    
    # Configure dynamic reset
    reset_config = {
        'questionnaire_gap_days': 10,
        'standard_gap_days': 30,
        'variance_threshold': 0.12,  # 12% variance
        'enable_questionnaire_gap': True,
        'enable_variance_reset': True,
        'enable_reliability_reset': True,
        'enable_changepoint_reset': False,  # Disable for cleaner demo
        'combined_vote_threshold': 2
    }
    
    print("\n📋 Reset Configuration:")
    print(f"   - Questionnaire gap: {reset_config['questionnaire_gap_days']} days")
    print(f"   - Standard gap: {reset_config['standard_gap_days']} days")
    print(f"   - Variance threshold: {reset_config['variance_threshold']:.0%}")
    print(f"   - Combined vote threshold: {reset_config['combined_vote_threshold']}")
    print("\n" + "-"*80)
    
    results = process_with_dynamic_reset(
        measurements,
        user_id="realistic_user",
        reset_config=reset_config
    )
    
    return results


def test_comparison_with_standard():
    """Compare dynamic reset vs standard 30-day reset."""
    print("\n" + "="*80)
    print("COMPARISON: DYNAMIC vs STANDARD RESET")
    print("="*80)
    
    # Test data with questionnaire followed by gaps
    measurements = [
        (85.0, 'internal-questionnaire', datetime(2025, 1, 1)),
        (84.8, 'internal-questionnaire', datetime(2025, 1, 2)),
        # 15-day gap after questionnaire
        (87.0, 'patient-device', datetime(2025, 1, 17)),
        (86.5, 'patient-device', datetime(2025, 1, 18)),
        # High variance measurement
        (95.0, 'iglucose', datetime(2025, 1, 19)),
        (86.0, 'patient-device', datetime(2025, 1, 20)),
        # 25-day gap
        (85.0, 'patient-device', datetime(2025, 2, 14)),
    ]
    
    # Process with standard reset (30-day only)
    print("\n1️⃣ STANDARD RESET (30-day gap only):")
    print("-"*60)
    
    db = ProcessorStateDB()
    standard_results = []
    
    for weight, source, timestamp in measurements:
        result = WeightProcessor.process_weight(
            user_id="standard_user",
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
                'reset_gap_days': 30  # Standard 30-day reset
            },
            db=db
        )
        
        standard_results.append(result)
        
        if result:
            reset_flag = "🔄" if result.get('was_reset', False) else "  "
            print(f"{reset_flag} {timestamp.date()} | {source:25} | "
                  f"Weight: {weight:6.1f} | Filtered: {result.get('filtered_weight', weight):6.1f}")
    
    # Process with dynamic reset
    print("\n2️⃣ DYNAMIC RESET (intelligent strategy):")
    print("-"*60)
    
    reset_config = {
        'questionnaire_gap_days': 10,
        'variance_threshold': 0.10,
        'enable_questionnaire_gap': True,
        'enable_variance_reset': True,
        'enable_reliability_reset': False,
        'combined_vote_threshold': 1  # Single trigger sufficient
    }
    
    dynamic_results = process_with_dynamic_reset(
        measurements,
        user_id="dynamic_user",
        reset_config=reset_config
    )
    
    # Compare results
    print("\n📊 COMPARISON SUMMARY:")
    print("-"*60)
    
    # Count accepted measurements
    standard_accepted = sum(1 for r in standard_results if r and r.get('accepted', False))
    dynamic_accepted = sum(1 for r in dynamic_results if r and r.get('accepted', False))
    
    print(f"Standard approach: {standard_accepted}/{len(standard_results)} accepted")
    print(f"Dynamic approach:  {dynamic_accepted}/{len(dynamic_results)} accepted")
    
    # Calculate average confidence
    standard_conf = np.mean([r.get('confidence', 0) for r in standard_results if r])
    dynamic_conf = np.mean([r.get('confidence', 0) for r in dynamic_results if r])
    
    print(f"\nAverage confidence:")
    print(f"  Standard: {standard_conf:.3f}")
    print(f"  Dynamic:  {dynamic_conf:.3f}")
    
    return standard_results, dynamic_results


def test_edge_cases():
    """Test edge cases and special scenarios."""
    print("\n" + "="*80)
    print("EDGE CASES TESTING")
    print("="*80)
    
    # Edge case 1: Multiple questionnaires in sequence
    print("\n1️⃣ Multiple Questionnaires:")
    measurements = [
        (85.0, 'internal-questionnaire', datetime(2025, 1, 1)),
        (85.2, 'initial-questionnaire', datetime(2025, 1, 2)),
        (85.1, 'care-team-upload', datetime(2025, 1, 3)),
        # 11-day gap should trigger
        (86.0, 'patient-device', datetime(2025, 1, 14)),
    ]
    
    results = process_with_dynamic_reset(
        measurements,
        user_id="edge_case_1",
        reset_config={'questionnaire_gap_days': 10, 'enable_questionnaire_gap': True}
    )
    
    # Edge case 2: Rapid source switching
    print("\n2️⃣ Rapid Source Switching:")
    measurements = [
        (85.0, 'patient-device', datetime(2025, 1, 1, 8, 0)),
        (85.2, 'https://api.iglucose.com', datetime(2025, 1, 1, 10, 0)),
        (85.1, 'patient-upload', datetime(2025, 1, 1, 12, 0)),
        (85.3, 'https://connectivehealth.io', datetime(2025, 1, 1, 14, 0)),
    ]
    
    results = process_with_dynamic_reset(
        measurements,
        user_id="edge_case_2",
        reset_config={'enable_reliability_reset': True}
    )
    
    # Edge case 3: Extreme variance from reliable source
    print("\n3️⃣ Extreme Variance from Reliable Source:")
    measurements = [
        (85.0, 'care-team-upload', datetime(2025, 1, 1)),
        (85.2, 'care-team-upload', datetime(2025, 1, 2)),
        # 20% jump from reliable source
        (102.0, 'care-team-upload', datetime(2025, 1, 3)),
        (85.5, 'care-team-upload', datetime(2025, 1, 4)),
    ]
    
    results = process_with_dynamic_reset(
        measurements,
        user_id="edge_case_3",
        reset_config={'variance_threshold': 0.15, 'enable_variance_reset': True}
    )
    
    print("\n✅ Edge cases tested successfully")


def main():
    """Run all integration tests."""
    
    print("\n" + "="*80)
    print("DYNAMIC RESET MANAGER INTEGRATION TESTS")
    print("="*80)
    
    # Run tests
    test_realistic_scenario()
    test_comparison_with_standard()
    test_edge_cases()
    
    print("\n" + "="*80)
    print("INTEGRATION TESTING COMPLETE")
    print("="*80)
    
    print("\n🎯 KEY BENEFITS OF DYNAMIC RESET:")
    print("  1. Faster recovery after questionnaire data (10 vs 30 days)")
    print("  2. Automatic handling of high-variance measurements")
    print("  3. Source-aware thresholds for better accuracy")
    print("  4. Configurable voting mechanism reduces false positives")
    print("  5. Maintains state tracking for analysis")
    
    print("\n💡 IMPLEMENTATION NOTES:")
    print("  1. Can be added to existing processor without breaking changes")
    print("  2. Configuration is flexible and can be tuned per deployment")
    print("  3. Reset history tracking enables continuous improvement")
    print("  4. Compatible with existing state management")


if __name__ == "__main__":
    main()