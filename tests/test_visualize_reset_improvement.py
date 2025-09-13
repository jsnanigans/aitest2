"""
Visualize the improvement in reset behavior with rejected data.
Shows before/after comparison of reset logic.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta
from src.processor import WeightProcessor
from src.database import ProcessorStateDB
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def simulate_user_0040872d_pattern():
    """Simulate the pattern from user 0040872d in the image."""
    
    print("\n" + "="*70)
    print("SIMULATING USER 0040872d PATTERN")
    print("Shows how the new logic prevents unnecessary resets")
    print("="*70)
    
    db = ProcessorStateDB()
    user_id = "simulated_0040872d"
    
    processing_config = {
        "min_weight": 30.0,
        "max_weight": 300.0,
        "max_daily_change": 0.03,
        "extreme_threshold": 0.15,
        "user_height_m": 1.67,  # Average height
    }
    
    kalman_config = {
        "reset_gap_days": 30,
        "questionnaire_reset_days": 10,
    }
    
    # Simulate measurement pattern similar to the image
    measurements = [
        # Initial stable period
        (datetime(2016, 7, 1), 87.0, "iGlucose API", True),
        (datetime(2016, 7, 5), 87.5, "iGlucose API", True),
        (datetime(2016, 7, 10), 88.0, "iGlucose API", True),
        
        # Gap with rejected measurements (simulating the pattern in the image)
        (datetime(2016, 8, 1), 95.0, "patient-device", False),  # Rejected
        (datetime(2016, 8, 10), 98.0, "patient-device", False),  # Rejected
        (datetime(2016, 8, 20), 100.0, "patient-device", False), # Rejected
        
        # After gap - should NOT reset because of rejected data
        (datetime(2016, 9, 5), 89.0, "iGlucose API", True),
        (datetime(2016, 9, 10), 88.5, "iGlucose API", True),
    ]
    
    results = []
    for timestamp, weight, source, expected_accept in measurements:
        result = WeightProcessor.process_weight(
            user_id=user_id,
            weight=weight,
            timestamp=timestamp,
            source=source,
            processing_config=processing_config,
            kalman_config=kalman_config,
            db=db
        )
        results.append({
            'timestamp': timestamp,
            'weight': weight,
            'accepted': result['accepted'],
            'was_reset': result.get('was_reset', False),
            'filtered': result.get('filtered_weight'),
            'source': source
        })
        
        status = "✅" if result['accepted'] else "❌"
        reset = " [RESET]" if result.get('was_reset') else ""
        print(f"{timestamp.date()}: {weight:6.1f}kg - {status}{reset} - {source}")
        if not result['accepted']:
            print(f"    Rejection: {result.get('reason')}")
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Top plot: Old behavior (simulated)
    ax1.set_title("OLD BEHAVIOR: Reset after 30 days regardless of rejected data")
    ax1.set_ylabel("Weight (kg)")
    
    # Plot accepted and rejected points
    for r in results[:3]:  # Initial period
        ax1.scatter(r['timestamp'], r['weight'], 
                   color='green' if r['accepted'] else 'red',
                   s=50, alpha=0.7)
    
    # Show reset point (old behavior would reset here)
    ax1.axvline(x=datetime(2016, 9, 5), color='red', linestyle='--', alpha=0.5)
    ax1.text(datetime(2016, 9, 5), 92, "RESET\n(old logic)", 
            ha='center', va='bottom', color='red')
    
    # Bottom plot: New behavior
    ax2.set_title("NEW BEHAVIOR: No reset when rejected data exists")
    ax2.set_ylabel("Weight (kg)")
    ax2.set_xlabel("Date")
    
    # Plot all points
    for r in results:
        marker = 'o' if r['accepted'] else 'x'
        ax2.scatter(r['timestamp'], r['weight'],
                   color='green' if r['accepted'] else 'orange',
                   marker=marker, s=50, alpha=0.7)
    
    # Show continuous line (no reset)
    accepted_data = [r for r in results if r['accepted']]
    if len(accepted_data) > 1:
        timestamps = [r['timestamp'] for r in accepted_data]
        weights = [r['weight'] for r in accepted_data]
        ax2.plot(timestamps, weights, 'b-', alpha=0.3, linewidth=2)
    
    # Add legend
    green_patch = mpatches.Patch(color='green', label='Accepted')
    orange_patch = mpatches.Patch(color='orange', label='Rejected (prevents reset)')
    ax2.legend(handles=[green_patch, orange_patch])
    
    # Format x-axis
    import matplotlib.dates as mdates
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('test_output/reset_improvement_visualization.png', dpi=100)
    print(f"\n📊 Visualization saved to test_output/reset_improvement_visualization.png")
    
    # Verify the key behavior
    print("\n" + "="*70)
    print("KEY IMPROVEMENT:")
    print("="*70)
    print("✅ Old logic: Would reset after 30 days even with rejected data")
    print("✅ New logic: No reset because rejected data shows user activity")
    print("✅ Result: Better continuity and fewer false resets")
    
    # Check that no reset occurred
    reset_count = sum(1 for r in results if r['was_reset'])
    print(f"\nResets in new implementation: {reset_count}")
    assert reset_count == 0, "Should have no resets with rejected data present"
    
    return results


if __name__ == "__main__":
    results = simulate_user_0040872d_pattern()
    print("\n✅ Test completed successfully!")