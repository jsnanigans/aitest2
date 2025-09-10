"""
Demo script to show rejection visualization with various rejection types
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from datetime import datetime, timedelta
import json
import tempfile
import os

import processor
import processor_database
import visualization


def create_demo_data():
    """Create demo data with various rejection types."""
    
    base_time = datetime.now() - timedelta(days=60)
    user_id = "demo_user_rejections"
    
    db = processor_database.ProcessorDatabase()
    
    processing_config = {
        'min_weight': 30,
        'max_weight': 400,
        'extreme_threshold': 0.15,
        'reset_gap_days': 30,
        'physiological': {
            'enable_physiological_limits': True,
            'max_change_1h_percent': 0.02,
            'max_change_1h_absolute': 3.0,
            'max_change_6h_percent': 0.025,
            'max_change_6h_absolute': 4.0,
            'max_change_24h_percent': 0.035,
            'max_change_24h_absolute': 5.0,
            'max_sustained_daily': 1.5,
            'session_timeout_minutes': 5,
            'session_variance_threshold': 5.0,
            'limit_tolerance': 0.10,
            'sustained_tolerance': 0.25
        }
    }
    
    kalman_config = {
        'process_noise': 0.01,
        'measurement_noise': 1.0,
        'initial_trend': 0.0,
        'trend_process_noise': 0.001,
        'innovation_threshold': 3.0,
        'min_measurements': 10,
        'transition_covariance_weight': 0.05,
        'transition_covariance_trend': 0.0005,
        'observation_covariance': 1.0,
        'adaptation': {
            'enabled': True,
            'window_size': 20,
            'innovation_alpha': 0.3,
            'noise_floor': 0.5,
            'noise_ceiling': 5.0
        }
    }
    
    measurements = [
        (base_time, 80.0, "scale", True, None),
        (base_time + timedelta(days=1), 80.2, "scale", True, None),
        (base_time + timedelta(days=2), 80.1, "scale", True, None),
        (base_time + timedelta(days=3), 80.3, "scale", True, None),
        (base_time + timedelta(days=4), 80.0, "scale", True, None),
        (base_time + timedelta(days=5), 79.8, "scale", True, None),
        (base_time + timedelta(days=6), 80.1, "scale", True, None),
        (base_time + timedelta(days=7), 80.2, "scale", True, None),
        (base_time + timedelta(days=8), 79.9, "scale", True, None),
        (base_time + timedelta(days=9), 80.0, "scale", True, None),
        (base_time + timedelta(days=10), 80.1, "scale", True, None),
        
        (base_time + timedelta(days=11), 25.0, "scale", False, "Weight 25.0kg outside bounds [30, 400]"),
        
        (base_time + timedelta(days=12), 80.3, "scale", True, None),
        
        (base_time + timedelta(days=13), 95.0, "scale", False, "Extreme deviation: 18.1%"),
        
        (base_time + timedelta(days=14), 80.5, "scale", True, None),
        
        (base_time + timedelta(days=15, hours=0), 80.4, "scale", True, None),
        (base_time + timedelta(days=15, hours=0.5), 85.0, "scale", False, "Change of 4.6kg in 0.5h exceeds hydration/bathroom limit"),
        
        (base_time + timedelta(days=16), 80.6, "scale", True, None),
        
        (base_time + timedelta(days=17, hours=0), 80.5, "scale", True, None),
        (base_time + timedelta(days=17, hours=4), 86.0, "scale", False, "Change of 5.5kg in 4.0h exceeds meals+hydration limit"),
        
        (base_time + timedelta(days=18), 80.7, "scale", True, None),
        
        (base_time + timedelta(days=19), 87.0, "scale", False, "Change of 6.3kg in 24.0h exceeds daily fluctuation limit"),
        
        (base_time + timedelta(days=20), 80.8, "scale", True, None),
        (base_time + timedelta(days=21), 80.9, "scale", True, None),
        (base_time + timedelta(days=22), 81.0, "scale", True, None),
        
        (base_time + timedelta(days=25), 90.0, "scale", False, "Change of 9.0kg in 72.0h exceeds sustained (1.5kg/day) limit"),
        
        (base_time + timedelta(days=26), 81.1, "scale", True, None),
        (base_time + timedelta(days=27), 81.0, "scale", True, None),
        
        (base_time + timedelta(days=28, minutes=0), 81.2, "scale", True, None),
        (base_time + timedelta(days=28, minutes=3), 88.0, "scale", False, "Session variance 6.8kg exceeds threshold (likely different user)"),
        
        (base_time + timedelta(days=29), 81.3, "scale", True, None),
        (base_time + timedelta(days=30), 81.2, "scale", True, None),
        
        (base_time + timedelta(days=31), 92.0, "scale", False, "Extreme deviation: 13.5%"),
        (base_time + timedelta(days=32), 93.0, "scale", False, "Extreme deviation: 14.8%"),
        (base_time + timedelta(days=33), 94.0, "scale", False, "Extreme deviation: 16.0%"),
        
        (base_time + timedelta(days=34), 81.4, "scale", True, None),
        (base_time + timedelta(days=35), 81.5, "scale", True, None),
        
        (base_time + timedelta(days=70), 82.0, "scale", True, None),
        (base_time + timedelta(days=71), 82.1, "scale", True, None),
        (base_time + timedelta(days=72), 82.0, "scale", True, None),
    ]
    
    results = []
    
    for timestamp, weight, source, should_accept, expected_reason in measurements:
        result = processor.WeightProcessor.process_weight(
            user_id=user_id,
            weight=weight,
            timestamp=timestamp,
            source=source,
            processing_config=processing_config,
            kalman_config=kalman_config,
            db=db
        )
        
        if result:
            if not should_accept and expected_reason:
                result['reason'] = expected_reason
            results.append(result)
    
    return user_id, results


def main():
    """Generate demo visualization with rejections."""
    
    print("\n=== Generating Rejection Visualization Demo ===\n")
    
    user_id, results = create_demo_data()
    
    print(f"Created {len(results)} measurements")
    accepted = [r for r in results if r.get('accepted', False)]
    rejected = [r for r in results if not r.get('accepted', False)]
    print(f"  Accepted: {len(accepted)}")
    print(f"  Rejected: {len(rejected)}")
    
    if rejected:
        print("\nRejection reasons:")
        reasons = {}
        for r in rejected:
            reason = r.get('reason', 'Unknown')
            reasons[reason] = reasons.get(reason, 0) + 1
        
        for reason, count in sorted(reasons.items(), key=lambda x: x[1], reverse=True):
            print(f"  {count}x: {reason}")
    
    output_dir = "output/rejection_demo"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    viz_config = {
        'dashboard_dpi': 100,
        'cropped_months': 2
    }
    
    print(f"\nGenerating visualization...")
    output_file = visualization.create_dashboard(
        user_id=user_id,
        results=results,
        output_dir=output_dir,
        viz_config=viz_config
    )
    
    if output_file:
        print(f"✓ Visualization saved to: {output_file}")
        
        file_size = os.path.getsize(output_file) / 1024
        print(f"  File size: {file_size:.1f} KB")
    else:
        print("✗ Failed to generate visualization")
    
    print("\n✅ Demo complete!\n")


if __name__ == "__main__":
    main()