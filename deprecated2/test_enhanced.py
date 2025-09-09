#!/usr/bin/env python3
"""
Test the enhanced implementation with logging and visualization.
"""

import numpy as np
from datetime import datetime, timedelta
import json
from pathlib import Path

from src.core.logger_config import setup_logging
from src.core.types import WeightMeasurement
from src.processing.weight_pipeline import WeightProcessingPipeline
from src.processing.user_processor import UserProcessor
from src.visualization import UserDashboard

# Setup enhanced logging
import logging
setup_logging()
logging.getLogger().setLevel(logging.DEBUG)


def generate_test_user_data(user_id: str, days: int = 60):
    """Generate realistic test data for a user."""
    measurements = []
    base_weight = 70.0 + np.random.normal(0, 5)  # Random starting weight
    trend = np.random.choice([-0.02, 0, 0.02])  # Losing, stable, or gaining
    
    for day in range(days):
        date = datetime.now() - timedelta(days=days-day)
        
        # Base weight with trend
        weight = base_weight + trend * day
        
        # Add daily variation
        daily_var = np.random.normal(0, 0.3)
        
        # Weekly pattern (higher on weekends)
        if date.weekday() in [5, 6]:
            weekly_effect = 0.5
        else:
            weekly_effect = 0
            
        # Add some outliers
        if np.random.random() < 0.05:  # 5% outlier rate
            outlier_effect = np.random.choice([-5, 5, 10])
        else:
            outlier_effect = 0
            
        final_weight = weight + daily_var + weekly_effect + outlier_effect
        
        measurements.append({
            'user_id': user_id,
            'weight': str(max(30, min(200, final_weight))),  # Convert to string like CSV
            'date': date.isoformat(),
            'source': np.random.choice(['patient-device', 'patient-upload', 'care-team-upload'])
        })
        
    return measurements


def test_single_user_pipeline():
    """Test the pipeline with a single user."""
    print("\n" + "="*60)
    print("TESTING SINGLE USER WITH ENHANCED FEATURES")
    print("="*60)
    
    # Generate test data
    user_id = "test_user_001"
    test_data = generate_test_user_data(user_id, days=45)
    
    # Create processor
    processor = UserProcessor(user_id, {
        'min_init_readings': 3,
        'validation_gamma': 3.0,
        'layer1': {'mad_threshold': 3.0},
        'kalman': {'process_noise_weight': 0.5}
    })
    
    # Process readings
    print(f"\nProcessing {len(test_data)} readings...")
    for row in test_data:
        result = processor.process_reading(row)
        
    # Get results
    results = processor.get_results()
    
    # Print summary
    print(f"\nUser: {user_id}")
    print(f"Initialized: {results['initialized']}")
    
    if results['initialized']:
        stats = results['stats']
        print(f"Total readings: {stats['total_readings']}")
        print(f"Accepted: {stats['accepted_readings']}")
        print(f"Rejected: {stats['rejected_readings']}")
        print(f"Acceptance rate: {stats.get('acceptance_rate', 0):.1%}")
        
        if 'baseline' in results:
            print(f"\nBaseline: {results['baseline']['weight']:.1f} kg")
            print(f"Confidence: {results['baseline']['confidence']}")
            
        if 'current_state' in results:
            current = results['current_state']
            print(f"\nCurrent weight: {current['weight']:.1f} kg")
            print(f"Trend: {current['trend_kg_per_week']:.2f} kg/week")
            
        # Outlier breakdown
        if stats.get('outliers_by_type'):
            print("\nOutlier types detected:")
            for outlier_type, count in stats['outliers_by_type'].items():
                print(f"  {outlier_type}: {count}")
    
    return results


def test_visualization():
    """Test visualization creation."""
    print("\n" + "="*60)
    print("TESTING VISUALIZATION")
    print("="*60)
    
    # Generate test data for multiple users
    results = {}
    for i in range(3):
        user_id = f"viz_test_{i:03d}"
        print(f"\nGenerating data for {user_id}...")
        
        # Process user
        processor = UserProcessor(user_id, {})
        test_data = generate_test_user_data(user_id, days=30)
        
        for row in test_data:
            processor.process_reading(row)
            
        results[user_id] = processor.get_results()
    
    # Create visualizations
    print("\nCreating dashboards...")
    output_dir = Path("output/test_visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for user_id, user_results in results.items():
        if user_results['initialized']:
            try:
                dashboard = UserDashboard(user_id, user_results, str(output_dir))
                path = dashboard.create_dashboard()
                if path:
                    print(f"✓ Created dashboard for {user_id}: {path}")
                else:
                    print(f"✗ Failed to create dashboard for {user_id}")
            except Exception as e:
                print(f"✗ Error for {user_id}: {e}")
    
    print(f"\nVisualizations saved to: {output_dir}")


def test_logging_levels():
    """Test different logging levels."""
    print("\n" + "="*60)
    print("TESTING LOGGING LEVELS")
    print("="*60)
    
    import logging
    logger = logging.getLogger("test_logger")
    
    print("\nLogging at different levels:")
    logger.debug("This is a DEBUG message (detailed diagnostic)")
    logger.info("This is an INFO message (general information)")
    logger.warning("This is a WARNING message (something unexpected)")
    logger.error("This is an ERROR message (serious problem)")
    
    print("\nCheck output/test_logs/ for detailed logs")


def main():
    """Run all tests."""
    print("="*60)
    print("ENHANCED WEIGHT PROCESSOR TEST SUITE")
    print("="*60)
    
    # Test logging
    test_logging_levels()
    
    # Test single user processing
    results = test_single_user_pipeline()
    
    # Test visualization
    test_visualization()
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETE")
    print("Check output/ directory for logs and visualizations")
    print("="*60)


if __name__ == "__main__":
    main()