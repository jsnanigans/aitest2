#!/usr/bin/env python
"""
Test script for filtering effectiveness analysis.
Creates sample data and runs a quick test of the analysis system.
"""

import sys
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_filtering_analysis import FilteringAnalysisRunner
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_data():
    """Create synthetic test data for analysis."""
    np.random.seed(42)

    users = ['user001', 'user002', 'user003']
    data = []

    for user_id in users:
        # Base weight for this user
        base_weight = np.random.uniform(70, 100)

        # Generate 50 measurements over 100 days
        for i in range(50):
            timestamp = datetime.now() - timedelta(days=100-i*2)

            # Add realistic weight variation
            weight = base_weight - (i * 0.05)  # Gradual weight loss
            weight += np.random.normal(0, 0.5)  # Daily variation

            # Add some outliers
            if np.random.random() < 0.1:  # 10% outlier rate
                weight += np.random.choice([-5, 5]) * np.random.random()

            # Determine source
            source = np.random.choice([
                'patient-device',
                'care-team-upload',
                'questionnaire',
                'iglucose.com'
            ], p=[0.6, 0.2, 0.15, 0.05])

            data.append({
                'user_id': user_id,
                'effectiveDateTime': timestamp.isoformat(),
                'weight': weight,
                'source': source,
                'unit': 'kg'
            })

    df = pd.DataFrame(data)
    return df


def main():
    """Run test analysis."""
    logger.info("Creating test data...")

    # Create test data
    test_df = create_test_data()

    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        test_df.to_csv(f, index=False)
        csv_path = f.name

    logger.info(f"Test data saved to {csv_path}")
    logger.info(f"Data shape: {test_df.shape}")
    logger.info(f"Users: {test_df['user_id'].nunique()}")

    try:
        # Initialize runner with test config
        runner = FilteringAnalysisRunner()

        # Override config for testing
        runner.config['analysis']['max_users'] = 5
        runner.config['analysis']['min_measurements'] = 10
        runner.config['analysis']['output_dir'] = 'reports/test_visualizations'

        # Load data
        logger.info("\nLoading test data...")
        raw_data = runner.load_raw_data(csv_path)

        if not raw_data:
            logger.error("Failed to load test data")
            return 1

        logger.info(f"Loaded {len(raw_data)} users")

        # Process through filtering
        logger.info("\nProcessing through filtering pipeline...")
        filtered_data = runner.process_filtered_data(raw_data)

        logger.info(f"Filtered data for {len(filtered_data)} users")

        # Run analysis
        logger.info("\nRunning analysis...")
        metrics = runner.run_analysis(raw_data, filtered_data)

        # Generate report
        logger.info("\nGenerating report...")
        report_path = runner.generate_report(metrics)

        # Save metrics
        json_path = runner.save_metrics_json(metrics)

        logger.info("\n" + "="*60)
        logger.info("TEST COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        logger.info(f"Report: {report_path}")
        logger.info(f"Metrics: {json_path}")
        logger.info(f"Visualizations: {runner.config['analysis']['output_dir']}")

        # Print summary statistics
        if 'aggregate' in metrics:
            agg = metrics['aggregate']
            logger.info("\nSummary Statistics:")
            logger.info(f"  - Total users: {agg.get('total_users', 0)}")
            logger.info(f"  - Avg removal rate: {agg.get('avg_removal_rate', 0):.1%}")
            logger.info(f"  - Avg outlier rate: {agg.get('outlier_summary', {}).get('avg_outlier_rate', 0):.1%}")

        return 0

    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        # Clean up temp file
        Path(csv_path).unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())