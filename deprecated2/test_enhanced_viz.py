#!/usr/bin/env python3
"""Test enhanced visualization with sample data."""

import json
from pathlib import Path
from datetime import datetime, timedelta
import logging
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.visualization.enhanced_dashboard import EnhancedDashboard, create_enhanced_dashboards

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_recent_results():
    """Load the most recent results file."""
    output_dir = Path("output/results")
    
    # Find the most recent all_results file
    all_results_files = list(output_dir.glob("all_results_*.json"))
    if not all_results_files:
        logger.error("No all_results files found")
        return None
    
    # Sort by modification time and get the most recent
    latest_file = max(all_results_files, key=lambda p: p.stat().st_mtime)
    logger.info(f"Loading results from: {latest_file}")
    
    with open(latest_file, 'r') as f:
        return json.load(f)


def test_single_user_dashboard():
    """Test creating a dashboard for a single user."""
    results = load_recent_results()
    if not results:
        logger.error("Could not load results")
        return
    
    # Get first user with sufficient data - check stats instead of total_readings
    for user_id, user_data in results.items():
        total_readings = user_data.get('stats', {}).get('total_readings', 0)
        if total_readings > 10:
            logger.info(f"Creating enhanced dashboard for user: {user_id[:12]}...")
            
            # Create dashboard
            dashboard = EnhancedDashboard(user_id, user_data)
            output_path = dashboard.create_dashboard()
            
            if output_path:
                logger.info(f"Dashboard created successfully: {output_path}")
                
                # Print some stats about the user
                logger.info(f"User stats:")
                logger.info(f"  - Total readings: {total_readings}")
                logger.info(f"  - Outliers: {user_data.get('outliers', 0)}")
                logger.info(f"  - Baseline weight: {user_data.get('baseline_weight', 'N/A')}")
                logger.info(f"  - Current weight: {user_data.get('time_series', [{}])[-1].get('weight', 'N/A')}")
                
                # Check for rejected measurements
                rejected_count = 0
                if 'all_readings_with_validation' in user_data:
                    for reading in user_data['all_readings_with_validation']:
                        if not reading.get('validated', False):
                            rejected_count += 1
                    logger.info(f"  - Rejected measurements: {rejected_count}")
                
                # Check for Kalman data
                if 'kalman_time_series' in user_data:
                    logger.info(f"  - Kalman time series: {len(user_data['kalman_time_series'])} points")
                
                # Check for baseline history
                if 'baseline_history' in user_data:
                    logger.info(f"  - Baseline history: {len(user_data['baseline_history'])} baselines")
            else:
                logger.warning("Dashboard creation returned None")
            
            break
    else:
        logger.warning("No users with sufficient data found")


def test_multiple_dashboards():
    """Test creating dashboards for multiple users."""
    results = load_recent_results()
    if not results:
        logger.error("Could not load results")
        return
    
    # Filter users with sufficient data
    valid_users = {
        user_id: data 
        for user_id, data in results.items() 
        if data.get('stats', {}).get('total_readings', 0) > 10
    }
    
    logger.info(f"Found {len(valid_users)} users with sufficient data")
    
    if valid_users:
        # Create dashboards for up to 5 users
        dashboard_paths = create_enhanced_dashboards(valid_users, max_users=5)
        
        logger.info(f"Created {len(dashboard_paths)} dashboards:")
        for user_id, path in dashboard_paths.items():
            logger.info(f"  - {user_id[:12]}: {path}")


if __name__ == "__main__":
    logger.info("Testing enhanced dashboard visualization...")
    
    # Test single user first
    logger.info("\n=== Testing single user dashboard ===")
    test_single_user_dashboard()
    
    # Then test multiple users
    logger.info("\n=== Testing multiple user dashboards ===")
    test_multiple_dashboards()
    
    logger.info("\nTest complete!")