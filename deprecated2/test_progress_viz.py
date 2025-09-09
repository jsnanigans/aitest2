#!/usr/bin/env python3
"""Test script to verify progress display and visualization creation."""

import csv
import json
from pathlib import Path
import sys

def test_progress_display():
    """Test the progress display with sample data."""
    from src.core.progress import ProgressIndicator, ProcessingStatus
    import time
    
    ProcessingStatus.section("Testing Progress Display")
    ProcessingStatus.info("Testing with sample data processing")
    
    # Test with known total
    progress = ProgressIndicator(total_items=1000, show_eta=True)
    progress.start("Processing sample data")
    
    # Simulate processing with user updates
    for i in range(1, 1001):
        # Simulate user changes every 50 rows
        user_num = i // 50 + 1
        progress.update(i, {
            'rows_processed': i,
            'users_processed': user_num,
            'current_user': f'user_{user_num:03d}',
            'accepted': int(i * 0.87),  # 87% acceptance rate
            'rejected': int(i * 0.13)
        })
        
        # Small delay to see progress
        if i % 100 == 0:
            time.sleep(0.1)
    
    progress.finish("Test complete")
    
    ProcessingStatus.success("Progress display test passed!")
    return True

def test_visualization_setup():
    """Test that visualization module is properly set up."""
    from src.core.progress import ProcessingStatus
    ProcessingStatus.section("Testing Visualization Setup")
    
    try:
        from src.visualization import UserDashboard
        ProcessingStatus.success("UserDashboard imported successfully")
        
        # Check matplotlib backend
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        ProcessingStatus.success(f"Matplotlib backend: {matplotlib.get_backend()}")
        
        # Create test data
        test_results = {
            'stats': {
                'total_readings': 100,
                'accepted_readings': 87,
                'rejected_readings': 13,
                'outliers_by_type': {
                    'physiological': 5,
                    'rate_of_change': 3,
                    'statistical': 5
                }
            },
            'time_series': [
                {
                    'date': f'2024-01-{i:02d}',
                    'weight': 70 + (i % 5) * 0.5,
                    'confidence': 0.95 - (i % 10) * 0.02,
                    'source': 'scale'
                }
                for i in range(1, 21)
            ],
            'baselines': [
                {
                    'baseline_weight': 70.0,
                    'confidence': 0.95,
                    'method': 'initial',
                    'readings_count': 5,
                    'date_range': ['2024-01-01', '2024-01-07']
                }
            ]
        }
        
        # Test dashboard creation
        output_dir = Path('./test_output')
        output_dir.mkdir(exist_ok=True)
        
        dashboard = UserDashboard('test_user', test_results, str(output_dir))
        path = dashboard.create_dashboard()
        
        if path and Path(path).exists():
            ProcessingStatus.success(f"Test dashboard created: {path}")
            # Clean up
            Path(path).unlink()
            output_dir.rmdir()
            return True
        else:
            ProcessingStatus.error("Dashboard creation failed")
            return False
            
    except ImportError as e:
        ProcessingStatus.error(f"Import error: {e}")
        ProcessingStatus.warning("You may need to install matplotlib: uv pip install matplotlib")
        return False
    except Exception as e:
        ProcessingStatus.error(f"Unexpected error: {e}")
        return False

def check_config():
    """Check configuration file."""
    from src.core.progress import ProcessingStatus
    ProcessingStatus.section("Checking Configuration")
    
    try:
        from src.core.config_loader import load_config
        config = load_config()
        
        # Check key settings
        source_file = config.get('source_file')
        if source_file:
            ProcessingStatus.success(f"Source file: {source_file}")
            if Path(source_file).exists():
                # Count lines
                with open(source_file, 'r') as f:
                    line_count = sum(1 for _ in f)
                ProcessingStatus.success(f"File exists with {line_count:,} lines")
            else:
                ProcessingStatus.warning("Source file not found")
        
        # Check visualization settings
        viz_config = config.get('visualization', {})
        ProcessingStatus.info(f"Visualization enabled: {viz_config.get('enabled', True)}")
        ProcessingStatus.info(f"Max users for viz: {viz_config.get('max_users', 10)}")
        ProcessingStatus.info(f"Viz output dir: {viz_config.get('output_dir', 'output/visualizations')}")
        
        return True
        
    except Exception as e:
        ProcessingStatus.error(f"Config error: {e}")
        return False

def main():
    """Run all tests."""
    from src.core.progress import ProcessingStatus
    
    print("\n" + "="*60)
    print("PROGRESS & VISUALIZATION TEST SUITE")
    print("="*60)
    
    tests = [
        ("Configuration", check_config),
        ("Progress Display", test_progress_display),
        ("Visualization Setup", test_visualization_setup)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            ProcessingStatus.error(f"{name} failed with exception: {e}")
            results.append((name, False))
    
    # Summary
    ProcessingStatus.section("Test Results Summary")
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        if result:
            print(f"  ✓ {name}: PASSED")
        else:
            print(f"  ✗ {name}: FAILED")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        ProcessingStatus.success("All tests passed! ✨")
        return 0
    else:
        ProcessingStatus.warning(f"{total - passed} test(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())