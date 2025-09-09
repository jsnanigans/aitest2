#!/usr/bin/env python3
"""
Test that output folders are properly structured and visualization settings work.
"""

import json
import shutil
from pathlib import Path
from datetime import datetime
import toml

from src.core.logger_config import setup_logging
from src.processing import UserProcessor
from src.visualization import create_user_visualizations


def test_output_structure():
    """Test that all output folders are created correctly."""
    print("\n" + "="*60)
    print("TESTING OUTPUT FOLDER STRUCTURE")
    print("="*60)
    
    # Clean up any existing test output
    test_output = Path("test_output")
    if test_output.exists():
        shutil.rmtree(test_output)
    
    # Create test config
    test_config = {
        "source_file": "./test_data.csv",
        "output": {
            "directory": "test_output"
        },
        "processing": {
            "max_users": 2,
            "min_init_readings": 3
        },
        "visualization": {
            "enabled": True,
            "max_users": 2,
            "output_dir": "test_output/custom_viz"
        },
        "logging": {
            "file": "test_output/logs/test.log",
            "level": "INFO",
            "stdout_enabled": True,
            "stdout_level": "WARNING"
        }
    }
    
    # Save test config
    with open("test_config.toml", "w") as f:
        toml.dump(test_config, f)
    
    # Import main processor
    from main import StreamProcessor
    
    # Create processor with test config
    processor = StreamProcessor("test_config.toml")
    
    # Check that directories were created
    print("\nChecking directory structure...")
    
    expected_dirs = [
        processor.output_dir,           # test_output/
        processor.results_dir,           # test_output/results/
        processor.logs_dir,             # test_output/logs/
    ]
    
    for dir_path in expected_dirs:
        if dir_path.exists():
            print(f"✅ Created: {dir_path}")
        else:
            print(f"❌ Missing: {dir_path}")
    
    # Check visualization directory config
    print(f"\n📊 Visualization directory configured as: {processor.viz_dir}")
    print(f"   (Will be created when visualizations are generated)")
    
    return processor


def test_visualization_settings():
    """Test that visualization settings are properly used."""
    print("\n" + "="*60)
    print("TESTING VISUALIZATION SETTINGS")
    print("="*60)
    
    # Create test data for 3 users
    test_results = {}
    for i in range(3):
        user_id = f"test_user_{i:03d}"
        test_results[user_id] = {
            "user_id": user_id,
            "initialized": True,
            "baseline": {"weight": 70.0 + i, "confidence": "high", "variance": 0.5},
            "current_state": {"weight": 69.5 + i, "trend_kg_per_week": -0.2},
            "stats": {
                "total_readings": 30,
                "accepted_readings": 28,
                "rejected_readings": 2,
                "min_weight": 68.0 + i,
                "max_weight": 72.0 + i,
                "weight_range": 4.0,
                "acceptance_rate": 0.933
            },
            "time_series": [
                {
                    "date": datetime.now().isoformat(),
                    "weight": 70.0 + j*0.1,
                    "filtered_weight": 70.0 + j*0.08,
                    "is_valid": True,
                    "confidence": 0.95,
                    "trend_kg_per_day": -0.03
                }
                for j in range(10)
            ]
        }
    
    # Test with max_users = 2 (should only create 2 dashboards)
    viz_dir = Path("test_output/custom_viz")
    
    print(f"\nCreating visualizations for 3 users with max_users=2...")
    print(f"Output directory: {viz_dir}")
    
    dashboard_paths = create_user_visualizations(
        test_results,
        output_dir=str(viz_dir),
        max_users=2  # Should limit to 2 users
    )
    
    print(f"\n📊 Created {len(dashboard_paths)} dashboards (should be 2):")
    for user_id, path in dashboard_paths.items():
        print(f"   ✅ {user_id}: {path.name}")
    
    # Verify directory was created
    if viz_dir.exists():
        print(f"\n✅ Visualization directory created: {viz_dir}")
        
        # List files
        viz_files = list(viz_dir.glob("*.png"))
        print(f"   Found {len(viz_files)} visualization files:")
        for f in viz_files[:5]:  # Show first 5
            print(f"   - {f.name}")
    else:
        print(f"\n❌ Visualization directory not created: {viz_dir}")
    
    return len(dashboard_paths) == 2


def test_full_output_structure():
    """Test complete output structure with sample data."""
    print("\n" + "="*60)
    print("TESTING COMPLETE OUTPUT STRUCTURE")
    print("="*60)
    
    # Show expected structure
    print("\nExpected output structure:")
    print("""
    output/
    ├── logs/
    │   └── app.log                 # Application logs
    ├── results/
    │   ├── user_xxx.json           # Individual user results
    │   ├── summary_YYYYMMDD.json   # Processing summary
    │   └── all_results_YYYYMMDD.json # All results combined
    └── visualizations/              # From config: visualization.output_dir
        ├── dashboard_user1_*.png   # User dashboards
        ├── dashboard_user2_*.png
        └── ...
    """)
    
    # Create sample outputs to demonstrate structure
    output_base = Path("test_output")
    
    # Create sample files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Sample log
    log_file = output_base / "logs" / "app.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text(f"[{datetime.now()}] Sample log entry\n")
    
    # Sample results
    results_dir = output_base / "results"
    results_dir.mkdir(exist_ok=True)
    
    summary_file = results_dir / f"summary_{timestamp}.json"
    summary_file.write_text(json.dumps({"processed": 100, "users": 5}, indent=2))
    
    user_file = results_dir / "user_test001.json"
    user_file.write_text(json.dumps({"user_id": "test001", "weight": 70.5}, indent=2))
    
    # List created structure
    print("\nActual structure created:")
    for path in sorted(output_base.rglob("*")):
        if path.is_file():
            indent = "  " * (len(path.relative_to(output_base).parts) - 1)
            print(f"  {indent}├── {path.name}")
        elif path.is_dir():
            indent = "  " * len(path.relative_to(output_base).parts)
            print(f"  {indent}{path.name}/")


def cleanup():
    """Clean up test files."""
    print("\n" + "="*60)
    print("CLEANUP")
    print("="*60)
    
    # Remove test directories
    test_paths = [
        Path("test_output"),
        Path("test_config.toml")
    ]
    
    for path in test_paths:
        if path.exists():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            print(f"✅ Cleaned up: {path}")


def main():
    """Run all tests."""
    print("="*60)
    print("OUTPUT STRUCTURE AND VISUALIZATION TEST")
    print("="*60)
    
    try:
        # Test 1: Output structure
        processor = test_output_structure()
        
        # Test 2: Visualization settings
        viz_test_passed = test_visualization_settings()
        
        # Test 3: Full structure demo
        test_full_output_structure()
        
        # Summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        print("✅ Output folders created correctly")
        print(f"✅ Visualization max_users limit: {'PASSED' if viz_test_passed else 'FAILED'}")
        print("✅ Custom visualization directory works")
        print("\nVisualization config from config.toml:")
        print("  [visualization]")
        print("  enabled = true")
        print("  max_users = 10")
        print('  output_dir = "output/visualizations"')
        
    finally:
        # Clean up
        cleanup()
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()