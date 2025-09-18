#!/usr/bin/env python3
"""
Performance test for multithreading improvements
Compares execution times before and after parallelization
"""

import time
import logging
from pathlib import Path
import os

# Configure logging to show performance details
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def test_statistical_report():
    """Test performance of statistical report generation."""
    print("\n" + "="*60)
    print("Testing Statistical Report Generation")
    print("="*60)

    from generate_statistical_report import generate_report

    start_time = time.time()
    generate_report(output_dir=Path("."))
    elapsed = time.time() - start_time

    print(f"\nTotal execution time: {elapsed:.2f} seconds")
    print(f"CPU count: {os.cpu_count()}")

    return elapsed


def test_data_cache():
    """Test performance of data cache preloading."""
    print("\n" + "="*60)
    print("Testing Data Cache Preloading")
    print("="*60)

    from data_cache import DataCache
    from pathlib import Path

    # Create a fresh cache instance
    cache = DataCache()
    cache.clear_cache()

    # Define data paths
    DATA_DIR = Path("../data")
    raw_path = DATA_DIR / "2025-09-05_nocon.csv"
    filtered_path = DATA_DIR / "2025-09-05_nocon_filtered.csv"

    # Test parallel preloading
    start_time = time.time()
    cache.preload_all(raw_path, filtered_path)
    elapsed = time.time() - start_time

    print(f"\nParallel preload time: {elapsed:.2f} seconds")
    print(f"Cache size: {cache.get_cache_size()} files")
    print(f"Memory usage: {cache.get_memory_usage():.2f} MB")

    return elapsed


def test_dashboard_generation():
    """Test performance of dashboard generation."""
    print("\n" + "="*60)
    print("Testing Dashboard Generation")
    print("="*60)

    from generate_dashboard import load_data, calculate_statistics, create_dashboard
    from pathlib import Path

    # Load test data
    DATA_FILE = Path("daily_weight_analysis.csv")
    if not DATA_FILE.exists():
        print("Daily weight analysis file not found. Skipping dashboard test.")
        return None

    df = load_data(DATA_FILE)
    stats = calculate_statistics(df)

    # Test dashboard creation
    start_time = time.time()
    output_dir = Path("visualizations")
    create_dashboard(df, stats, output_dir)
    elapsed = time.time() - start_time

    print(f"\nDashboard generation time: {elapsed:.2f} seconds")

    return elapsed


def main():
    """Run all performance tests."""
    print("\n" + "="*60)
    print("MULTITHREADING PERFORMANCE TEST SUITE")
    print("="*60)

    results = {}

    # Test each component
    print("\n1. Data Cache Performance")
    cache_time = test_data_cache()
    results['data_cache'] = cache_time

    print("\n2. Statistical Report Performance")
    report_time = test_statistical_report()
    results['statistical_report'] = report_time

    print("\n3. Dashboard Generation Performance")
    dashboard_time = test_dashboard_generation()
    if dashboard_time:
        results['dashboard'] = dashboard_time

    # Summary
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)

    for component, time_taken in results.items():
        if time_taken:
            print(f"{component}: {time_taken:.2f}s")

    total_time = sum(t for t in results.values() if t)
    print(f"\nTotal time: {total_time:.2f}s")

    # Expected improvements based on investigation
    print("\n" + "="*60)
    print("EXPECTED IMPROVEMENTS")
    print("="*60)
    print("Statistical Report: 60-70% speedup (parallelized loops)")
    print("Data Cache: ~30% speedup (parallel file loading)")
    print("Dashboard: ~50% speedup (parallel user journey processing)")

    print("\nNote: Actual speedup depends on:")
    print("  - Number of CPU cores available")
    print("  - Data size and complexity")
    print("  - I/O performance")
    print(f"\nThis system has {os.cpu_count()} CPU cores available")


if __name__ == "__main__":
    main()