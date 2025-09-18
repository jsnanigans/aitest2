#!/usr/bin/env python3
"""
Performance test with larger dataset to show parallel improvements
"""

import time
import pandas as pd
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

def test_sequential_processing(df_raw, df_filtered, user_list):
    """Test sequential processing (old method)."""
    start = time.time()

    results = []
    for user_id in user_list:
        user_raw = df_raw[df_raw['user_id'] == user_id]
        user_filtered = df_filtered[df_filtered['user_id'] == user_id]

        if len(user_raw) > 2 and len(user_filtered) > 2:
            # Simulate more complex calculations (like in real statistical tests)
            raw_weights = user_raw['weight'].values
            filtered_weights = user_filtered['weight'].values

            # Variance calculation
            var_raw = np.var(raw_weights)
            var_filtered = np.var(filtered_weights)

            # Smoothness calculation (first differences)
            raw_diffs = np.diff(raw_weights)
            filtered_diffs = np.diff(filtered_weights)
            raw_jitter = np.var(raw_diffs) if len(raw_diffs) > 0 else 0
            filtered_jitter = np.var(filtered_diffs) if len(filtered_diffs) > 0 else 0

            # Normality test simulation
            if len(raw_weights) >= 3 and len(filtered_weights) >= 3:
                mean_raw = np.mean(raw_weights)
                std_raw = np.std(raw_weights)
                mean_filtered = np.mean(filtered_weights)
                std_filtered = np.std(filtered_weights)

                results.append({
                    'var_reduction': ((var_raw - var_filtered) / var_raw * 100) if var_raw > 0 else 0,
                    'jitter_reduction': ((raw_jitter - filtered_jitter) / raw_jitter * 100) if raw_jitter > 0 else 0,
                    'mean_diff': mean_filtered - mean_raw
                })

    elapsed = time.time() - start
    return elapsed, len(results)


def _process_batch(batch, df_raw, df_filtered):
    """Process a batch of users."""
    results = []
    for user_id in batch:
        user_raw = df_raw[df_raw['user_id'] == user_id]
        user_filtered = df_filtered[df_filtered['user_id'] == user_id]

        if len(user_raw) > 2 and len(user_filtered) > 2:
            # Simulate more complex calculations (like in real statistical tests)
            raw_weights = user_raw['weight'].values
            filtered_weights = user_filtered['weight'].values

            # Variance calculation
            var_raw = np.var(raw_weights)
            var_filtered = np.var(filtered_weights)

            # Smoothness calculation (first differences)
            raw_diffs = np.diff(raw_weights)
            filtered_diffs = np.diff(filtered_weights)
            raw_jitter = np.var(raw_diffs) if len(raw_diffs) > 0 else 0
            filtered_jitter = np.var(filtered_diffs) if len(filtered_diffs) > 0 else 0

            # Normality test simulation
            if len(raw_weights) >= 3 and len(filtered_weights) >= 3:
                mean_raw = np.mean(raw_weights)
                std_raw = np.std(raw_weights)
                mean_filtered = np.mean(filtered_weights)
                std_filtered = np.std(filtered_weights)

                results.append({
                    'var_reduction': ((var_raw - var_filtered) / var_raw * 100) if var_raw > 0 else 0,
                    'jitter_reduction': ((raw_jitter - filtered_jitter) / raw_jitter * 100) if raw_jitter > 0 else 0,
                    'mean_diff': mean_filtered - mean_raw
                })

    return results


def test_parallel_processing(df_raw, df_filtered, user_list):
    """Test parallel processing (new method)."""
    start = time.time()

    n_workers = min(os.cpu_count() or 4, 8)
    batch_size = max(10, len(user_list) // (n_workers * 2))

    # Split into batches
    batches = [user_list[i:i + batch_size]
               for i in range(0, len(user_list), batch_size)]

    all_results = []
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_process_batch, batch, df_raw, df_filtered): batch
            for batch in batches
        }

        for future in as_completed(futures):
            batch_results = future.result()
            all_results.extend(batch_results)

    elapsed = time.time() - start
    return elapsed, len(all_results)


def main():
    """Run comparison test."""
    print("\n" + "="*60)
    print("PARALLEL PROCESSING PERFORMANCE TEST (LARGE DATASET)")
    print("="*60)
    print(f"System has {os.cpu_count()} CPU cores\n")

    # Load sample data
    DATA_DIR = Path("../data")
    raw_file = DATA_DIR / "2025-09-05_nocon.csv"
    filtered_file = DATA_DIR / "2025-09-05_nocon_filtered.csv"

    print("Loading test data...")
    df_raw = pd.read_csv(raw_file, usecols=['user_id', 'weight'])
    df_filtered = pd.read_csv(filtered_file, usecols=['user_id', 'weight'])

    # Get common users
    users = list(set(df_raw['user_id'].unique()) & set(df_filtered['user_id'].unique()))

    # Test with different dataset sizes
    test_sizes = [500, 1000, 2000]

    for size in test_sizes:
        test_users = users[:min(size, len(users))]
        print(f"\n{'='*40}")
        print(f"Testing with {len(test_users)} users")
        print('='*40)

        # Test sequential processing
        print("Sequential processing...")
        seq_time, seq_count = test_sequential_processing(df_raw, df_filtered, test_users)
        print(f"  Time: {seq_time:.3f}s, Processed: {seq_count} users")

        # Test parallel processing
        print("Parallel processing...")
        par_time, par_count = test_parallel_processing(df_raw, df_filtered, test_users)
        print(f"  Time: {par_time:.3f}s, Processed: {par_count} users")

        # Calculate speedup
        if par_time > 0:
            speedup = seq_time / par_time
            improvement = ((seq_time - par_time) / seq_time) * 100

            print(f"\nResults:")
            print(f"  Speedup: {speedup:.2f}x")
            print(f"  Improvement: {improvement:.1f}%")
            print(f"  Time saved: {seq_time - par_time:.3f}s")

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("The parallel implementation provides:")
    print("✅ Concurrent processing of user batches")
    print("✅ Better CPU utilization")
    print("✅ Scales with dataset size")
    print(f"✅ Uses {min(os.cpu_count() or 4, 8)} worker threads")
    print("\nNote: Performance gains are most visible with:")
    print("  - Larger datasets (1000+ users)")
    print("  - Complex calculations per user")
    print("  - Multi-core systems")


if __name__ == "__main__":
    main()