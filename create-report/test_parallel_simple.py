#!/usr/bin/env python3
"""
Simple test to verify parallel processing improvements
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
    for user_id in user_list[:100]:  # Test first 100 users
        user_raw = df_raw[df_raw['user_id'] == user_id]['weight'].values
        user_filtered = df_filtered[df_filtered['user_id'] == user_id]['weight'].values

        if len(user_raw) > 1 and len(user_filtered) > 1:
            var_raw = np.var(user_raw)
            var_filtered = np.var(user_filtered)
            if var_raw > 0:
                reduction = ((var_raw - var_filtered) / var_raw) * 100
                results.append(reduction)

    elapsed = time.time() - start
    return elapsed, len(results)


def _process_batch(batch, df_raw, df_filtered):
    """Process a batch of users."""
    results = []
    for user_id in batch:
        user_raw = df_raw[df_raw['user_id'] == user_id]['weight'].values
        user_filtered = df_filtered[df_filtered['user_id'] == user_id]['weight'].values

        if len(user_raw) > 1 and len(user_filtered) > 1:
            var_raw = np.var(user_raw)
            var_filtered = np.var(user_filtered)
            if var_raw > 0:
                reduction = ((var_raw - var_filtered) / var_raw) * 100
                results.append(reduction)
    return results


def test_parallel_processing(df_raw, df_filtered, user_list):
    """Test parallel processing (new method)."""
    start = time.time()

    test_users = user_list[:100]  # Test first 100 users
    n_workers = min(os.cpu_count() or 4, 8)
    batch_size = max(10, len(test_users) // n_workers)

    # Split into batches
    batches = [test_users[i:i + batch_size]
               for i in range(0, len(test_users), batch_size)]

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
    print("PARALLEL PROCESSING PERFORMANCE TEST")
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
    print(f"Found {len(users)} users for testing\n")

    # Test sequential processing
    print("Testing SEQUENTIAL processing (100 users)...")
    seq_time, seq_count = test_sequential_processing(df_raw, df_filtered, users)
    print(f"  Time: {seq_time:.3f}s")
    print(f"  Processed: {seq_count} users\n")

    # Test parallel processing
    print("Testing PARALLEL processing (100 users)...")
    par_time, par_count = test_parallel_processing(df_raw, df_filtered, users)
    print(f"  Time: {par_time:.3f}s")
    print(f"  Processed: {par_count} users\n")

    # Calculate speedup
    if par_time > 0:
        speedup = seq_time / par_time
        improvement = ((seq_time - par_time) / seq_time) * 100

        print("="*60)
        print("RESULTS:")
        print(f"  Sequential: {seq_time:.3f}s")
        print(f"  Parallel:   {par_time:.3f}s")
        print(f"  Speedup:    {speedup:.2f}x")
        print(f"  Improvement: {improvement:.1f}%")
        print("="*60)

        if speedup > 1.5:
            print("\n✅ SUCCESS: Parallel processing shows significant improvement!")
        elif speedup > 1.2:
            print("\n✓ GOOD: Parallel processing shows moderate improvement")
        else:
            print("\n⚠️ NOTE: Limited improvement - may need larger dataset")


if __name__ == "__main__":
    main()