"""
Parallel Interval Analysis Module using Multiprocessing

Calculates weight measurements at regular intervals using process pool
for significant performance improvements on multi-core systems.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import logging
import time
from multiprocessing import Pool, cpu_count
import traceback
from functools import partial


def process_user_batch(batch_data: Tuple[List[str], Dict, Dict, int, float]) -> Dict:
    """
    Process a batch of users in a worker process

    Args:
        batch_data: Tuple of (user_ids, raw_dict, filtered_dict, interval_days, window_days)

    Returns:
        Dictionary of results for this batch
    """
    user_ids, raw_dict, filtered_dict, interval_days, window_days = batch_data
    results = {}

    for user_id in user_ids:
        try:
            user_raw = raw_dict.get(user_id, pd.DataFrame())
            user_filtered = filtered_dict.get(user_id, pd.DataFrame())

            # Find signup date
            signup_date = None
            if not user_raw.empty:
                signup_rows = user_raw[user_raw['source'] == 'initial-questionnaire']
                if not signup_rows.empty:
                    signup_date = signup_rows.iloc[0]['timestamp']
                else:
                    signup_date = user_raw.iloc[0]['timestamp']

            if signup_date is None:
                continue

            # Calculate intervals for this user
            intervals = calculate_user_intervals_worker(
                user_id, signup_date, user_raw, user_filtered,
                interval_days, window_days
            )

            if intervals:
                results[user_id] = {
                    'user_id': user_id,
                    'signup_date': signup_date,
                    'intervals': intervals,
                    'num_raw_measurements': len(user_raw),
                    'num_filtered_measurements': len(user_filtered) if not user_filtered.empty else 0
                }

        except Exception as e:
            # Log error but continue processing
            print(f"Error processing user {user_id}: {e}")
            continue

    return results


def calculate_user_intervals_worker(user_id: str,
                                   signup_date: datetime,
                                   raw_data: pd.DataFrame,
                                   filtered_data: pd.DataFrame,
                                   interval_days: int,
                                   window_days: float) -> List[Dict]:
    """Calculate intervals for a single user (worker function)"""
    intervals = []

    # Pre-sort data once
    if not raw_data.empty:
        raw_data = raw_data.sort_values('timestamp')
    if not filtered_data.empty:
        filtered_data = filtered_data.sort_values('timestamp')

    # Determine maximum interval
    max_date_raw = raw_data['timestamp'].max() if not raw_data.empty else signup_date
    max_date_filtered = filtered_data['timestamp'].max() if not filtered_data.empty else signup_date
    max_date = max(max_date_raw, max_date_filtered)

    # Calculate number of intervals (cap at 72)
    days_elapsed = (max_date - signup_date).days
    num_intervals = min((days_elapsed // interval_days) + 1, 72)

    # Helper function for weight selection
    def select_weight(data, target_date, window):
        if data.empty:
            return None

        window_start = target_date - timedelta(days=window)
        window_end = target_date + timedelta(days=window)

        mask = (data['timestamp'] >= window_start) & (data['timestamp'] <= window_end)
        in_window = data.loc[mask]

        if in_window.empty:
            return None

        in_window = in_window.copy()
        in_window['distance_seconds'] = abs((in_window['timestamp'] - target_date).dt.total_seconds())

        closest_idx = in_window['distance_seconds'].idxmin()
        best = in_window.loc[closest_idx]

        return {
            'weight': best['weight'],
            'timestamp': best['timestamp'],
            'source': best['source'],
            'distance_days': best['distance_seconds'] / (24 * 3600),
            'quality_score': best.get('quality_score', None)
        }

    # Get baseline weight
    baseline_raw = select_weight(raw_data, signup_date, 3)
    baseline_filtered = select_weight(filtered_data, signup_date, 3) if not filtered_data.empty else None

    # Add baseline
    intervals.append({
        'interval_num': 0,
        'interval_days': 0,
        'target_date': signup_date,
        'raw_weight': baseline_raw['weight'] if baseline_raw else None,
        'filtered_weight': baseline_filtered['weight'] if baseline_filtered else None,
        'raw_source': baseline_raw['source'] if baseline_raw else None,
        'filtered_source': baseline_filtered['source'] if baseline_filtered else None,
        'raw_distance_days': baseline_raw['distance_days'] if baseline_raw else None,
        'filtered_distance_days': baseline_filtered['distance_days'] if baseline_filtered else None
    })

    # Calculate subsequent intervals
    for i in range(1, num_intervals + 1):
        interval_date = signup_date + timedelta(days=i * interval_days)

        if interval_date > datetime.now():
            break

        raw_weight = select_weight(raw_data, interval_date, window_days)
        filtered_weight = select_weight(filtered_data, interval_date, window_days) if not filtered_data.empty else None

        intervals.append({
            'interval_num': i,
            'interval_days': i * interval_days,
            'target_date': interval_date,
            'raw_weight': raw_weight['weight'] if raw_weight else None,
            'filtered_weight': filtered_weight['weight'] if filtered_weight else None,
            'raw_source': raw_weight['source'] if raw_weight else None,
            'filtered_source': filtered_weight['source'] if filtered_weight else None,
            'raw_distance_days': raw_weight['distance_days'] if raw_weight else None,
            'filtered_distance_days': filtered_weight['distance_days'] if filtered_weight else None
        })

    return intervals


class ParallelIntervalAnalyzer:
    """Parallel version of interval analyzer using multiprocessing"""

    def __init__(self, interval_days: int = 5, window_days: float = 2.5, n_workers: Optional[int] = None):
        """
        Initialize parallel interval analyzer

        Args:
            interval_days: Days between each interval
            window_days: ±days tolerance for finding measurements
            n_workers: Number of worker processes (None = cpu_count)
        """
        self.interval_days = interval_days
        self.window_days = window_days
        self.n_workers = n_workers or cpu_count()
        self.logger = logging.getLogger(__name__)

    def calculate_all_users(self,
                          raw_data: pd.DataFrame,
                          filtered_data: pd.DataFrame) -> Dict:
        """
        Calculate interval weights for all users using parallel processing

        Args:
            raw_data: DataFrame with raw measurements
            filtered_data: DataFrame with filtered measurements

        Returns:
            Dictionary mapping user_id to interval data
        """
        start_time = time.time()
        results = {}

        print("\n🔍 Phase 3: Calculating Intervals (Parallel)")
        print("-" * 60)

        if filtered_data.empty:
            print("⚠️ Warning: Filtered data is empty!")
            self.logger.warning("Filtered data is empty, no users to process")
            return {}

        # Get unique users
        filtered_users = set(filtered_data['user_id'].unique())
        raw_users = set(raw_data['user_id'].unique())
        excluded_users = raw_users - filtered_users

        print(f"✓ Found {len(filtered_users):,} users in filtered data")
        print(f"  • {len(raw_users):,} users in raw data")
        print(f"  • {len(excluded_users):,} users excluded")
        print(f"  • Using {self.n_workers} worker processes")

        # Convert dataframes to dictionaries grouped by user_id for serialization
        print("📊 Preparing data for parallel processing...")
        prep_start = time.time()

        raw_dict = {
            user_id: group.reset_index(drop=True)
            for user_id, group in raw_data.groupby('user_id')
        }

        filtered_dict = {
            user_id: group.reset_index(drop=True)
            for user_id, group in filtered_data.groupby('user_id')
        } if not filtered_data.empty else {}

        prep_time = time.time() - prep_start
        print(f"✓ Data prepared in {prep_time:.2f}s")

        # Split users into batches for workers
        user_list = list(filtered_users)
        batch_size = max(1, len(user_list) // (self.n_workers * 4))  # 4 batches per worker
        batches = []

        for i in range(0, len(user_list), batch_size):
            batch_users = user_list[i:i + batch_size]
            batches.append((
                batch_users,
                {uid: raw_dict[uid] for uid in batch_users if uid in raw_dict},
                {uid: filtered_dict[uid] for uid in batch_users if uid in filtered_dict},
                self.interval_days,
                self.window_days
            ))

        print(f"🚀 Processing {len(user_list):,} users in {len(batches)} batches")
        print("-" * 60)

        # Process batches in parallel
        process_start = time.time()

        with Pool(processes=self.n_workers) as pool:
            # Progress tracking
            total_processed = 0
            batch_results = pool.map_async(process_user_batch, batches)

            # Show progress while waiting
            while not batch_results.ready():
                time.sleep(0.5)
                print(f"\r⏳ Processing... (elapsed: {time.time() - process_start:.1f}s)", end='', flush=True)

            # Collect results
            for batch_result in batch_results.get():
                results.update(batch_result)

        print("\r" + " " * 60 + "\r", end='')  # Clear progress line

        process_time = time.time() - process_start
        total_time = time.time() - start_time

        # Calculate statistics
        users_with_intervals = len(results)
        total_intervals = sum(len(r['intervals']) for r in results.values())
        avg_intervals_per_user = total_intervals / users_with_intervals if users_with_intervals else 0
        avg_time_per_user = total_time / len(user_list) if user_list else 0

        # Print summary
        print("\n" + "=" * 60)
        print("✅ INTERVAL CALCULATION COMPLETE (PARALLEL)")
        print("=" * 60)
        print(f"📊 Total time: {total_time:.2f}s")
        print(f"  • Data prep: {prep_time:.2f}s")
        print(f"  • Processing: {process_time:.2f}s")
        print(f"👥 Users processed: {len(user_list):,}")
        print(f"📈 Users with valid intervals: {users_with_intervals:,}")
        print(f"📍 Total intervals calculated: {total_intervals:,}")
        print(f"📊 Average intervals per user: {avg_intervals_per_user:.1f}")
        print(f"⏱️ Average time per user: {avg_time_per_user:.3f}s")
        print(f"🚀 Speedup: {self.n_workers}x workers")
        print("=" * 60 + "\n")

        # Log summary
        self.logger.info(f"Parallel processing complete: {total_time:.2f}s for {len(user_list)} users")
        self.logger.info(f"Used {self.n_workers} workers, achieved {process_time:.2f}s processing time")

        return results

    def get_interval_summary(self, user_intervals: Dict) -> pd.DataFrame:
        """Generate summary statistics for intervals (same as original)"""
        summary_data = []

        max_intervals = max(
            len(data['intervals'])
            for data in user_intervals.values()
        )

        for interval_num in range(max_intervals):
            raw_weights = []
            filtered_weights = []

            for user_data in user_intervals.values():
                if interval_num < len(user_data['intervals']):
                    interval = user_data['intervals'][interval_num]
                    if interval['raw_weight'] is not None:
                        raw_weights.append(interval['raw_weight'])
                    if interval['filtered_weight'] is not None:
                        filtered_weights.append(interval['filtered_weight'])

            if raw_weights or filtered_weights:
                summary_data.append({
                    'interval_num': interval_num,
                    'interval_days': interval_num * self.interval_days,
                    'raw_count': len(raw_weights),
                    'filtered_count': len(filtered_weights),
                    'raw_mean': pd.Series(raw_weights).mean() if raw_weights else None,
                    'filtered_mean': pd.Series(filtered_weights).mean() if filtered_weights else None,
                    'raw_std': pd.Series(raw_weights).std() if len(raw_weights) > 1 else None,
                    'filtered_std': pd.Series(filtered_weights).std() if len(filtered_weights) > 1 else None
                })

        return pd.DataFrame(summary_data)