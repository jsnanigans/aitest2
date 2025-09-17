"""
Fast Interval Analysis Module (In-Memory)

Calculates weight measurements at regular intervals (e.g., 30-day)
from each user's signup date without database dependency.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import logging
import time


class FastIntervalAnalyzer:
    """Analyzes weight measurements at regular intervals from signup (optimized version)"""

    def __init__(self, interval_days: int = 5, window_days: float = 2.5):
        """
        Initialize interval analyzer

        Args:
            interval_days: Days between each interval (default: 5)
            window_days: ±days tolerance for finding measurements (default: 2.5)
        """
        self.interval_days = interval_days
        self.window_days = window_days
        self.logger = logging.getLogger(__name__)

    def calculate_all_users(self,
                          raw_data: pd.DataFrame,
                          filtered_data: pd.DataFrame) -> Dict:
        """
        Calculate interval weights for all users

        Args:
            raw_data: DataFrame with raw measurements
            filtered_data: DataFrame with filtered measurements

        Returns:
            Dictionary mapping user_id to interval data
        """
        start_time = time.time()
        results = {}

        # Get unique users - only from filtered data (users with 20+ measurements and some non-outliers)
        print("\n🔍 Phase 3: Calculating Intervals")
        print("-" * 60)
        print("Identifying users from filtered data...")
        self.logger.info("Identifying users from filtered data...")

        if filtered_data.empty:
            print("⚠️ Warning: Filtered data is empty!")
            self.logger.warning("Filtered data is empty, no users to process")
            return {}

        filtered_users = set(filtered_data['user_id'].unique())
        raw_users = set(raw_data['user_id'].unique())

        # Users that were completely filtered out (< 20 measurements or all outliers)
        excluded_users = raw_users - filtered_users

        print(f"✓ Found {len(filtered_users):,} users in filtered data")
        print(f"  • {len(raw_users):,} users in raw data")
        print(f"  • {len(excluded_users):,} users excluded (<20 measurements or all outliers)")

        self.logger.info(f"Found {len(filtered_users)} users in filtered data to process")
        self.logger.info(f"Raw data: {len(raw_users)} users, Excluded: {len(excluded_users)} users")
        self.logger.info(f"Raw data shape: {raw_data.shape}, Filtered data shape: {filtered_data.shape}")

        # Use filtered users as our processing list
        all_users = filtered_users

        # Pre-index data for faster lookups
        print("📊 Pre-indexing data for efficient access...")
        self.logger.info("Pre-indexing data by user_id for efficient access...")
        index_start = time.time()
        raw_grouped = raw_data.groupby('user_id')
        filtered_grouped = filtered_data.groupby('user_id') if not filtered_data.empty else None
        index_time = time.time() - index_start
        print(f"✓ Data indexed in {index_time:.2f}s")
        self.logger.info(f"Data indexing complete in {index_time:.2f}s")

        # Process users in batches for progress tracking
        user_list = list(all_users)
        batch_size = 100
        total_batches = (len(user_list) + batch_size - 1) // batch_size

        print(f"\n🚀 Processing {len(user_list):,} users in {total_batches} batches")
        print("-" * 60)
        self.logger.info(f"Processing users in {total_batches} batches of {batch_size}")

        users_with_signup = 0
        users_without_signup = 0
        users_with_intervals = 0
        total_intervals_calculated = 0

        for batch_num, batch_idx in enumerate(range(0, len(user_list), batch_size), 1):
            batch_users = user_list[batch_idx:batch_idx + batch_size]
            batch_start = time.time()
            batch_results = 0

            # Show progress bar
            progress = min(batch_idx + batch_size, len(user_list))
            percentage = (progress / len(user_list)) * 100
            bar_length = 40
            filled_length = int(bar_length * progress // len(user_list))
            bar = '█' * filled_length + '░' * (bar_length - filled_length)

            print(f"\rBatch {batch_num}/{total_batches}: [{bar}] {percentage:.1f}% ({progress:,}/{len(user_list):,} users)", end='', flush=True)

            self.logger.debug(f"Starting batch {batch_num}/{total_batches} with {len(batch_users)} users")

            for user_idx, user_id in enumerate(batch_users):
                try:
                    # Get user data efficiently
                    user_raw = raw_grouped.get_group(user_id) if user_id in raw_grouped.groups else pd.DataFrame()
                    user_filtered = filtered_grouped.get_group(user_id) if filtered_grouped and user_id in filtered_grouped.groups else pd.DataFrame()

                    # Find signup date
                    signup_date = self._find_signup_date(user_raw)
                    if signup_date is None:
                        users_without_signup += 1
                        self.logger.debug(f"User {user_id}: No signup date found, skipping")
                        continue

                    users_with_signup += 1

                    # Calculate intervals
                    intervals = self._calculate_user_intervals_optimized(
                        user_id, signup_date, user_raw, user_filtered
                    )

                    if intervals:
                        results[user_id] = {
                            'user_id': user_id,
                            'signup_date': signup_date,
                            'intervals': intervals,
                            'num_raw_measurements': len(user_raw),
                            'num_filtered_measurements': len(user_filtered) if not user_filtered.empty else 0
                        }
                        batch_results += 1
                        users_with_intervals += 1
                        total_intervals_calculated += len(intervals)

                        if len(results) % 500 == 0 and len(results) > 0:
                            print(f"\n✨ Milestone: {len(results):,} users processed successfully!", end='')
                            self.logger.info(f"Milestone: Processed {len(results)} users successfully")

                except Exception as e:
                    self.logger.warning(f"Error processing user {user_id}: {e}")
                    continue

            # Log batch progress (but don't print to avoid disrupting progress bar)
            batch_time = time.time() - batch_start
            processed = min(batch_idx + batch_size, len(user_list))
            avg_time = batch_time/len(batch_users) if batch_users else 0

            self.logger.info(f"Batch {batch_num}/{total_batches} complete: "
                           f"{processed}/{len(user_list)} users processed | "
                           f"Batch: {batch_results} successful | "
                           f"Time: {batch_time:.2f}s ({avg_time:.3f}s/user)")

        # Clear the progress bar line
        print("\r" + " " * 80 + "\r", end='')

        total_time = time.time() - start_time
        avg_time_per_user = total_time/len(all_users) if all_users else 0
        avg_intervals_per_user = total_intervals_calculated/len(results) if results else 0

        # Print summary to stdout
        print("\n" + "=" * 60)
        print("✅ INTERVAL CALCULATION COMPLETE")
        print("=" * 60)
        print(f"📊 Total time: {total_time:.2f}s")
        print(f"👥 Users processed: {len(all_users):,} (from filtered data)")
        print(f"✓  Users with signup date: {users_with_signup:,}")
        print(f"✗  Users without signup date: {users_without_signup:,}")
        print(f"📈 Users with valid intervals: {users_with_intervals:,}")
        print(f"📍 Total intervals calculated: {total_intervals_calculated:,}")
        print(f"📊 Average intervals per user: {avg_intervals_per_user:.1f}")
        print(f"⏱️  Average time per user: {avg_time_per_user:.3f}s")
        print("=" * 60 + "\n")

        # Also log to file
        self.logger.info("=" * 60)
        self.logger.info("INTERVAL CALCULATION SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Total time: {total_time:.2f}s")
        self.logger.info(f"Users processed: {len(all_users)}")
        self.logger.info(f"Users with signup date: {users_with_signup}")
        self.logger.info(f"Users without signup date: {users_without_signup}")
        self.logger.info(f"Users with intervals: {users_with_intervals}")
        self.logger.info(f"Total intervals calculated: {total_intervals_calculated}")
        self.logger.info(f"Average intervals per user: {avg_intervals_per_user:.1f}")
        self.logger.info(f"Average time per user: {avg_time_per_user:.3f}s")
        self.logger.info("=" * 60)

        return results

    def _find_signup_date(self, user_data: pd.DataFrame) -> Optional[datetime]:
        """Find user's signup date from initial-questionnaire source"""
        if user_data.empty:
            return None

        signup_rows = user_data[user_data['source'] == 'initial-questionnaire']

        if not signup_rows.empty:
            return signup_rows.iloc[0]['timestamp']

        # Fallback to first measurement
        return user_data.iloc[0]['timestamp']

    def _calculate_user_intervals_optimized(self,
                                           user_id: str,
                                           signup_date: datetime,
                                           raw_data: pd.DataFrame,
                                           filtered_data: pd.DataFrame) -> List[Dict]:
        """Calculate interval measurements for a single user (optimized)"""
        intervals = []

        # Pre-sort data once for efficiency
        if not raw_data.empty:
            raw_data = raw_data.sort_values('timestamp')
        if not filtered_data.empty:
            filtered_data = filtered_data.sort_values('timestamp')

        # Determine maximum interval
        max_date_raw = raw_data['timestamp'].max() if not raw_data.empty else signup_date
        max_date_filtered = filtered_data['timestamp'].max() if not filtered_data.empty else signup_date
        max_date = max(max_date_raw, max_date_filtered)

        # Calculate number of intervals (with 5-day intervals, cap at 72 = ~1 year)
        days_elapsed = (max_date - signup_date).days
        num_intervals = min((days_elapsed // self.interval_days) + 1, 72)  # Cap at 72 intervals (~360 days)

        self.logger.debug(f"User {user_id}: {days_elapsed} days elapsed, calculating {num_intervals} intervals")

        # Get baseline weight (optimized)
        baseline_raw = self._select_weight_optimized(
            raw_data, signup_date, window_days=3
        )
        baseline_filtered = self._select_weight_optimized(
            filtered_data, signup_date, window_days=3
        ) if not filtered_data.empty else None

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
            interval_date = signup_date + timedelta(days=i * self.interval_days)

            if interval_date > datetime.now():
                break

            raw_weight = self._select_weight_optimized(raw_data, interval_date, self.window_days)
            filtered_weight = self._select_weight_optimized(
                filtered_data, interval_date, self.window_days
            ) if not filtered_data.empty else None

            # Debug: Check if we're getting the same weights
            if raw_weight and filtered_weight and i <= 3:  # Only log first few intervals
                if abs(raw_weight['weight'] - filtered_weight['weight']) < 0.01:
                    self.logger.debug(f"User {user_id}, Interval {i}: Same weight selected from raw and filtered: {raw_weight['weight']:.2f}")
                else:
                    self.logger.debug(f"User {user_id}, Interval {i}: Different weights - Raw: {raw_weight['weight']:.2f}, Filtered: {filtered_weight['weight']:.2f}")

            intervals.append({
                'interval_num': i,
                'interval_days': i * self.interval_days,
                'target_date': interval_date,
                'raw_weight': raw_weight['weight'] if raw_weight else None,
                'filtered_weight': filtered_weight['weight'] if filtered_weight else None,
                'raw_source': raw_weight['source'] if raw_weight else None,
                'filtered_source': filtered_weight['source'] if filtered_weight else None,
                'raw_distance_days': raw_weight['distance_days'] if raw_weight else None,
                'filtered_distance_days': filtered_weight['distance_days'] if filtered_weight else None
            })

        return intervals

    def _select_weight_optimized(self,
                                data: pd.DataFrame,
                                target_date: datetime,
                                window_days: float) -> Optional[Dict]:
        """
        Select the CLOSEST weight measurement near a target date (simplified)
        Only criteria: closest time to target date
        """
        if data.empty:
            return None

        # Use vectorized operations
        window_start = target_date - timedelta(days=window_days)
        window_end = target_date + timedelta(days=window_days)

        # Vectorized filtering
        mask = (data['timestamp'] >= window_start) & (data['timestamp'] <= window_end)
        in_window = data.loc[mask]

        if in_window.empty:
            return None

        # Calculate absolute time distance (in seconds for precision)
        in_window = in_window.copy()
        in_window['distance_seconds'] = abs((in_window['timestamp'] - target_date).dt.total_seconds())

        # Simply get the closest measurement by time - nothing else matters
        closest_idx = in_window['distance_seconds'].idxmin()
        best = in_window.loc[closest_idx]

        # Convert seconds back to days for reporting
        distance_days = best['distance_seconds'] / (24 * 3600)

        return {
            'weight': best['weight'],
            'timestamp': best['timestamp'],
            'source': best['source'],
            'distance_days': distance_days,
            'quality_score': best.get('quality_score', None)
        }

    def get_interval_summary(self, user_intervals: Dict) -> pd.DataFrame:
        """Generate summary statistics for intervals"""
        summary_data = []

        # Collect data for each interval
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