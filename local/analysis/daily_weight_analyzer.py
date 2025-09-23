"""
Daily Weight Analyzer Module

Calculates daily weight averages from raw and filtered data to demonstrate
the Kalman filter's impact on noisy same-day measurements.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging


class DailyWeightAnalyzer:
    """Analyzes weight measurements on a daily basis"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def calculate_daily_weights(self,
                                raw_data: pd.DataFrame,
                                filtered_data: pd.DataFrame) -> Dict:
        """
        Calculate daily average weights for both raw and filtered data

        Args:
            raw_data: DataFrame with columns [user_id, weight, timestamp, source]
            filtered_data: DataFrame with same structure

        Returns:
            Dictionary with daily analysis results per user
        """
        results = {}

        # Process raw data
        raw_daily = self._process_daily_data(raw_data, 'raw')

        # Process filtered data
        filtered_daily = self._process_daily_data(filtered_data, 'filtered')

        # Combine and analyze
        all_user_ids = set(raw_daily.keys()) | set(filtered_daily.keys())

        for user_id in all_user_ids:
            user_result = self._analyze_user_daily(
                raw_daily.get(user_id, {}),
                filtered_daily.get(user_id, {})
            )

            if user_result['has_data']:
                results[user_id] = user_result

        self.logger.info(f"Processed daily weights for {len(results)} users")

        return results

    def _process_daily_data(self, df: pd.DataFrame, data_type: str) -> Dict:
        """
        Process dataframe to get daily averages per user

        Returns:
            Dict[user_id, Dict[date, DailyStats]]
        """
        if df.empty:
            return {}

        # Ensure timestamp is datetime
        df = df.copy()

        # Check if timestamp is already datetime
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            # Fix common date errors (year 0025 should be 2025) only if it's a string
            if pd.api.types.is_string_dtype(df['timestamp']):
                df['timestamp'] = df['timestamp'].str.replace(r'^0025-', '2025-', regex=True)
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            # Drop rows with unparseable dates
            df = df.dropna(subset=['timestamp'])
        df['date'] = df['timestamp'].dt.date

        results = {}

        # Group by user and date
        for user_id, user_data in df.groupby('user_id'):
            daily_weights = {}

            for date, day_data in user_data.groupby('date'):
                weights = day_data['weight'].values
                sources = day_data['source'].values if 'source' in day_data.columns else []

                daily_weights[date] = {
                    'weights': weights.tolist(),
                    'mean': float(np.mean(weights)),
                    'median': float(np.median(weights)),
                    'std': float(np.std(weights)) if len(weights) > 1 else 0,
                    'min': float(np.min(weights)),
                    'max': float(np.max(weights)),
                    'count': len(weights),
                    'range': float(np.max(weights) - np.min(weights)),
                    'sources': list(set(sources)) if len(sources) > 0 else []
                }

            results[user_id] = daily_weights

        return results

    def _analyze_user_daily(self, raw_daily: Dict, filtered_daily: Dict) -> Dict:
        """
        Analyze daily differences for a single user
        """
        if not raw_daily and not filtered_daily:
            return {'has_data': False}

        # Get all dates
        all_dates = sorted(set(raw_daily.keys()) | set(filtered_daily.keys()))

        if not all_dates:
            return {'has_data': False}

        # Calculate baseline (first day with data)
        baseline_date = all_dates[0]
        baseline_raw = raw_daily.get(baseline_date, {}).get('mean')
        baseline_filtered = filtered_daily.get(baseline_date, {}).get('mean')

        # Build daily comparison
        daily_comparison = []
        total_raw_variance = 0
        total_filtered_variance = 0
        total_difference = 0
        max_difference = 0
        max_daily_range_raw = 0
        max_daily_range_filtered = 0

        for date in all_dates:
            raw_stats = raw_daily.get(date, {})
            filtered_stats = filtered_daily.get(date, {})

            comparison = {
                'date': date,
                'days_from_start': (date - baseline_date).days,
                'raw_mean': raw_stats.get('mean'),
                'raw_count': raw_stats.get('count', 0),
                'raw_std': raw_stats.get('std', 0),
                'raw_range': raw_stats.get('range', 0),
                'filtered_mean': filtered_stats.get('mean'),
                'filtered_count': filtered_stats.get('count', 0),
                'filtered_std': filtered_stats.get('std', 0),
                'filtered_range': filtered_stats.get('range', 0),
                'difference': None,
                'measurements_rejected': 0
            }

            # Calculate difference
            if raw_stats.get('mean') is not None and filtered_stats.get('mean') is not None:
                diff = abs(raw_stats['mean'] - filtered_stats['mean'])
                comparison['difference'] = diff
                total_difference += diff
                max_difference = max(max_difference, diff)

            # Track measurements rejected
            comparison['measurements_rejected'] = raw_stats.get('count', 0) - filtered_stats.get('count', 0)

            # Track variance
            if raw_stats.get('std'):
                total_raw_variance += raw_stats['std'] ** 2
            if filtered_stats.get('std'):
                total_filtered_variance += filtered_stats['std'] ** 2

            # Track max daily ranges
            if raw_stats.get('range'):
                max_daily_range_raw = max(max_daily_range_raw, raw_stats['range'])
            if filtered_stats.get('range'):
                max_daily_range_filtered = max(max_daily_range_filtered, filtered_stats['range'])

            daily_comparison.append(comparison)

        # Calculate overall statistics
        valid_comparisons = [c for c in daily_comparison if c['difference'] is not None]

        if valid_comparisons:
            avg_difference = total_difference / len(valid_comparisons)
            differences = [c['difference'] for c in valid_comparisons]
            median_difference = float(np.median(differences))
        else:
            avg_difference = 0
            median_difference = 0

        # Calculate dramatic difference score
        dramatic_score = self._calculate_dramatic_score(
            daily_comparison, max_difference, max_daily_range_raw
        )

        # Calculate trajectory from first to last day
        if len(all_dates) > 1:
            last_date = all_dates[-1]

            # Raw trajectory
            if baseline_raw and raw_daily.get(last_date, {}).get('mean'):
                raw_total_change = raw_daily[last_date]['mean'] - baseline_raw
            else:
                raw_total_change = None

            # Filtered trajectory
            if baseline_filtered and filtered_daily.get(last_date, {}).get('mean'):
                filtered_total_change = filtered_daily[last_date]['mean'] - baseline_filtered
            else:
                filtered_total_change = None
        else:
            raw_total_change = None
            filtered_total_change = None

        return {
            'has_data': True,
            'user_id': None,  # Will be set by caller
            'total_days': len(all_dates),
            'days_with_multiple_measurements': sum(1 for c in daily_comparison
                                                   if c['raw_count'] > 1),
            'daily_comparison': daily_comparison,
            'avg_daily_difference': avg_difference,
            'median_daily_difference': median_difference,
            'max_daily_difference': max_difference,
            'max_daily_range_raw': max_daily_range_raw,
            'max_daily_range_filtered': max_daily_range_filtered,
            'total_raw_measurements': sum(c['raw_count'] for c in daily_comparison),
            'total_filtered_measurements': sum(c['filtered_count'] for c in daily_comparison),
            'rejection_rate': 1 - (sum(c['filtered_count'] for c in daily_comparison) /
                                   sum(c['raw_count'] for c in daily_comparison))
                              if sum(c['raw_count'] for c in daily_comparison) > 0 else 0,
            'raw_total_change': raw_total_change,
            'filtered_total_change': filtered_total_change,
            'dramatic_score': dramatic_score,
            'noise_reduction': {
                'raw_avg_std': (total_raw_variance / len(all_dates)) ** 0.5 if total_raw_variance > 0 else 0,
                'filtered_avg_std': (total_filtered_variance / len(all_dates)) ** 0.5 if total_filtered_variance > 0 else 0
            }
        }

    def _calculate_dramatic_score(self, daily_comparison: List[Dict],
                                  max_difference: float,
                                  max_daily_range: float) -> float:
        """
        Calculate a score indicating how dramatically the filter affected this user's data

        Higher scores indicate more dramatic filtering impact
        """
        score = 0

        # Factor 1: Maximum daily difference (weight: 40%)
        if max_difference > 10:
            score += 40
        elif max_difference > 5:
            score += 30
        elif max_difference > 2:
            score += 20
        elif max_difference > 1:
            score += 10

        # Factor 2: Frequency of differences > 2 lbs (weight: 30%)
        significant_diffs = sum(1 for c in daily_comparison
                               if c['difference'] is not None and c['difference'] > 2)
        if len(daily_comparison) > 0:
            diff_frequency = significant_diffs / len(daily_comparison)
            score += diff_frequency * 30

        # Factor 3: Maximum daily range in raw data (weight: 20%)
        if max_daily_range > 10:
            score += 20
        elif max_daily_range > 5:
            score += 15
        elif max_daily_range > 2:
            score += 10
        elif max_daily_range > 1:
            score += 5

        # Factor 4: Rejection rate (weight: 10%)
        total_raw = sum(c['raw_count'] for c in daily_comparison)
        total_filtered = sum(c['filtered_count'] for c in daily_comparison)
        if total_raw > 0:
            rejection_rate = (total_raw - total_filtered) / total_raw
            score += rejection_rate * 10

        return round(score, 1)

    def get_most_dramatic_users(self, daily_results: Dict, top_n: int = 10) -> List[Dict]:
        """
        Get users with the most dramatic filtering differences
        """
        dramatic_users = []

        for user_id, result in daily_results.items():
            if result['dramatic_score'] > 0:
                dramatic_users.append({
                    'user_id': user_id,
                    'dramatic_score': result['dramatic_score'],
                    'max_daily_difference': result['max_daily_difference'],
                    'avg_daily_difference': result['avg_daily_difference'],
                    'days_with_multiple': result['days_with_multiple_measurements'],
                    'rejection_rate': result['rejection_rate'],
                    'total_days': result['total_days']
                })

        # Sort by dramatic score
        dramatic_users.sort(key=lambda x: x['dramatic_score'], reverse=True)

        return dramatic_users[:top_n]

    def generate_daily_statistics(self, daily_results: Dict) -> Dict:
        """
        Generate overall statistics from daily analysis
        """
        if not daily_results:
            return {}

        all_differences = []
        all_rejection_rates = []
        users_with_dramatic_impact = 0
        total_measurements_raw = 0
        total_measurements_filtered = 0

        for user_id, result in daily_results.items():
            if result['avg_daily_difference'] > 0:
                all_differences.append(result['avg_daily_difference'])

            all_rejection_rates.append(result['rejection_rate'])

            if result['dramatic_score'] > 50:  # Dramatic impact threshold
                users_with_dramatic_impact += 1

            total_measurements_raw += result['total_raw_measurements']
            total_measurements_filtered += result['total_filtered_measurements']

        return {
            'total_users_analyzed': len(daily_results),
            'users_with_dramatic_impact': users_with_dramatic_impact,
            'avg_daily_difference_across_users': float(np.mean(all_differences)) if all_differences else 0,
            'median_daily_difference_across_users': float(np.median(all_differences)) if all_differences else 0,
            'max_daily_difference_observed': max((r['max_daily_difference'] for r in daily_results.values()), default=0),
            'avg_rejection_rate': float(np.mean(all_rejection_rates)) if all_rejection_rates else 0,
            'total_measurements_raw': total_measurements_raw,
            'total_measurements_filtered': total_measurements_filtered,
            'overall_rejection_rate': 1 - (total_measurements_filtered / total_measurements_raw)
                                      if total_measurements_raw > 0 else 0
        }