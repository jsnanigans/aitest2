"""
Statistics Calculator Module

Comprehensive statistical analysis for weight loss data,
comparing raw vs filtered measurements across intervals.
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from scipy import stats
import logging


class StatisticsCalculator:
    """Calculate comprehensive statistics for weight loss analysis"""

    def __init__(self):
        """Initialize statistics calculator"""
        self.logger = logging.getLogger(__name__)

    def calculate_all_statistics(self, user_intervals: Dict) -> Dict:
        """
        Calculate all statistics for the interval analysis

        Args:
            user_intervals: Dictionary of user interval data

        Returns:
            Comprehensive statistics dictionary
        """
        if not user_intervals:
            return self._empty_statistics()

        # Calculate per-user statistics
        user_stats = self._calculate_user_statistics(user_intervals)

        # Calculate interval-level statistics
        interval_stats = self._calculate_interval_statistics(user_intervals)

        # Calculate population-level statistics
        population_stats = self._calculate_population_statistics(user_intervals)

        # Calculate divergence metrics
        divergence_stats = self._calculate_divergence_statistics(user_intervals)

        return {
            'user_statistics': user_stats,
            'interval_statistics': interval_stats,
            'population_statistics': population_stats,
            'divergence_statistics': divergence_stats,
            'total_users': len(user_intervals),
            'intervals_analyzed': self._get_intervals_list(user_intervals)
        }

    def _calculate_user_statistics(self, user_intervals: Dict) -> Dict:
        """Calculate statistics for each individual user"""
        user_stats = {}

        for user_id, data in user_intervals.items():
            intervals = data['intervals']

            # Get baseline and final weights
            baseline = intervals[0] if intervals else None
            final = self._get_last_valid_interval(intervals)

            if baseline and final:
                # Calculate weight changes
                raw_change = self._calculate_weight_change(
                    baseline.get('raw_weight'),
                    final.get('raw_weight')
                )
                filtered_change = self._calculate_weight_change(
                    baseline.get('filtered_weight'),
                    final.get('filtered_weight')
                )

                # Calculate trajectory metrics
                raw_trajectory = self._calculate_trajectory_metrics(intervals, 'raw_weight')
                filtered_trajectory = self._calculate_trajectory_metrics(intervals, 'filtered_weight')

                # Calculate outlier impact
                outlier_impact = self._calculate_outlier_impact(intervals)

                user_stats[user_id] = {
                    'baseline_weight': {
                        'raw': baseline.get('raw_weight'),
                        'filtered': baseline.get('filtered_weight')
                    },
                    'final_weight': {
                        'raw': final.get('raw_weight'),
                        'filtered': final.get('filtered_weight')
                    },
                    'weight_change': {
                        'raw': raw_change,
                        'filtered': filtered_change
                    },
                    'trajectory': {
                        'raw': raw_trajectory,
                        'filtered': filtered_trajectory
                    },
                    'outlier_impact': outlier_impact,
                    'num_intervals': len([i for i in intervals if i.get('raw_weight') or i.get('filtered_weight')]),
                    'data_completeness': self._calculate_completeness(intervals)
                }

        return user_stats

    def _calculate_interval_statistics(self, user_intervals: Dict) -> List[Dict]:
        """Calculate statistics for each interval across all users"""
        interval_stats = []

        # Determine maximum number of intervals
        max_intervals = max(len(data['intervals']) for data in user_intervals.values())

        for interval_num in range(max_intervals):
            raw_weights = []
            filtered_weights = []
            raw_changes = []
            filtered_changes = []

            for user_data in user_intervals.values():
                if interval_num < len(user_data['intervals']):
                    interval = user_data['intervals'][interval_num]
                    baseline = user_data['intervals'][0]

                    # Collect weights
                    if interval['raw_weight'] is not None:
                        raw_weights.append(interval['raw_weight'])
                        if baseline['raw_weight'] is not None and interval_num > 0:
                            change = interval['raw_weight'] - baseline['raw_weight']
                            raw_changes.append(change)

                    if interval['filtered_weight'] is not None:
                        filtered_weights.append(interval['filtered_weight'])
                        if baseline['filtered_weight'] is not None and interval_num > 0:
                            change = interval['filtered_weight'] - baseline['filtered_weight']
                            filtered_changes.append(change)

            # Calculate statistics for this interval
            if raw_weights or filtered_weights:
                stats_dict = {
                    'interval_num': interval_num,
                    'interval_days': interval_num * 30,  # Assuming 30-day intervals
                    'raw': self._calculate_distribution_stats(raw_weights, raw_changes),
                    'filtered': self._calculate_distribution_stats(filtered_weights, filtered_changes),
                    'n_users': {
                        'raw': len(raw_weights),
                        'filtered': len(filtered_weights)
                    }
                }

                # Calculate difference between raw and filtered
                if raw_changes and filtered_changes:
                    stats_dict['difference'] = {
                        'mean_diff': np.mean(filtered_changes) - np.mean(raw_changes),
                        'median_diff': np.median(filtered_changes) - np.median(raw_changes),
                        'correlation': np.corrcoef(raw_changes[:min(len(raw_changes), len(filtered_changes))],
                                                  filtered_changes[:min(len(raw_changes), len(filtered_changes))])[0, 1]
                        if len(raw_changes) > 1 and len(filtered_changes) > 1 else None
                    }

                interval_stats.append(stats_dict)

        return interval_stats

    def _calculate_population_statistics(self, user_intervals: Dict) -> Dict:
        """Calculate population-level statistics"""
        all_raw_changes = []
        all_filtered_changes = []
        success_rates = {'raw': 0, 'filtered': 0}
        total_users = len(user_intervals)

        for user_data in user_intervals.values():
            intervals = user_data['intervals']
            if len(intervals) < 2:
                continue

            baseline = intervals[0]
            final = self._get_last_valid_interval(intervals)

            if baseline and final:
                # Raw changes
                if baseline.get('raw_weight') and final.get('raw_weight'):
                    change = final['raw_weight'] - baseline['raw_weight']
                    all_raw_changes.append(change)
                    # Count as success if lost >5% of body weight
                    pct_change = (change / baseline['raw_weight']) * 100
                    if pct_change <= -5:
                        success_rates['raw'] += 1

                # Filtered changes
                if baseline.get('filtered_weight') and final.get('filtered_weight'):
                    change = final['filtered_weight'] - baseline['filtered_weight']
                    all_filtered_changes.append(change)
                    # Count as success if lost >5% of body weight
                    pct_change = (change / baseline['filtered_weight']) * 100
                    if pct_change <= -5:
                        success_rates['filtered'] += 1

        return {
            'total_users': total_users,
            'users_with_data': len(all_raw_changes),
            'average_weight_loss': {
                'raw': np.mean(all_raw_changes) if all_raw_changes else None,
                'filtered': np.mean(all_filtered_changes) if all_filtered_changes else None
            },
            'median_weight_loss': {
                'raw': np.median(all_raw_changes) if all_raw_changes else None,
                'filtered': np.median(all_filtered_changes) if all_filtered_changes else None
            },
            'success_rate_5pct': {
                'raw': (success_rates['raw'] / total_users * 100) if total_users > 0 else 0,
                'filtered': (success_rates['filtered'] / total_users * 100) if total_users > 0 else 0
            },
            'weight_loss_distribution': {
                'raw': self._calculate_distribution_stats(all_raw_changes),
                'filtered': self._calculate_distribution_stats(all_filtered_changes)
            }
        }

    def _calculate_divergence_statistics(self, user_intervals: Dict) -> Dict:
        """Calculate statistics on divergence between raw and filtered data"""
        divergences = []
        impact_scores = []

        for user_id, data in user_intervals.items():
            user_divergence = self._calculate_user_divergence(data['intervals'])
            if user_divergence['has_divergence']:
                divergences.append(user_divergence['average_divergence'])
                impact_scores.append(user_divergence['impact_score'])

        return {
            'users_with_divergence': len(divergences),
            'average_divergence': np.mean(divergences) if divergences else 0,
            'max_divergence': max(divergences) if divergences else 0,
            'divergence_distribution': self._calculate_distribution_stats(divergences),
            'impact_distribution': self._calculate_distribution_stats(impact_scores)
        }

    def _calculate_weight_change(self, initial: Optional[float], final: Optional[float]) -> Dict:
        """Calculate weight change metrics"""
        if initial is None or final is None:
            return {'absolute': None, 'percentage': None}

        absolute_change = final - initial
        percentage_change = (absolute_change / initial) * 100

        return {
            'absolute': round(absolute_change, 2),
            'percentage': round(percentage_change, 2)
        }

    def _calculate_trajectory_metrics(self, intervals: List[Dict], weight_key: str) -> Dict:
        """Calculate trajectory metrics for weight changes over time"""
        weights = []
        days = []

        for interval in intervals:
            if interval.get(weight_key) is not None:
                weights.append(interval[weight_key])
                days.append(interval['interval_days'])

        if len(weights) < 2:
            return {'trend': None, 'volatility': None, 'consistency': None}

        # Calculate linear trend
        if len(weights) >= 2:
            slope, intercept, r_value, p_value, std_err = stats.linregress(days, weights)
            trend = slope * 30  # Weight change per 30 days
        else:
            trend = None
            r_value = None

        # Calculate volatility (standard deviation of changes)
        changes = np.diff(weights)
        volatility = np.std(changes) if len(changes) > 0 else None

        # Calculate consistency (R-squared of linear fit)
        consistency = r_value ** 2 if r_value is not None else None

        return {
            'trend': round(trend, 3) if trend is not None else None,
            'volatility': round(volatility, 3) if volatility is not None else None,
            'consistency': round(consistency, 3) if consistency is not None else None
        }

    def _calculate_outlier_impact(self, intervals: List[Dict]) -> Dict:
        """Calculate the impact of outlier filtering on measurements"""
        differences = []

        for interval in intervals:
            if interval.get('raw_weight') and interval.get('filtered_weight'):
                diff = abs(interval['raw_weight'] - interval['filtered_weight'])
                differences.append(diff)

        if not differences:
            return {'has_impact': False}

        return {
            'has_impact': True,
            'total_deviation': sum(differences),
            'avg_deviation': np.mean(differences),
            'max_deviation': max(differences),
            'deviation_pattern': self._classify_deviation_pattern(differences)
        }

    def _classify_deviation_pattern(self, deviations: List[float]) -> str:
        """Classify the pattern of deviations"""
        if not deviations:
            return 'none'

        cv = np.std(deviations) / np.mean(deviations) if np.mean(deviations) > 0 else 0

        if cv < 0.3:
            return 'consistent'
        elif cv < 0.7:
            return 'moderate'
        else:
            return 'sporadic'

    def _calculate_user_divergence(self, intervals: List[Dict]) -> Dict:
        """Calculate divergence metrics for a single user"""
        divergences = []

        for interval in intervals:
            if interval.get('raw_weight') and interval.get('filtered_weight'):
                divergence = abs(interval['raw_weight'] - interval['filtered_weight'])
                divergences.append(divergence)

        if not divergences:
            return {'has_divergence': False}

        # Calculate impact score
        avg_divergence = np.mean(divergences)
        consistency = 1 / (1 + np.var(divergences)) if len(divergences) > 1 else 1

        return {
            'has_divergence': True,
            'average_divergence': avg_divergence,
            'max_divergence': max(divergences),
            'consistency': consistency,
            'impact_score': avg_divergence * (2 - consistency)  # Higher divergence and lower consistency = higher impact
        }

    def _calculate_distribution_stats(self, values: List[float], changes: List[float] = None) -> Dict:
        """Calculate distribution statistics for a list of values"""
        if not values:
            return {}

        stats_dict = {
            'mean': np.mean(values),
            'median': np.median(values),
            'std': np.std(values),
            'min': min(values),
            'max': max(values),
            'q1': np.percentile(values, 25),
            'q3': np.percentile(values, 75),
            'iqr': np.percentile(values, 75) - np.percentile(values, 25),
            'n': len(values)
        }

        # Add change statistics if provided
        if changes:
            stats_dict['change'] = {
                'mean': np.mean(changes),
                'median': np.median(changes),
                'std': np.std(changes) if len(changes) > 1 else 0
            }

        return stats_dict

    def _calculate_completeness(self, intervals: List[Dict]) -> float:
        """Calculate data completeness percentage"""
        if not intervals:
            return 0

        valid_count = sum(1 for i in intervals if i.get('raw_weight') or i.get('filtered_weight'))
        return (valid_count / len(intervals)) * 100

    def _get_last_valid_interval(self, intervals: List[Dict]) -> Optional[Dict]:
        """Get the last interval with valid weight data"""
        for interval in reversed(intervals):
            if interval.get('raw_weight') or interval.get('filtered_weight'):
                return interval
        return None

    def _get_intervals_list(self, user_intervals: Dict) -> List[int]:
        """Get list of interval days analyzed"""
        intervals_set = set()
        for data in user_intervals.values():
            for interval in data['intervals']:
                intervals_set.add(interval['interval_days'])
        return sorted(list(intervals_set))

    def _empty_statistics(self) -> Dict:
        """Return empty statistics structure"""
        return {
            'user_statistics': {},
            'interval_statistics': [],
            'population_statistics': {},
            'divergence_statistics': {},
            'total_users': 0,
            'intervals_analyzed': []
        }