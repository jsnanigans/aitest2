"""
CSV Generator Module

Generates comprehensive CSV outputs for weight loss interval analysis.
"""

from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import logging


class CSVGenerator:
    """Generates CSV files for analysis results"""

    def __init__(self, output_dir: Path):
        """
        Initialize CSV generator

        Args:
            output_dir: Directory for CSV output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)

    def generate_all_csvs(self,
                         raw_data: pd.DataFrame,
                         filtered_data: pd.DataFrame,
                         user_intervals: Dict,
                         statistics: Dict,
                         top_users: List[Dict]) -> Dict[str, Path]:
        """
        Generate all CSV output files

        Args:
            raw_data: Raw measurement data
            filtered_data: Filtered measurement data
            user_intervals: User interval analysis results
            statistics: Calculated statistics
            top_users: Top divergent users

        Returns:
            Dictionary mapping file type to file path
        """
        files = {}

        # 1. Generate filtered data CSV (raw data with rejected measurements removed)
        files['filtered_data'] = self._generate_filtered_csv(raw_data, filtered_data)

        # 2. Generate user progress CSVs (raw and filtered)
        files['user_progress_raw'] = self._generate_user_progress_csv(
            user_intervals, 'raw', 'user_progress_raw.csv'
        )
        files['user_progress_filtered'] = self._generate_user_progress_csv(
            user_intervals, 'filtered', 'user_progress_filtered.csv'
        )

        # 3. Generate interval statistics CSV
        files['interval_statistics'] = self._generate_interval_statistics_csv(
            statistics['interval_statistics']
        )

        # 4. Generate user comparison CSV
        files['user_comparison'] = self._generate_user_comparison_csv(
            user_intervals, statistics, top_users
        )

        self.logger.info(f"Generated {len(files)} CSV files in {self.output_dir}")
        return files

    def _generate_filtered_csv(self,
                              raw_data: pd.DataFrame,
                              filtered_data: pd.DataFrame) -> Path:
        """Generate CSV with rejected measurements removed"""
        output_path = self.output_dir / 'filtered_data.csv'

        # If we have filtered data from database, use it
        if not filtered_data.empty:
            # Ensure we have all necessary columns
            output_df = filtered_data.copy()

            # Add any missing columns from raw data
            for col in raw_data.columns:
                if col not in output_df.columns and col != 'is_outlier':
                    output_df[col] = None
        else:
            # Fallback: use raw data (no filtering available)
            output_df = raw_data.copy()
            self.logger.warning("No filtered data available, using raw data")

        # Sort by user and timestamp
        output_df = output_df.sort_values(['user_id', 'timestamp'])

        # Save to CSV
        output_df.to_csv(output_path, index=False)
        self.logger.info(f"Saved filtered data: {len(output_df)} records")

        return output_path

    def _generate_user_progress_csv(self,
                                   user_intervals: Dict,
                                   data_type: str,
                                   filename: str) -> Path:
        """Generate user progress CSV for raw or filtered data"""
        output_path = self.output_dir / filename

        # Prepare data for DataFrame
        progress_data = []

        for user_id, user_data in user_intervals.items():
            row = {'user_id': user_id}

            # Add signup date and baseline
            row['baseline_date'] = user_data['signup_date']

            # Add weight at each interval
            for interval in user_data['intervals']:
                interval_key = f"day_{interval['interval_days']}"
                weight_key = f"{data_type}_weight"

                if weight_key in interval:
                    row[interval_key] = interval[weight_key]
                else:
                    row[interval_key] = None

                # Add baseline weight separately
                if interval['interval_days'] == 0:
                    row['baseline_weight'] = interval.get(weight_key)

            progress_data.append(row)

        # Create DataFrame
        df = pd.DataFrame(progress_data)

        # Reorder columns
        base_cols = ['user_id', 'baseline_date', 'baseline_weight']
        interval_cols = sorted([c for c in df.columns if c.startswith('day_')],
                              key=lambda x: int(x.split('_')[1]))
        ordered_cols = base_cols + interval_cols
        df = df[ordered_cols]

        # Calculate summary statistics
        df = self._add_progress_summary(df)

        # Save to CSV
        df.to_csv(output_path, index=False)
        self.logger.info(f"Saved {data_type} progress: {len(df)} users")

        return output_path

    def _generate_interval_statistics_csv(self, interval_stats: List[Dict]) -> Path:
        """Generate interval statistics CSV for box plots"""
        output_path = self.output_dir / 'interval_statistics.csv'

        rows = []
        for interval in interval_stats:
            # Raw statistics
            if interval.get('raw'):
                raw_stats = interval['raw']
                rows.append({
                    'interval': interval['interval_days'],
                    'dataset': 'raw',
                    'mean_change': raw_stats.get('change', {}).get('mean') if 'change' in raw_stats else None,
                    'median_change': raw_stats.get('change', {}).get('median') if 'change' in raw_stats else None,
                    'std_dev': raw_stats.get('change', {}).get('std') if 'change' in raw_stats else None,
                    'q1': raw_stats.get('q1'),
                    'q3': raw_stats.get('q3'),
                    'min': raw_stats.get('min'),
                    'max': raw_stats.get('max'),
                    'n_users': interval['n_users']['raw'],
                    'mean_weight': raw_stats.get('mean'),
                    'median_weight': raw_stats.get('median')
                })

            # Filtered statistics
            if interval.get('filtered'):
                filtered_stats = interval['filtered']
                rows.append({
                    'interval': interval['interval_days'],
                    'dataset': 'filtered',
                    'mean_change': filtered_stats.get('change', {}).get('mean') if 'change' in filtered_stats else None,
                    'median_change': filtered_stats.get('change', {}).get('median') if 'change' in filtered_stats else None,
                    'std_dev': filtered_stats.get('change', {}).get('std') if 'change' in filtered_stats else None,
                    'q1': filtered_stats.get('q1'),
                    'q3': filtered_stats.get('q3'),
                    'min': filtered_stats.get('min'),
                    'max': filtered_stats.get('max'),
                    'n_users': interval['n_users']['filtered'],
                    'mean_weight': filtered_stats.get('mean'),
                    'median_weight': filtered_stats.get('median')
                })

        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        self.logger.info(f"Saved interval statistics: {len(df)} rows")

        return output_path

    def _generate_user_comparison_csv(self,
                                     user_intervals: Dict,
                                     statistics: Dict,
                                     top_users: List[Dict]) -> Path:
        """Generate user comparison CSV with divergence rankings"""
        output_path = self.output_dir / 'user_comparison.csv'

        rows = []
        for rank, user_info in enumerate(top_users, 1):
            user_id = user_info['user_id']

            # Get user statistics
            user_stats = statistics['user_statistics'].get(user_id, {})

            # Calculate total difference
            total_diff = user_info.get('total_diff', 0)
            avg_diff = user_info.get('divergence_score', 0)
            num_intervals = user_info.get('num_intervals', 0)

            # Get outlier impact
            outlier_impact = user_stats.get('outlier_impact', {})
            max_deviation = outlier_impact.get('max_deviation', 0)

            # Calculate consistency score
            consistency = self._calculate_consistency_score(user_id, user_intervals)

            rows.append({
                'user_id': user_id,
                'divergence_rank': rank,
                'total_diff_lbs': round(total_diff, 2),
                'avg_diff_per_interval': round(avg_diff, 2),
                'max_single_diff': round(max_deviation, 2),
                'consistency_score': round(consistency, 3),
                'num_intervals': num_intervals,
                'data_completeness': user_stats.get('data_completeness', 0),
                'weight_loss_raw': user_stats.get('weight_change', {}).get('raw', {}).get('percentage'),
                'weight_loss_filtered': user_stats.get('weight_change', {}).get('filtered', {}).get('percentage'),
                'outcome_changed': self._check_outcome_changed(user_stats)
            })

        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        self.logger.info(f"Saved user comparison: {len(df)} users")

        return output_path

    def _add_progress_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add summary columns to progress DataFrame"""
        # Get day columns
        day_cols = [c for c in df.columns if c.startswith('day_') and c != 'day_0']

        if not day_cols:
            return df

        # Calculate total weight change
        if 'baseline_weight' in df.columns:
            # Find last valid measurement for each user
            df['final_weight'] = df[day_cols].apply(
                lambda row: row.dropna().iloc[-1] if not row.dropna().empty else None,
                axis=1
            )

            # Calculate total change
            df['total_change'] = df['final_weight'] - df['baseline_weight']
            df['percent_change'] = (df['total_change'] / df['baseline_weight']) * 100

            # Round values (convert to numeric first to handle None/NaN)
            df['total_change'] = pd.to_numeric(df['total_change'], errors='coerce').round(2)
            df['percent_change'] = pd.to_numeric(df['percent_change'], errors='coerce').round(2)

        # Count valid measurements
        df['num_measurements'] = df[day_cols].notna().sum(axis=1)

        return df

    def _calculate_consistency_score(self, user_id: str, user_intervals: Dict) -> float:
        """Calculate consistency score for a user"""
        if user_id not in user_intervals:
            return 0.0

        intervals = user_intervals[user_id]['intervals']
        differences = []

        for interval in intervals:
            if interval.get('raw_weight') and interval.get('filtered_weight'):
                diff = abs(interval['raw_weight'] - interval['filtered_weight'])
                differences.append(diff)

        if len(differences) <= 1:
            return 1.0  # Perfect consistency if 0 or 1 measurement

        # Calculate coefficient of variation
        mean_diff = np.mean(differences)
        std_diff = np.std(differences)

        if mean_diff == 0:
            return 1.0  # Perfect consistency if no differences

        cv = std_diff / mean_diff
        # Convert to score (lower CV = higher consistency)
        consistency = 1 / (1 + cv)

        return consistency

    def _check_outcome_changed(self, user_stats: Dict) -> bool:
        """Check if filtering changed the weight loss outcome classification"""
        raw_change = user_stats.get('weight_change', {}).get('raw', {}).get('percentage')
        filtered_change = user_stats.get('weight_change', {}).get('filtered', {}).get('percentage')

        if raw_change is None or filtered_change is None:
            return False

        # Define outcome categories
        def classify_outcome(change):
            if change is None:
                return 'unknown'
            elif change <= -5:
                return 'significant_loss'
            elif change <= -2:
                return 'moderate_loss'
            elif change <= 2:
                return 'stable'
            else:
                return 'gain'

        raw_outcome = classify_outcome(raw_change)
        filtered_outcome = classify_outcome(filtered_change)

        return raw_outcome != filtered_outcome