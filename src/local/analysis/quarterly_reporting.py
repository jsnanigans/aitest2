"""
Quarterly reporting analysis module.
Analyzes weight loss metrics for users in the program for 90+ days,
comparing raw vs filtered data for accurate reporting.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class QuarterlyMetrics:
    """Metrics for quarterly weight loss reporting."""

    # Overall metrics for 90+ day users
    total_users: int
    eligible_users: int  # Users with 90+ days in program

    # Weight loss statistics
    mean_weight_loss_kg: float
    median_weight_loss_kg: float
    std_weight_loss_kg: float

    mean_weight_loss_pct: float
    median_weight_loss_pct: float
    std_weight_loss_pct: float

    # Distribution metrics
    q1_weight_loss: float  # 25th percentile
    q3_weight_loss: float  # 75th percentile
    min_weight_loss: float
    max_weight_loss: float

    # Success rates
    users_losing_5pct: int
    users_losing_10pct: int
    users_losing_15pct: int

    # Data quality
    users_with_valid_start: int
    users_with_valid_endpoint: int
    average_measurements_per_user: float


@dataclass
class CohortAnalysis:
    """Analysis of weight loss at specific time intervals."""

    day_checkpoint: int  # Days since program start (90, 105, 120, etc.)
    total_users_at_checkpoint: int  # Users who have been in program this long
    users_with_data: int  # Users with weight data at this checkpoint

    # Raw data metrics
    raw_mean_loss_kg: float
    raw_mean_loss_pct: float
    raw_std_loss_pct: float
    raw_5pct_success_rate: float
    raw_10pct_success_rate: float

    # Filtered data metrics
    filtered_mean_loss_kg: float
    filtered_mean_loss_pct: float
    filtered_std_loss_pct: float
    filtered_5pct_success_rate: float
    filtered_10pct_success_rate: float

    # Difference metrics
    mean_loss_difference: float  # filtered - raw
    success_rate_5pct_diff: float
    success_rate_10pct_diff: float
    data_availability_improvement: float  # % more users with valid data


class QuarterlyReportingAnalyzer:
    """
    Analyzes weight loss metrics for quarterly reporting,
    focusing on users who have been in the program for 90+ days.
    """

    def __init__(self, today_date: str = "2025-09-05"):
        """
        Initialize the quarterly reporting analyzer.

        Args:
            today_date: The reference date for analysis (when data was exported)
        """
        self.today = pd.to_datetime(today_date)
        logger.info(f"Quarterly analyzer initialized with reference date: {self.today}")

    def load_program_start_dates(self, employer_csv_path: str) -> pd.DataFrame:
        """
        Load program start dates from employer CSV.

        Args:
            employer_csv_path: Path to employer CSV with start_date column

        Returns:
            DataFrame with user_id and start_date columns
        """
        try:
            df = pd.read_csv(employer_csv_path)

            # Ensure required columns exist
            if 'user_id' not in df.columns or 'start_date' not in df.columns:
                logger.error(f"Missing required columns. Found: {df.columns.tolist()}")
                return pd.DataFrame()

            # Parse start_date
            df['start_date'] = pd.to_datetime(df['start_date'])

            # Calculate days in program
            df['days_in_program'] = (self.today - df['start_date']).dt.days

            logger.info(f"Loaded start dates for {len(df)} users")
            logger.info(f"Users with 90+ days: {(df['days_in_program'] >= 90).sum()}")

            return df[['user_id', 'start_date', 'days_in_program']]

        except Exception as e:
            logger.error(f"Error loading start dates: {e}")
            return pd.DataFrame()

    def get_weight_at_date(
        self,
        weight_df: pd.DataFrame,
        target_date: pd.Timestamp,
        before_only: bool = True
    ) -> Optional[float]:
        """
        Get the weight closest to a target date.

        Args:
            weight_df: DataFrame with timestamp and weight columns
            target_date: Target date to find weight for
            before_only: If True, only consider weights on or before target date

        Returns:
            Weight value closest to target date, or None if no valid weight
        """
        if weight_df.empty:
            return None

        # Filter to weights on or before target date if required
        if before_only:
            valid_weights = weight_df[weight_df['timestamp'] <= target_date]
        else:
            valid_weights = weight_df

        if valid_weights.empty:
            return None

        # Find closest weight to target date
        time_diffs = abs(valid_weights['timestamp'] - target_date)
        closest_idx = time_diffs.idxmin()

        return valid_weights.loc[closest_idx, 'weight']

    def calculate_weight_loss_at_checkpoint(
        self,
        user_id: str,
        weight_df: pd.DataFrame,
        start_date: pd.Timestamp,
        checkpoint_days: int
    ) -> Optional[Tuple[float, float]]:
        """
        Calculate weight loss at a specific checkpoint.

        Args:
            user_id: User identifier
            weight_df: User's weight data
            start_date: Program start date
            checkpoint_days: Days since start to calculate loss

        Returns:
            Tuple of (weight_loss_kg, weight_loss_pct) or None
        """
        # Get start weight (within 14 days of start date)
        start_window = start_date + timedelta(days=14)
        start_weight = self.get_weight_at_date(weight_df, start_date, before_only=False)

        # If no exact start date weight, look within window
        if start_weight is None:
            start_weights = weight_df[
                (weight_df['timestamp'] >= start_date - timedelta(days=7)) &
                (weight_df['timestamp'] <= start_window)
            ]
            if not start_weights.empty:
                start_weight = start_weights.iloc[0]['weight']

        if start_weight is None or start_weight <= 0:
            return None

        # Get checkpoint weight
        checkpoint_date = start_date + timedelta(days=checkpoint_days)
        checkpoint_weight = self.get_weight_at_date(weight_df, checkpoint_date, before_only=True)

        if checkpoint_weight is None:
            return None

        # Calculate loss
        weight_loss_kg = start_weight - checkpoint_weight
        weight_loss_pct = (weight_loss_kg / start_weight) * 100

        return weight_loss_kg, weight_loss_pct

    def analyze_cohort_by_duration(
        self,
        raw_data: Dict[str, pd.DataFrame],
        filtered_data: Dict[str, pd.DataFrame],
        start_dates_df: pd.DataFrame,
        checkpoint_days: List[int] = None
    ) -> List[CohortAnalysis]:
        """
        Analyze weight loss at specific time checkpoints.

        Args:
            raw_data: Raw weight data by user
            filtered_data: Filtered weight data by user
            start_dates_df: DataFrame with user start dates
            checkpoint_days: List of days to analyze (default: 90 to 210 by 15)

        Returns:
            List of CohortAnalysis objects for each checkpoint
        """
        if checkpoint_days is None:
            checkpoint_days = list(range(90, 225, 15))  # 90, 105, 120, ... 210

        cohort_results = []

        for days in checkpoint_days:
            logger.info(f"Analyzing {days}-day checkpoint")

            # Find eligible users (in program for at least this many days)
            eligible_users = start_dates_df[
                start_dates_df['days_in_program'] >= days
            ]['user_id'].tolist()

            raw_losses = []
            filtered_losses = []

            for user_id in eligible_users:
                if user_id not in raw_data:
                    continue

                # Get start date for this user
                user_start = start_dates_df[
                    start_dates_df['user_id'] == user_id
                ]['start_date'].iloc[0]

                # Calculate raw data loss
                raw_result = self.calculate_weight_loss_at_checkpoint(
                    user_id, raw_data[user_id], user_start, days
                )
                if raw_result:
                    raw_losses.append(raw_result)

                # Calculate filtered data loss
                if user_id in filtered_data:
                    filtered_result = self.calculate_weight_loss_at_checkpoint(
                        user_id, filtered_data[user_id], user_start, days
                    )
                    if filtered_result:
                        filtered_losses.append(filtered_result)

            # Calculate statistics
            raw_kg_losses = [loss[0] for loss in raw_losses]
            raw_pct_losses = [loss[1] for loss in raw_losses]
            filtered_kg_losses = [loss[0] for loss in filtered_losses]
            filtered_pct_losses = [loss[1] for loss in filtered_losses]

            cohort = CohortAnalysis(
                day_checkpoint=days,
                total_users_at_checkpoint=len(eligible_users),
                users_with_data=len(raw_losses),

                # Raw metrics
                raw_mean_loss_kg=np.mean(raw_kg_losses) if raw_kg_losses else 0,
                raw_mean_loss_pct=np.mean(raw_pct_losses) if raw_pct_losses else 0,
                raw_std_loss_pct=np.std(raw_pct_losses) if raw_pct_losses else 0,
                raw_5pct_success_rate=sum(1 for x in raw_pct_losses if x >= 5) / len(raw_losses) * 100 if raw_losses else 0,
                raw_10pct_success_rate=sum(1 for x in raw_pct_losses if x >= 10) / len(raw_losses) * 100 if raw_losses else 0,

                # Filtered metrics
                filtered_mean_loss_kg=np.mean(filtered_kg_losses) if filtered_kg_losses else 0,
                filtered_mean_loss_pct=np.mean(filtered_pct_losses) if filtered_pct_losses else 0,
                filtered_std_loss_pct=np.std(filtered_pct_losses) if filtered_pct_losses else 0,
                filtered_5pct_success_rate=sum(1 for x in filtered_pct_losses if x >= 5) / len(filtered_losses) * 100 if filtered_losses else 0,
                filtered_10pct_success_rate=sum(1 for x in filtered_pct_losses if x >= 10) / len(filtered_losses) * 100 if filtered_losses else 0,

                # Differences
                mean_loss_difference=(np.mean(filtered_pct_losses) if filtered_pct_losses else 0) -
                                   (np.mean(raw_pct_losses) if raw_pct_losses else 0),
                success_rate_5pct_diff=(sum(1 for x in filtered_pct_losses if x >= 5) / len(filtered_losses) * 100 if filtered_losses else 0) -
                                      (sum(1 for x in raw_pct_losses if x >= 5) / len(raw_losses) * 100 if raw_losses else 0),
                success_rate_10pct_diff=(sum(1 for x in filtered_pct_losses if x >= 10) / len(filtered_losses) * 100 if filtered_losses else 0) -
                                       (sum(1 for x in raw_pct_losses if x >= 10) / len(raw_losses) * 100 if raw_losses else 0),
                data_availability_improvement=(len(filtered_losses) - len(raw_losses)) / len(raw_losses) * 100 if raw_losses else 0
            )

            cohort_results.append(cohort)

            logger.info(f"  {days} days: {len(raw_losses)} users with raw data, "
                       f"{len(filtered_losses)} with filtered data")
            logger.info(f"  Raw mean loss: {cohort.raw_mean_loss_pct:.2f}%, "
                       f"Filtered: {cohort.filtered_mean_loss_pct:.2f}%")

        return cohort_results

    def analyze_all_90plus_users(
        self,
        raw_data: Dict[str, pd.DataFrame],
        filtered_data: Dict[str, pd.DataFrame],
        start_dates_df: pd.DataFrame
    ) -> Tuple[QuarterlyMetrics, QuarterlyMetrics, pd.DataFrame]:
        """
        Analyze all users with 90+ days in program.

        Args:
            raw_data: Raw weight data by user
            filtered_data: Filtered weight data by user
            start_dates_df: DataFrame with user start dates

        Returns:
            Tuple of (raw_metrics, filtered_metrics, detailed_results_df)
        """
        # Get eligible users (90+ days in program)
        eligible_df = start_dates_df[start_dates_df['days_in_program'] >= 90]
        eligible_users = eligible_df['user_id'].tolist()

        logger.info(f"Analyzing {len(eligible_users)} users with 90+ days in program")

        # Collect weight loss data
        results = []

        for _, row in eligible_df.iterrows():
            user_id = row['user_id']
            start_date = row['start_date']

            if user_id not in raw_data:
                continue

            # Get raw data weight loss
            raw_df = raw_data[user_id]
            raw_start = self.get_weight_at_date(raw_df, start_date, before_only=False)

            # Try to get start weight within 14 days if exact date not available
            if raw_start is None:
                start_window = raw_df[
                    (raw_df['timestamp'] >= start_date - timedelta(days=7)) &
                    (raw_df['timestamp'] <= start_date + timedelta(days=14))
                ]
                if not start_window.empty:
                    raw_start = start_window.iloc[0]['weight']

            # Get last weight (up to today)
            raw_end_df = raw_df[raw_df['timestamp'] <= self.today]
            raw_end = raw_end_df['weight'].iloc[-1] if not raw_end_df.empty else None

            # Calculate raw loss
            if raw_start and raw_end and raw_start > 0:
                raw_loss_kg = raw_start - raw_end
                raw_loss_pct = (raw_loss_kg / raw_start) * 100
            else:
                raw_loss_kg = None
                raw_loss_pct = None

            # Get filtered data weight loss
            filtered_loss_kg = None
            filtered_loss_pct = None

            if user_id in filtered_data:
                filtered_df = filtered_data[user_id]
                filtered_start = self.get_weight_at_date(filtered_df, start_date, before_only=False)

                if filtered_start is None:
                    start_window = filtered_df[
                        (filtered_df['timestamp'] >= start_date - timedelta(days=7)) &
                        (filtered_df['timestamp'] <= start_date + timedelta(days=14))
                    ]
                    if not start_window.empty:
                        filtered_start = start_window.iloc[0]['weight']

                filtered_end_df = filtered_df[filtered_df['timestamp'] <= self.today]
                filtered_end = filtered_end_df['weight'].iloc[-1] if not filtered_end_df.empty else None

                if filtered_start and filtered_end and filtered_start > 0:
                    filtered_loss_kg = filtered_start - filtered_end
                    filtered_loss_pct = (filtered_loss_kg / filtered_start) * 100

            results.append({
                'user_id': user_id,
                'start_date': start_date,
                'days_in_program': row['days_in_program'],
                'raw_start_weight': raw_start,
                'raw_end_weight': raw_end,
                'raw_loss_kg': raw_loss_kg,
                'raw_loss_pct': raw_loss_pct,
                'filtered_start_weight': filtered_start if user_id in filtered_data else None,
                'filtered_end_weight': filtered_end if user_id in filtered_data else None,
                'filtered_loss_kg': filtered_loss_kg,
                'filtered_loss_pct': filtered_loss_pct
            })

        # Create results DataFrame
        results_df = pd.DataFrame(results)

        # Calculate raw metrics
        raw_valid = results_df.dropna(subset=['raw_loss_pct'])
        raw_metrics = self._calculate_quarterly_metrics(
            raw_valid, 'raw_loss_kg', 'raw_loss_pct', len(eligible_users)
        )

        # Calculate filtered metrics
        filtered_valid = results_df.dropna(subset=['filtered_loss_pct'])
        filtered_metrics = self._calculate_quarterly_metrics(
            filtered_valid, 'filtered_loss_kg', 'filtered_loss_pct', len(eligible_users)
        )

        logger.info(f"Raw data: {len(raw_valid)} users with valid weight loss data")
        logger.info(f"Filtered data: {len(filtered_valid)} users with valid weight loss data")

        return raw_metrics, filtered_metrics, results_df

    def _calculate_quarterly_metrics(
        self,
        df: pd.DataFrame,
        kg_col: str,
        pct_col: str,
        total_eligible: int
    ) -> QuarterlyMetrics:
        """Calculate quarterly metrics from a results DataFrame."""

        if df.empty:
            return QuarterlyMetrics(
                total_users=total_eligible,
                eligible_users=total_eligible,
                mean_weight_loss_kg=0,
                median_weight_loss_kg=0,
                std_weight_loss_kg=0,
                mean_weight_loss_pct=0,
                median_weight_loss_pct=0,
                std_weight_loss_pct=0,
                q1_weight_loss=0,
                q3_weight_loss=0,
                min_weight_loss=0,
                max_weight_loss=0,
                users_losing_5pct=0,
                users_losing_10pct=0,
                users_losing_15pct=0,
                users_with_valid_start=0,
                users_with_valid_endpoint=0,
                average_measurements_per_user=0
            )

        return QuarterlyMetrics(
            total_users=total_eligible,
            eligible_users=total_eligible,

            mean_weight_loss_kg=df[kg_col].mean(),
            median_weight_loss_kg=df[kg_col].median(),
            std_weight_loss_kg=df[kg_col].std(),

            mean_weight_loss_pct=df[pct_col].mean(),
            median_weight_loss_pct=df[pct_col].median(),
            std_weight_loss_pct=df[pct_col].std(),

            q1_weight_loss=df[pct_col].quantile(0.25),
            q3_weight_loss=df[pct_col].quantile(0.75),
            min_weight_loss=df[pct_col].min(),
            max_weight_loss=df[pct_col].max(),

            users_losing_5pct=len(df[df[pct_col] >= 5]),
            users_losing_10pct=len(df[df[pct_col] >= 10]),
            users_losing_15pct=len(df[df[pct_col] >= 15]),

            users_with_valid_start=len(df),
            users_with_valid_endpoint=len(df),
            average_measurements_per_user=0  # Would need to calculate from raw data
        )