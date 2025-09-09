"""Utilities for filling gaps in time series with Kalman predictions."""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def add_daily_prediction(
    time_series: List[Dict[str, Any]], kalman_filter: Optional[Any] = None, max_gap_days: int = 0
) -> List[Dict[str, Any]]:
    """
    Fill ALL days in time series data to create a continuous daily series.

    Creates a complete daily time series from first to last date, filling any
    missing days with Kalman filter predictions using the most recent filter state.

    Args:
        time_series: Original time series data with potential gaps
        kalman_filter: Kalman filter instance for predictions
        max_gap_days: Not used (kept for compatibility)

    Returns:
        Complete daily time series with predicted values for all missing days
    """

    if not time_series or len(time_series) < 2:
        return time_series

    first_date = datetime.fromisoformat(time_series[0]["date"].replace("Z", "+00:00"))
    last_date = datetime.fromisoformat(time_series[-1]["date"].replace("Z", "+00:00"))

    # add 1 week to the end to allow for future predictions
    last_date += timedelta(days=7)

    existing_data = {
        datetime.fromisoformat(point["date"].replace("Z", "+00:00")).date(): point
        for point in time_series
    }

    sorted_dates = sorted(
        [datetime.fromisoformat(p["date"].replace("Z", "+00:00")) for p in time_series]
    )

    complete_series = []
    current_date = first_date

    while current_date <= last_date:
        current_date_only = current_date.date()

        # if current_date_only in existing_data:
        #     complete_series.append(existing_data[current_date_only])

        prev_point = None
        next_point = None
        prev_date = None
        next_date = None

        for i, date in enumerate(sorted_dates):
            if date.date() < current_date_only:
                prev_date = date
                prev_point = existing_data[date.date()]
            elif date.date() > current_date_only and next_point is None:
                next_date = date
                next_point = existing_data[date.date()]
                break

        if prev_point and next_point and prev_date and next_date:
            days_from_prev = (current_date - prev_date).days
            total_gap = (next_date - prev_date).days

            predicted_point = create_predicted_point(
                prev_point,
                next_point,
                current_date,
                days_from_prev,
                total_gap,
                kalman_filter,
            )
            complete_series.append(predicted_point)
        elif prev_point and prev_date:
            days_from_prev = (current_date - prev_date).days

            if kalman_filter and prev_point.get("kalman_filtered") is not None:
                kalman_weight = prev_point.get("kalman_filtered", prev_point["weight"])
                kalman_trend = prev_point.get("kalman_trend_kg_per_day", 0)
                kalman_uncertainty = prev_point.get("kalman_uncertainty", 0.5)

                predicted_weight = kalman_weight + (kalman_trend * days_from_prev)
                uncertainty = kalman_uncertainty * np.sqrt(1 + days_from_prev)
            else:
                last_trend = prev_point.get("kalman_trend_kg_per_day", 0)
                predicted_weight = prev_point["weight"] + (last_trend * days_from_prev)
                uncertainty = 1.0 + (days_from_prev * 0.3)

            confidence = max(0.2, 0.8 - (days_from_prev * 0.05))

            complete_series.append(
                {
                    "date": current_date.isoformat(),
                    "weight": round(predicted_weight, 2),
                    "confidence": round(confidence, 3),
                    "source": "kalman-extrapolation",
                    "is_predicted": True,
                    "prediction_gap_days": days_from_prev,
                    "kalman_filtered": round(predicted_weight, 2),
                    "kalman_uncertainty": round(uncertainty, 3),
                }
            )

        current_date += timedelta(days=1)

    return complete_series


def create_predicted_point(
    prev_point: Dict[str, Any],
    next_point: Dict[str, Any],
    pred_date: datetime,
    days_from_prev: int,
    total_gap: int,
    kalman_filter: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Create a predicted data point using the Kalman filter state from the previous actual measurement.

    Args:
        prev_point: Previous actual measurement (contains the filter state after that measurement)
        next_point: Next actual measurement
        pred_date: Date for the prediction
        days_from_prev: Days since previous measurement
        total_gap: Total gap size in days
        kalman_filter: Optional Kalman filter for advanced prediction

    Returns:
        Predicted data point dictionary
    """

    prev_weight = prev_point["weight"]
    next_weight = next_point["weight"]

    if kalman_filter and prev_point.get("kalman_filtered") is not None:
        kalman_weight = prev_point.get("kalman_filtered", prev_weight)
        kalman_trend = prev_point.get("kalman_trend_kg_per_day", 0)
        kalman_uncertainty = prev_point.get("kalman_uncertainty", 0.5)

        predicted_weight = kalman_weight + (kalman_trend * days_from_prev)
        uncertainty = kalman_uncertainty * np.sqrt(1 + days_from_prev)
    else:
        predicted_weight = prev_weight + (next_weight - prev_weight) * (days_from_prev / total_gap)
        uncertainty = 1.0 + (days_from_prev * 0.2)

    confidence = max(0.3, 0.9 - (days_from_prev * 0.1))

    predicted_point = {
        "date": pred_date.isoformat(),
        "weight": round(predicted_weight, 2),
        "confidence": round(confidence, 3),
        "source": "kalman-prediction",
        "is_predicted": True,
        "prediction_gap_days": days_from_prev,
        "kalman_filtered": round(predicted_weight, 2),
        "kalman_uncertainty": round(uncertainty, 3),
    }

    if prev_point.get("kalman_trend_kg_per_day"):
        predicted_point["kalman_trend_kg_per_day"] = prev_point["kalman_trend_kg_per_day"]

    return predicted_point





def interpolate_kalman_predictions(
    dates: List[datetime],
    weights: List[float],
    kalman_results: List[Dict[str, Any]],
    max_gap_days: int = 3,
) -> Tuple[List[datetime], List[float], List[float], List[bool]]:
    """
    Interpolate Kalman predictions for visualization.

    Args:
        dates: Original measurement dates
        weights: Original weight measurements
        kalman_results: Kalman filter results
        max_gap_days: Maximum gap before adding prediction

    Returns:
        Tuple of (all_dates, all_weights, all_kalman, is_predicted_mask)
    """

    if not dates or len(dates) < 2:
        return dates, weights, [], []

    all_dates = []
    all_weights = []
    all_kalman = []
    is_predicted = []

    for i in range(len(dates)):
        all_dates.append(dates[i])
        all_weights.append(weights[i])

        if i < len(kalman_results) and kalman_results[i].get("filtered_weight") is not None:
            all_kalman.append(kalman_results[i]["filtered_weight"])
        else:
            all_kalman.append(weights[i])

        is_predicted.append(False)

        if i < len(dates) - 1:
            gap_days = (dates[i + 1] - dates[i]).days

            if gap_days > max_gap_days:
                num_predictions = gap_days // max_gap_days

                for pred_idx in range(1, num_predictions + 1):
                    pred_date = dates[i] + timedelta(days=pred_idx * max_gap_days)

                    if pred_date >= dates[i + 1]:
                        break

                    if i < len(kalman_results) - 1:
                        if kalman_results[i].get("filtered_weight") and kalman_results[i + 1].get(
                            "filtered_weight"
                        ):
                            k1 = kalman_results[i]["filtered_weight"]
                            k2 = kalman_results[i + 1]["filtered_weight"]
                            frac = (pred_idx * max_gap_days) / gap_days
                            pred_kalman = k1 + (k2 - k1) * frac
                        else:
                            pred_kalman = weights[i] + (weights[i + 1] - weights[i]) * (
                                pred_idx * max_gap_days / gap_days
                            )
                    else:
                        pred_kalman = weights[i]

                    pred_weight = weights[i] + (weights[i + 1] - weights[i]) * (
                        pred_idx * max_gap_days / gap_days
                    )

                    all_dates.append(pred_date)
                    all_weights.append(pred_weight)
                    all_kalman.append(pred_kalman)
                    is_predicted.append(True)

    return all_dates, all_weights, all_kalman, is_predicted
