import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import scipy.stats as stats

def extract_kalman_data(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Extract Kalman filter data from processing results."""
    data = []
    
    for r in results:
        if not r:
            continue
        
        # Extract basic data
        row = {
            "timestamp": pd.to_datetime(r.get("timestamp")),
            "raw_weight": r.get("raw_weight"),
            "cleaned_weight": r.get("cleaned_weight", r.get("raw_weight")),
            "filtered_weight": r.get("filtered_weight"),
            "trend": r.get("trend"),
            "trend_weekly": r.get("trend_weekly"),
            "accepted": r.get("accepted", False),
            "source": r.get("source", "unknown"),
            "confidence": r.get("confidence"),
            "innovation": r.get("innovation"),
            "normalized_innovation": r.get("normalized_innovation"),
            "quality_score": r.get("quality_score"),
            "rejection_reason": r.get("rejection_reason") or r.get("reason"),
            "is_reset": r.get("is_reset", False),
            "gap_days": r.get("gap_days", 0),
        }
        
        # Extract Kalman state if available
        if "kalman_state" in r:
            ks = r["kalman_state"]
            row.update({
                "state_weight": ks.get("filtered_weight"),
                "state_trend": ks.get("filtered_trend"),
                "prediction": ks.get("prediction"),
                "innovation_covariance": ks.get("innovation_covariance"),
                "kalman_gain": ks.get("gain"),
            })
        
        # Extract state covariance if available
        if "state_covariance" in r:
            cov = r["state_covariance"]
            if isinstance(cov, (list, np.ndarray)) and len(cov) >= 2:
                row["weight_variance"] = cov[0][0] if isinstance(cov[0], (list, tuple)) else cov[0]
                row["trend_variance"] = cov[1][1] if isinstance(cov[1], (list, tuple)) and len(cov[1]) > 1 else None
        
        data.append(row)
    
    df = pd.DataFrame(data)
    
    if not df.empty:
        df = df.sort_values("timestamp")
        
        # Calculate derived metrics
        df = calculate_kalman_metrics(df)
        
        # Calculate uncertainty bands
        df = calculate_uncertainty_bands(df)
        
        # Identify reset points and gaps
        df = identify_resets_and_gaps(df)
    
    return df

def calculate_kalman_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate additional Kalman filter metrics."""
    if df.empty:
        return df
    
    # Calculate prediction for next measurement
    if "filtered_weight" in df.columns and "trend" in df.columns:
        df["prediction_next"] = df["filtered_weight"] + df["trend"].fillna(0)
    
    # Calculate actual innovation if not present
    if "innovation" not in df.columns or df["innovation"].isna().all():
        if "raw_weight" in df.columns and "filtered_weight" in df.columns:
            df["innovation"] = df["raw_weight"] - df["filtered_weight"].shift(1)
    
    # Calculate daily changes
    if "filtered_weight" in df.columns:
        df["daily_change"] = df["filtered_weight"].diff()
        
        # Calculate rolling statistics
        df["daily_change_ma7"] = df["daily_change"].rolling(window=7, min_periods=1).mean()
        df["daily_change_std7"] = df["daily_change"].rolling(window=7, min_periods=1).std()
    
    # Calculate innovation statistics
    if "innovation" in df.columns:
        df["innovation_ma"] = df["innovation"].rolling(window=10, min_periods=1).mean()
        df["innovation_std"] = df["innovation"].rolling(window=10, min_periods=1).std()
        
        # Check for white noise (innovation should be zero-mean)
        df["innovation_zscore"] = (df["innovation"] - df["innovation_ma"]) / df["innovation_std"].replace(0, 1)
    
    # Calculate convergence metric (decreasing variance over time)
    if "confidence" in df.columns:
        df["convergence_rate"] = df["confidence"].pct_change(fill_method=None)
    
    return df

def calculate_uncertainty_bands(df: pd.DataFrame, confidence_levels: List[float] = [1, 2]) -> pd.DataFrame:
    """Calculate uncertainty bands for Kalman filtered values."""
    if df.empty or "filtered_weight" not in df.columns:
        return df
    
    # Use weight variance if available, otherwise estimate from innovation
    if "weight_variance" in df.columns:
        std_dev = np.sqrt(df["weight_variance"].fillna(1.0))
    elif "innovation_std" in df.columns:
        std_dev = df["innovation_std"].fillna(1.0)
    else:
        # Estimate from data
        if "innovation" in df.columns:
            std_dev = df["innovation"].rolling(window=20, min_periods=1).std().fillna(1.0)
        else:
            std_dev = pd.Series([1.0] * len(df), index=df.index)
    
    # Calculate bands for each confidence level
    for sigma in confidence_levels:
        df[f"upper_{sigma}sigma"] = df["filtered_weight"] + sigma * std_dev
        df[f"lower_{sigma}sigma"] = df["filtered_weight"] - sigma * std_dev
    
    # Calculate confidence interval (95%)
    df["ci_upper"] = df["filtered_weight"] + 1.96 * std_dev
    df["ci_lower"] = df["filtered_weight"] - 1.96 * std_dev
    
    return df

def identify_resets_and_gaps(df: pd.DataFrame, gap_threshold_days: int = 7) -> pd.DataFrame:
    """Identify reset points and measurement gaps."""
    if df.empty:
        return df
    
    # Calculate time gaps between measurements
    df["time_gap_days"] = df["timestamp"].diff().dt.total_seconds() / 86400
    
    # Mark large gaps
    df["has_gap"] = df["time_gap_days"] > gap_threshold_days
    
    # Identify resets (already marked or detected from large state changes)
    if "is_reset" not in df.columns:
        df["is_reset"] = False
        
        # Detect resets from large jumps in filtered weight
        if "filtered_weight" in df.columns:
            weight_change = df["filtered_weight"].diff().abs()
            median_change = weight_change.median()
            df["is_reset"] = weight_change > (5 * median_change)
    
    # Mark gap regions for visualization
    df["gap_start"] = df["has_gap"].shift(1).astype('boolean').fillna(False)
    df["gap_end"] = df["has_gap"]
    
    return df

def calculate_filter_performance(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate Kalman filter performance metrics."""
    metrics = {}
    
    if df.empty:
        return metrics
    
    # Innovation statistics (should be zero-mean white noise)
    if "innovation" in df.columns and not df["innovation"].isna().all():
        innovation = df["innovation"].dropna()
        metrics["innovation_mean"] = innovation.mean()
        metrics["innovation_std"] = innovation.std()
        metrics["innovation_skew"] = innovation.skew()
        metrics["innovation_kurtosis"] = innovation.kurtosis()
        
        # Normality test
        if len(innovation) > 3:
            _, p_value = stats.normaltest(innovation)
            metrics["innovation_normality_pvalue"] = p_value
            metrics["innovation_is_normal"] = p_value > 0.05
    
    # Normalized innovation squared (NIS) - should follow chi-squared distribution
    if "normalized_innovation" in df.columns:
        nis = (df["normalized_innovation"] ** 2).dropna()
        if len(nis) > 0:
            metrics["nis_mean"] = nis.mean()
            metrics["nis_std"] = nis.std()
            # For 1-DOF, mean should be ~1, std should be ~sqrt(2)
            metrics["nis_consistency"] = abs(nis.mean() - 1.0) < 0.5
    
    # Confidence/convergence metrics
    if "confidence" in df.columns:
        confidence = df["confidence"].dropna()
        if len(confidence) > 0:
            metrics["avg_confidence"] = confidence.mean()
            metrics["final_confidence"] = confidence.iloc[-1]
            metrics["confidence_trend"] = confidence.iloc[-1] - confidence.iloc[0] if len(confidence) > 1 else 0
    
    # Filter effectiveness (how well it tracks accepted measurements)
    accepted_df = df[df["accepted"] == True]
    if len(accepted_df) > 0 and "innovation" in accepted_df.columns:
        tracking_error = accepted_df["innovation"].abs().mean()
        metrics["tracking_error"] = tracking_error
        metrics["tracking_quality"] = "good" if tracking_error < 2.0 else "poor"
    
    # Stability metrics
    if "filtered_weight" in df.columns:
        filtered = df["filtered_weight"].dropna()
        if len(filtered) > 1:
            metrics["filter_stability"] = filtered.std()
            metrics["filter_range"] = filtered.max() - filtered.min()
    
    return metrics

def calculate_baseline_and_thresholds(df: pd.DataFrame, config: Dict[str, Any] = {}) -> Dict[str, float]:
    """Calculate baseline weight and extreme thresholds."""
    thresholds = {}
    
    if df.empty or "filtered_weight" not in df.columns:
        return thresholds
    
    # Baseline: Use first valid filtered weight or median
    filtered_weights = df["filtered_weight"].dropna()
    if len(filtered_weights) > 0:
        thresholds["baseline"] = filtered_weights.iloc[0]
        thresholds["median_weight"] = filtered_weights.median()
        thresholds["current_weight"] = filtered_weights.iloc[-1]
    
    # Calculate extreme thresholds
    extreme_pct = config.get("extreme_threshold", 0.20)
    if "baseline" in thresholds:
        thresholds["upper_extreme"] = thresholds["baseline"] * (1 + extreme_pct)
        thresholds["lower_extreme"] = thresholds["baseline"] * (1 - extreme_pct)
    
    # Calculate physiological limits
    thresholds["min_physiological"] = config.get("min_weight", 30.0)
    thresholds["max_physiological"] = config.get("max_weight", 400.0)
    
    return thresholds

def prepare_kalman_visualization_data(df: pd.DataFrame, config: Dict[str, Any] = {}) -> Dict[str, Any]:
    """Prepare all data needed for Kalman visualization."""
    viz_data = {
        "dataframe": df,
        "metrics": calculate_filter_performance(df),
        "thresholds": calculate_baseline_and_thresholds(df, config),
    }
    
    # Separate accepted and rejected measurements
    viz_data["accepted_df"] = df[df["accepted"] == True].copy()
    viz_data["rejected_df"] = df[df["accepted"] == False].copy()
    
    # Extract reset points
    viz_data["reset_points"] = df[df["is_reset"] == True].copy()
    
    # Extract gap regions
    gap_starts = df[df["gap_start"] == True]["timestamp"].tolist()
    gap_ends = df[df["gap_end"] == True]["timestamp"].tolist()
    viz_data["gaps"] = list(zip(gap_starts, gap_ends))
    
    # Calculate daily change distribution
    if "daily_change" in df.columns:
        daily_changes = df["daily_change"].dropna()
        if len(daily_changes) > 0:
            viz_data["daily_change_hist"] = {
                "values": daily_changes.tolist(),
                "mean": daily_changes.mean(),
                "std": daily_changes.std(),
                "percentiles": {
                    "p5": daily_changes.quantile(0.05),
                    "p25": daily_changes.quantile(0.25),
                    "p50": daily_changes.quantile(0.50),
                    "p75": daily_changes.quantile(0.75),
                    "p95": daily_changes.quantile(0.95),
                }
            }
    
    # Innovation distribution
    if "innovation" in df.columns:
        innovations = df["innovation"].dropna()
        if len(innovations) > 0:
            viz_data["innovation_dist"] = {
                "values": innovations.tolist(),
                "mean": innovations.mean(),
                "std": innovations.std(),
                "zero_centered": abs(innovations.mean()) < 0.5,
            }
    
    return viz_data

def format_kalman_hover_text(row: pd.Series) -> str:
    """Format hover text for Kalman visualization."""
    lines = []
    
    # Timestamp
    lines.append(f"<b>{row['timestamp'].strftime('%Y-%m-%d %H:%M')}</b>")
    
    # Weight values
    if pd.notna(row.get('raw_weight')):
        lines.append(f"Raw Weight: <b>{row['raw_weight']:.1f} kg</b>")
    if pd.notna(row.get('filtered_weight')):
        lines.append(f"Filtered: <b>{row['filtered_weight']:.1f} kg</b>")
    
    # Kalman metrics
    if pd.notna(row.get('innovation')):
        lines.append(f"Innovation: {row['innovation']:.2f}")
    if pd.notna(row.get('normalized_innovation')):
        lines.append(f"Normalized: {row['normalized_innovation']:.2f}")
    if pd.notna(row.get('confidence')):
        lines.append(f"Confidence: {row['confidence']:.1%}")
    
    # Quality score
    if pd.notna(row.get('quality_score')):
        lines.append(f"Quality: {row['quality_score']:.2f}")
    
    # Source
    if row.get('source'):
        lines.append(f"Source: {row['source']}")
    
    # Status
    if row.get('accepted'):
        lines.append("<b>Status: Accepted</b>")
    else:
        reason = row.get('rejection_reason', 'Unknown')
        lines.append(f"<b>Status: Rejected</b><br>Reason: {reason}")
    
    return "<br>".join(lines)