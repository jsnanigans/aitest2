import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta
from collections import defaultdict

def extract_quality_data(results: List[Dict[str, Any]]) -> pd.DataFrame:
    data = []
    for r in results:
        if not r:
            continue
        
        quality_score = r.get("quality_score", None)
        quality_components = r.get("quality_components", {})
        
        data.append({
            "timestamp": pd.to_datetime(r.get("timestamp")),
            "weight": r.get("weight"),
            "accepted": r.get("accepted", False),
            "source": r.get("source", "unknown"),
            "quality_score": quality_score,
            "safety": quality_components.get("safety", None),
            "plausibility": quality_components.get("plausibility", None),
            "consistency": quality_components.get("consistency", None),
            "reliability": quality_components.get("reliability", None),
            "rejection_reason": r.get("rejection_reason", None),
            "is_reset": r.get("is_reset", False),
            "gap_days": r.get("gap_days", 0)
        })
    
    df = pd.DataFrame(data)
    if not df.empty:
        df = df.sort_values("timestamp")
        if "quality_score" in df.columns and not df["quality_score"].isna().all():
            df["quality_category"] = pd.cut(
                df["quality_score"].fillna(0),
                bins=[0, 0.4, 0.55, 0.6, 0.7, 0.8, 1.0],
                labels=["Very Low", "Low", "Near Threshold", "Acceptable", "Good", "High"]
            )
    
    return df

def calculate_quality_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {}
    
    accepted_df = df[df["accepted"] == True]
    rejected_df = df[df["accepted"] == False]
    
    stats = {
        "overall_quality": df["quality_score"].mean() if not df["quality_score"].isna().all() else None,
        "quality_std": df["quality_score"].std() if not df["quality_score"].isna().all() else None,
        "acceptance_rate": len(accepted_df) / len(df) * 100 if len(df) > 0 else 0,
        "total_measurements": len(df),
        "accepted_count": len(accepted_df),
        "rejected_count": len(rejected_df),
    }
    
    if not df["quality_score"].isna().all():
        stats.update({
            "quality_percentiles": {
                "p25": df["quality_score"].quantile(0.25),
                "p50": df["quality_score"].quantile(0.50),
                "p75": df["quality_score"].quantile(0.75),
            },
            "near_threshold_count": len(df[(df["quality_score"] >= 0.55) & (df["quality_score"] <= 0.65)]),
            "high_quality_count": len(df[df["quality_score"] > 0.8]),
        })
    
    for component in ["safety", "plausibility", "consistency", "reliability"]:
        if component in df.columns and not df[component].isna().all():
            stats[f"{component}_mean"] = df[component].mean()
            stats[f"{component}_std"] = df[component].std()
    
    return stats

def detect_quality_patterns(df: pd.DataFrame) -> List[Dict[str, Any]]:
    patterns = []
    
    if df.empty or "quality_score" not in df.columns:
        return patterns
    
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["date"] = df["timestamp"].dt.date
    
    hourly_quality = df.groupby("hour")["quality_score"].mean()
    if not hourly_quality.empty and not hourly_quality.isna().all():
        best_hour = hourly_quality.idxmax()
        worst_hour = hourly_quality.idxmin()
        if not pd.isna(best_hour) and not pd.isna(worst_hour) and hourly_quality[best_hour] - hourly_quality[worst_hour] > 0.1:
            patterns.append({
                "type": "time_of_day",
                "description": f"Quality scores are {(hourly_quality[best_hour] - hourly_quality[worst_hour])*100:.0f}% higher at {best_hour:02d}:00 than at {worst_hour:02d}:00",
                "severity": "info"
            })
    
    gap_quality = df[df["gap_days"] > 7]["quality_score"].mean()
    no_gap_quality = df[df["gap_days"] <= 1]["quality_score"].mean()
    if not pd.isna(gap_quality) and not pd.isna(no_gap_quality):
        quality_drop = (no_gap_quality - gap_quality) / no_gap_quality * 100
        if quality_drop > 20:
            patterns.append({
                "type": "gap_effect",
                "description": f"Quality scores drop {quality_drop:.0f}% after gaps > 7 days",
                "severity": "warning"
            })
    
    source_quality = df.groupby("source")["quality_score"].mean().sort_values()
    if len(source_quality) > 1:
        worst_source = source_quality.index[0]
        best_source = source_quality.index[-1]
        quality_diff = source_quality[best_source] - source_quality[worst_source]
        if quality_diff > 0.15:
            patterns.append({
                "type": "source_disparity",
                "description": f"Source '{worst_source}' has {quality_diff*100:.0f}% lower quality than '{best_source}'",
                "severity": "warning"
            })
    
    return patterns

def generate_quality_insights(df: pd.DataFrame, stats: Dict[str, Any], patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    insights = []
    
    if stats.get("overall_quality"):
        if stats["overall_quality"] < 0.65:
            insights.append({
                "priority": 1,
                "type": "alert",
                "title": "Low Overall Quality",
                "description": f"Average quality score is {stats['overall_quality']:.2f}, close to rejection threshold",
                "recommendation": "Review data collection procedures and consider adjusting validation rules"
            })
    
    if stats.get("near_threshold_count", 0) > len(df) * 0.1:
        insights.append({
            "priority": 2,
            "type": "warning",
            "title": "Many Borderline Measurements",
            "description": f"{stats['near_threshold_count']} measurements are near the rejection threshold",
            "recommendation": "These measurements may be unreliable - consider reviewing the threshold or improving data quality"
        })
    
    weakest_component = None
    weakest_score = 1.0
    for component in ["safety", "plausibility", "consistency", "reliability"]:
        component_mean = stats.get(f"{component}_mean")
        if component_mean and component_mean < weakest_score:
            weakest_score = component_mean
            weakest_component = component
    
    if weakest_component and weakest_score < 0.7:
        insights.append({
            "priority": 3,
            "type": "info",
            "title": f"Weak {weakest_component.capitalize()} Scores",
            "description": f"{weakest_component.capitalize()} component averages {weakest_score:.2f}",
            "recommendation": f"Focus on improving {weakest_component} to increase overall quality"
        })
    
    for pattern in patterns:
        if pattern["severity"] == "warning":
            insights.append({
                "priority": 4,
                "type": pattern["severity"],
                "title": pattern["type"].replace("_", " ").title(),
                "description": pattern["description"],
                "recommendation": "Consider adjusting processing parameters or data collection timing"
            })
    
    if stats.get("acceptance_rate", 0) < 90:
        insights.append({
            "priority": 5,
            "type": "alert",
            "title": "Low Acceptance Rate",
            "description": f"Only {stats['acceptance_rate']:.1f}% of measurements are accepted",
            "recommendation": "Review rejection reasons and consider adjusting quality thresholds"
        })
    
    insights.sort(key=lambda x: x["priority"])
    
    return insights[:5]

def analyze_rejection_reasons(df: pd.DataFrame) -> Dict[str, Any]:
    rejected_df = df[df["accepted"] == False]
    
    if rejected_df.empty:
        return {"total_rejections": 0, "reasons": {}}
    
    reason_counts = rejected_df["rejection_reason"].value_counts().to_dict()
    
    reason_quality = {}
    for reason in reason_counts.keys():
        reason_df = rejected_df[rejected_df["rejection_reason"] == reason]
        if not reason_df["quality_score"].isna().all():
            reason_quality[reason] = {
                "count": reason_counts[reason],
                "avg_quality": reason_df["quality_score"].mean(),
                "min_quality": reason_df["quality_score"].min(),
                "max_quality": reason_df["quality_score"].max(),
            }
    
    component_failures = defaultdict(int)
    for _, row in rejected_df.iterrows():
        if row["quality_score"] and row["quality_score"] < 0.6:
            min_component = None
            min_score = 1.0
            for comp in ["safety", "plausibility", "consistency", "reliability"]:
                if row[comp] and row[comp] < min_score:
                    min_score = row[comp]
                    min_component = comp
            if min_component:
                component_failures[min_component] += 1
    
    return {
        "total_rejections": len(rejected_df),
        "reasons": reason_quality,
        "component_failures": dict(component_failures),
        "rejection_rate": len(rejected_df) / len(df) * 100 if len(df) > 0 else 0
    }

def analyze_source_quality(df: pd.DataFrame) -> Dict[str, Any]:
    source_stats = {}
    
    for source in df["source"].unique():
        source_df = df[df["source"] == source]
        
        source_stats[source] = {
            "count": len(source_df),
            "acceptance_rate": len(source_df[source_df["accepted"] == True]) / len(source_df) * 100,
            "avg_quality": source_df["quality_score"].mean() if not source_df["quality_score"].isna().all() else None,
            "quality_std": source_df["quality_score"].std() if not source_df["quality_score"].isna().all() else None,
        }
        
        for component in ["safety", "plausibility", "consistency", "reliability"]:
            if component in source_df.columns and not source_df[component].isna().all():
                source_stats[source][f"{component}_mean"] = source_df[component].mean()
    
    return source_stats

def calculate_quality_trends(df: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    if df.empty or "quality_score" not in df.columns:
        return pd.DataFrame()
    
    df = df.copy()
    df = df.set_index("timestamp").sort_index()
    
    df["quality_ma"] = df["quality_score"].rolling(window=window, min_periods=1).mean()
    df["quality_std_rolling"] = df["quality_score"].rolling(window=window, min_periods=1).std()
    
    for component in ["safety", "plausibility", "consistency", "reliability"]:
        if component in df.columns:
            df[f"{component}_ma"] = df[component].rolling(window=window, min_periods=1).mean()
    
    df["quality_trend"] = df["quality_ma"].diff()
    df["improving"] = df["quality_trend"] > 0.01
    df["declining"] = df["quality_trend"] < -0.01
    
    return df.reset_index()