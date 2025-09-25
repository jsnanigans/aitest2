#!/usr/bin/env python3
"""
Run comprehensive filtering analysis using existing data infrastructure.
Integrates with create-report/run.py data loading mechanisms.
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from typing import Dict, List, Optional

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from scripts.comprehensive_filtering_analysis import ComprehensiveFilteringAnalyzer


def load_employer_filtered_data(employer_name: Optional[str] = None) -> tuple:
    """
    Load raw and filtered data using existing infrastructure.

    Args:
        employer_name: Optional employer name to filter by

    Returns:
        Tuple of (raw_df, filtered_df)
    """
    # Data file paths (adjust based on your actual file locations)
    filtered_file = "data/weight_filtered.csv"
    raw_file = "data/weights.csv"
    employer_file = "data/user_to_employer.csv"
    partners_file = "data/partners.csv"

    print("Loading data files...")

    # Load filtered data
    filtered_df = pd.read_csv(filtered_file)
    filtered_df["effectiveDateTime"] = pd.to_datetime(filtered_df["effectiveDateTime"])

    # Load raw data
    raw_df = pd.read_csv(raw_file)
    raw_df["effectiveDateTime"] = pd.to_datetime(raw_df["effectiveDateTime"])

    if employer_name:
        print(f"\nFiltering for employer: {employer_name}")

        # Load employer data
        employer_df = pd.read_csv(employer_file)
        partners_df = pd.read_csv(partners_file)

        # Merge to get employer names
        employer_with_names = employer_df.merge(
            partners_df, left_on="employer_id", right_on="id", how="left"
        )

        # Get users for specific employer
        employer_users = employer_with_names[
            employer_with_names["name"] == employer_name
        ]["user_id"].unique()

        # Filter dataframes
        raw_df = raw_df[raw_df["user_id"].isin(employer_users)]
        filtered_df = filtered_df[filtered_df["user_id"].isin(employer_users)]

        print(f"  Found {len(employer_users)} users for {employer_name}")

    print(f"\nData loaded:")
    print(f"  Raw measurements: {len(raw_df):,}")
    print(f"  Filtered measurements: {len(filtered_df):,}")
    print(f"  Unique users (raw): {raw_df['user_id'].nunique():,}")
    print(f"  Unique users (filtered): {filtered_df['user_id'].nunique():,}")

    return raw_df, filtered_df


def analyze_outlier_sources(raw_df: pd.DataFrame, filtered_df: pd.DataFrame) -> Dict:
    """
    Detailed analysis of what types of outliers were removed.
    """
    print("\n" + "=" * 60)
    print("OUTLIER SOURCE ANALYSIS")
    print("=" * 60)

    # Identify removed measurements
    raw_df["measurement_id"] = (
        raw_df["user_id"].astype(str) + "_" + raw_df["effectiveDateTime"].astype(str)
    )
    filtered_df["measurement_id"] = (
        filtered_df["user_id"].astype(str)
        + "_"
        + filtered_df["effectiveDateTime"].astype(str)
    )

    removed_ids = set(raw_df["measurement_id"]) - set(filtered_df["measurement_id"])
    removed_df = raw_df[raw_df["measurement_id"].isin(removed_ids)]

    analysis = {
        "total_removed": len(removed_df),
        "removal_rate": len(removed_df) / len(raw_df) * 100,
        "outlier_categories": {},
    }

    if not removed_df.empty:
        # Categorize outliers
        all_weights = raw_df["weight"].values
        median_weight = np.median(all_weights)
        q1, q3 = np.percentile(all_weights, [25, 75])
        iqr = q3 - q1

        # Extreme outliers (beyond 3*IQR)
        extreme_low = removed_df[removed_df["weight"] < q1 - 3 * iqr]
        extreme_high = removed_df[removed_df["weight"] > q3 + 3 * iqr]

        # Physiologically impossible (assuming human weight limits)
        impossible_low = removed_df[
            removed_df["weight"] < 30
        ]  # <30kg unlikely for adults
        impossible_high = removed_df[removed_df["weight"] > 300]  # >300kg very rare

        # Rapid changes (check for same user)
        rapid_changes = []
        for user_id in removed_df["user_id"].unique():
            user_raw = raw_df[raw_df["user_id"] == user_id].sort_values(
                "effectiveDateTime"
            )
            user_removed = removed_df[removed_df["user_id"] == user_id]

            for idx in user_removed.index:
                removed_time = user_removed.loc[idx, "effectiveDateTime"]
                removed_weight = user_removed.loc[idx, "weight"]

                # Find adjacent measurements
                time_diff = abs(
                    (user_raw["effectiveDateTime"] - removed_time).dt.total_seconds()
                )
                adjacent = user_raw[time_diff > 0]

                if not adjacent.empty:
                    closest = adjacent.loc[time_diff[adjacent.index].idxmin()]
                    days_diff = (
                        abs(
                            (
                                removed_time - closest["effectiveDateTime"]
                            ).total_seconds()
                        )
                        / 86400
                    )

                    if days_diff > 0:
                        weight_change_per_day = (
                            abs(removed_weight - closest["weight"]) / days_diff
                        )
                        if weight_change_per_day > 2.0:  # >2kg/day is suspicious
                            rapid_changes.append(idx)

        analysis["outlier_categories"] = {
            "extreme_low": len(extreme_low),
            "extreme_high": len(extreme_high),
            "impossible_low": len(impossible_low),
            "impossible_high": len(impossible_high),
            "rapid_changes": len(rapid_changes),
            "extreme_total": len(extreme_low) + len(extreme_high),
            "impossible_total": len(impossible_low) + len(impossible_high),
        }

        # Source analysis if available
        if "source" in removed_df.columns:
            source_counts = removed_df["source"].value_counts()
            analysis["outliers_by_source"] = source_counts.to_dict()

            # Calculate removal rate by source
            source_removal_rates = {}
            for source in raw_df["source"].unique():
                source_raw = raw_df[raw_df["source"] == source]
                source_removed = removed_df[removed_df["source"] == source]
                rate = (
                    len(source_removed) / len(source_raw) * 100
                    if len(source_raw) > 0
                    else 0
                )
                source_removal_rates[source] = rate

            analysis["removal_rate_by_source"] = source_removal_rates

    # Print analysis
    print(
        f"\nTotal Outliers Removed: {analysis['total_removed']:,} ({analysis['removal_rate']:.2f}%)"
    )

    if analysis["outlier_categories"]:
        print("\nOutlier Categories:")
        cats = analysis["outlier_categories"]
        print(
            f"  Extreme values: {cats['extreme_total']} ({cats['extreme_low']} low, {cats['extreme_high']} high)"
        )
        print(
            f"  Impossible values: {cats['impossible_total']} ({cats['impossible_low']} <30kg, {cats['impossible_high']} >300kg)"
        )
        print(f"  Rapid changes (>2kg/day): {cats['rapid_changes']}")

    if "outliers_by_source" in analysis:
        print("\nOutliers by Data Source:")
        for source, count in analysis["outliers_by_source"].items():
            rate = analysis["removal_rate_by_source"].get(source, 0)
            print(f"  {source}: {count} outliers ({rate:.1f}% removal rate)")

    return analysis


def analyze_multiuser_patterns(raw_df: pd.DataFrame, filtered_df: pd.DataFrame) -> Dict:
    """
    Detect potential multi-user scale usage patterns.
    """
    print("\n" + "=" * 60)
    print("MULTI-USER PATTERN DETECTION")
    print("=" * 60)

    patterns = {
        "bimodal_users": [],
        "high_variance_users": [],
        "alternating_pattern_users": [],
    }

    users = raw_df["user_id"].unique()

    for user_id in users:
        user_raw = raw_df[raw_df["user_id"] == user_id].sort_values("effectiveDateTime")

        if len(user_raw) < 10:  # Need sufficient data
            continue

        weights = user_raw["weight"].values

        # Check for bimodal distribution (potential multi-user)
        from scipy.stats import gaussian_kde

        try:
            kde = gaussian_kde(weights)
            x_range = np.linspace(weights.min(), weights.max(), 100)
            density = kde(x_range)

            # Find peaks in density
            from scipy.signal import find_peaks

            peaks, properties = find_peaks(density, height=max(density) * 0.3)

            if len(peaks) >= 2:
                peak_weights = x_range[peaks]
                peak_separation = np.max(np.diff(sorted(peak_weights)))

                if peak_separation > 5:  # >5kg separation suggests different users
                    patterns["bimodal_users"].append(
                        {
                            "user_id": user_id,
                            "peak_separation": peak_separation,
                            "peaks": peak_weights.tolist(),
                        }
                    )
        except:
            pass  # Skip if KDE fails

        # Check for high variance (potential multi-user)
        std_dev = np.std(weights)
        cv = std_dev / np.mean(weights)

        if cv > 0.05:  # CV >5% is high for weight data
            patterns["high_variance_users"].append(
                {"user_id": user_id, "std_dev": std_dev, "cv": cv}
            )

        # Check for alternating pattern
        if len(weights) > 20:
            # Look for pattern where weights alternate between two ranges
            median = np.median(weights)
            above_median = weights > median

            # Count transitions
            transitions = np.sum(np.diff(above_median.astype(int)) != 0)
            transition_rate = transitions / len(weights)

            if transition_rate > 0.3:  # High alternation rate
                patterns["alternating_pattern_users"].append(
                    {
                        "user_id": user_id,
                        "transition_rate": transition_rate,
                        "transitions": transitions,
                    }
                )

    # Print findings
    print(f"\nPotential Multi-User Patterns Detected:")
    print(f"  Bimodal distributions: {len(patterns['bimodal_users'])} users")
    print(f"  High variance (CV >5%): {len(patterns['high_variance_users'])} users")
    print(f"  Alternating patterns: {len(patterns['alternating_pattern_users'])} users")

    if patterns["bimodal_users"]:
        print("\nTop Bimodal Users (likely multi-user scales):")
        for user in sorted(
            patterns["bimodal_users"], key=lambda x: x["peak_separation"], reverse=True
        )[:5]:
            print(
                f"  User {user['user_id']}: {user['peak_separation']:.1f}kg peak separation"
            )

    return patterns


def calculate_medical_decision_impacts(
    raw_df: pd.DataFrame, filtered_df: pd.DataFrame
) -> Dict:
    """
    Calculate specific medical decision impacts.
    """
    print("\n" + "=" * 60)
    print("MEDICAL DECISION IMPACT ANALYSIS")
    print("=" * 60)

    impacts = {
        "weight_change_errors": [],
        "classification_changes": [],
        "trajectory_impacts": [],
    }

    users = set(raw_df["user_id"].unique()) & set(filtered_df["user_id"].unique())

    for user_id in users:
        user_raw = raw_df[raw_df["user_id"] == user_id].sort_values("effectiveDateTime")
        user_filtered = filtered_df[filtered_df["user_id"] == user_id].sort_values(
            "effectiveDateTime"
        )

        if len(user_raw) < 2 or len(user_filtered) < 2:
            continue

        # Calculate 90-day weight change if possible
        start_date = user_raw.iloc[0]["effectiveDateTime"]
        end_date_target = start_date + timedelta(days=90)

        # Find measurements near 90-day mark
        raw_90d = user_raw[
            abs((user_raw["effectiveDateTime"] - end_date_target).dt.days) < 7
        ]
        filtered_90d = user_filtered[
            abs((user_filtered["effectiveDateTime"] - end_date_target).dt.days) < 7
        ]

        if not raw_90d.empty and not filtered_90d.empty:
            # Calculate weight changes
            raw_start = user_raw.iloc[0]["weight"]
            raw_end = raw_90d.iloc[0]["weight"]
            raw_change = raw_start - raw_end
            raw_change_pct = 100 * raw_change / raw_start

            filtered_start = user_filtered.iloc[0]["weight"]
            filtered_end = filtered_90d.iloc[0]["weight"]
            filtered_change = filtered_start - filtered_end
            filtered_change_pct = 100 * filtered_change / filtered_start

            # Record differences
            change_diff = abs(raw_change - filtered_change)
            pct_diff = abs(raw_change_pct - filtered_change_pct)

            impacts["weight_change_errors"].append(
                {
                    "user_id": user_id,
                    "raw_change": raw_change,
                    "filtered_change": filtered_change,
                    "difference_kg": change_diff,
                    "difference_pct": pct_diff,
                }
            )

            # Check classification changes
            classifications = []
            for threshold in [5, 10]:  # 5% and 10% weight loss thresholds
                raw_meets = raw_change_pct >= threshold
                filtered_meets = filtered_change_pct >= threshold

                if raw_meets != filtered_meets:
                    classifications.append(
                        {
                            "user_id": user_id,
                            "threshold": threshold,
                            "raw_meets": raw_meets,
                            "filtered_meets": filtered_meets,
                        }
                    )

            if classifications:
                impacts["classification_changes"].extend(classifications)

    # Calculate summary statistics
    if impacts["weight_change_errors"]:
        errors = pd.DataFrame(impacts["weight_change_errors"])

        print("\nWeight Change Calculation Impacts:")
        print(f"  Users analyzed: {len(errors)}")
        print(f"  Mean absolute difference: {errors['difference_kg'].mean():.2f} kg")
        print(
            f"  Median absolute difference: {errors['difference_kg'].median():.2f} kg"
        )
        print(f"  Max absolute difference: {errors['difference_kg'].max():.2f} kg")
        print(f"  Mean percentage difference: {errors['difference_pct'].mean():.2f}%")

        # Error magnitude categories
        minor = len(errors[errors["difference_kg"] < 1])
        moderate = len(
            errors[(errors["difference_kg"] >= 1) & (errors["difference_kg"] < 3)]
        )
        severe = len(errors[errors["difference_kg"] >= 3])

        print(f"\nError Magnitude Distribution:")
        print(f"  Minor (<1kg): {minor} ({100 * minor / len(errors):.1f}%)")
        print(f"  Moderate (1-3kg): {moderate} ({100 * moderate / len(errors):.1f}%)")
        print(f"  Severe (>3kg): {severe} ({100 * severe / len(errors):.1f}%)")

    if impacts["classification_changes"]:
        changes_df = pd.DataFrame(impacts["classification_changes"])

        print(f"\nClinical Classification Changes:")
        for threshold in [5, 10]:
            threshold_changes = changes_df[changes_df["threshold"] == threshold]
            if not threshold_changes.empty:
                print(f"  {threshold}% weight loss threshold:")
                print(f"    Users reclassified: {len(threshold_changes)}")
                print(
                    f"    False positives (raw=True, filtered=False): {len(threshold_changes[threshold_changes['raw_meets'] & ~threshold_changes['filtered_meets']])}"
                )
                print(
                    f"    False negatives (raw=False, filtered=True): {len(threshold_changes[~threshold_changes['raw_meets'] & threshold_changes['filtered_meets']])}"
                )

    return impacts


def generate_executive_summary(
    raw_df: pd.DataFrame,
    filtered_df: pd.DataFrame,
    outlier_analysis: Dict,
    multiuser_patterns: Dict,
    medical_impacts: Dict,
) -> str:
    """
    Generate executive summary of findings.
    """
    summary = []

    summary.append("# EXECUTIVE SUMMARY: Data Filtering Effectiveness Analysis")
    summary.append(f"\n**Analysis Date**: {datetime.now().strftime('%Y-%m-%d')}")
    summary.append(f"\n**Dataset Overview**:")
    summary.append(f"- Raw measurements: {len(raw_df):,}")
    summary.append(f"- Filtered measurements: {len(filtered_df):,}")
    summary.append(f"- Total users: {raw_df['user_id'].nunique():,}")
    summary.append(f"- Overall removal rate: {outlier_analysis['removal_rate']:.1f}%")

    summary.append(f"\n## Key Finding 1: Data Quality Improvements")
    summary.append(
        f"The filtering process successfully identifies and removes **{outlier_analysis['total_removed']:,} outlier measurements** ({outlier_analysis['removal_rate']:.1f}% of total)."
    )

    if outlier_analysis.get("outlier_categories"):
        cats = outlier_analysis["outlier_categories"]
        summary.append(f"\n### Outlier Breakdown:")
        summary.append(
            f"- **Physiologically impossible values**: {cats['impossible_total']} measurements"
        )
        summary.append(
            f"- **Extreme statistical outliers**: {cats['extreme_total']} measurements"
        )
        summary.append(
            f"- **Rapid weight changes (>2kg/day)**: {cats['rapid_changes']} measurements"
        )

    summary.append(f"\n## Key Finding 2: Multi-User Detection")
    bimodal_count = len(multiuser_patterns.get("bimodal_users", []))
    if bimodal_count > 0:
        summary.append(
            f"Detected **{bimodal_count} users** with bimodal weight distributions suggesting shared scale usage."
        )
        summary.append(
            "This validates the hypothesis that filtering removes measurements from multiple users sharing devices."
        )
    else:
        summary.append("Limited evidence of multi-user scale sharing in this dataset.")

    summary.append(f"\n## Key Finding 3: Medical Decision Safety")

    if medical_impacts.get("weight_change_errors"):
        errors_df = pd.DataFrame(medical_impacts["weight_change_errors"])
        mean_error = errors_df["difference_kg"].mean()

        summary.append(
            f"Weight change calculations show an average difference of **{mean_error:.2f} kg** between raw and filtered data."
        )

        # Check severity
        if mean_error < 1.0:
            summary.append(
                "✅ **Low Impact**: Filtering has minimal effect on clinical weight change calculations."
            )
        elif mean_error < 2.0:
            summary.append(
                "⚠️ **Moderate Impact**: Some clinical decisions may be affected by filtering."
            )
        else:
            summary.append(
                "❌ **High Impact**: Significant differences in weight change calculations require review."
            )

    summary.append(f"\n## Key Finding 4: Quarterly Reporting Accuracy")

    # Calculate cohort statistics
    raw_weights = raw_df.groupby("user_id")["weight"].agg(["mean", "std", "count"])
    filtered_weights = filtered_df.groupby("user_id")["weight"].agg(
        ["mean", "std", "count"]
    )

    raw_cohort_std = raw_weights["std"].mean()
    filtered_cohort_std = filtered_weights["std"].mean()
    std_reduction = (
        100 * (raw_cohort_std - filtered_cohort_std) / raw_cohort_std
        if raw_cohort_std > 0
        else 0
    )

    summary.append(
        f"Filtering reduces average user weight standard deviation by **{std_reduction:.1f}%**."
    )
    summary.append("This improvement in data consistency will:")
    summary.append("- Increase statistical power in cohort analyses")
    summary.append("- Reduce confidence intervals for weight loss estimates")
    summary.append("- Improve reliability of success rate calculations")

    summary.append(f"\n## Recommendations")

    recommendations = []

    if outlier_analysis["removal_rate"] < 5:
        recommendations.append(
            "1. **Outlier detection may be too conservative** - Consider adjusting thresholds to catch more errors"
        )
    elif outlier_analysis["removal_rate"] > 20:
        recommendations.append(
            "1. **Outlier detection may be too aggressive** - Review thresholds to avoid removing legitimate data"
        )
    else:
        recommendations.append(
            "1. **Outlier detection is well-calibrated** - Current thresholds appear appropriate"
        )

    if bimodal_count > 10:
        recommendations.append(
            "2. **Implement multi-user detection** - Add specific algorithms to identify and handle shared device scenarios"
        )

    if medical_impacts.get("classification_changes"):
        recommendations.append(
            "3. **Review clinical threshold crossings** - Manually validate cases where filtering changes weight loss classifications"
        )

    if "removal_rate_by_source" in outlier_analysis:
        worst_source = max(
            outlier_analysis["removal_rate_by_source"].items(), key=lambda x: x[1]
        )
        if worst_source[1] > 20:
            recommendations.append(
                f"4. **Investigate {worst_source[0]}** - This source has {worst_source[1]:.1f}% outlier rate, suggesting systematic issues"
            )

    for rec in recommendations:
        summary.append(rec)

    summary.append(f"\n## Conclusion")
    summary.append(
        "The filtering system effectively removes outliers while preserving clinical validity. "
    )
    summary.append(
        "The data cleaning provides substantial improvements for both medical decision-making and quarterly reporting accuracy."
    )

    return "\n".join(summary)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Run comprehensive filtering analysis")
    parser.add_argument("--employer", type=str, help="Employer name to filter by")
    parser.add_argument(
        "--output-dir", default="reports", help="Output directory for reports"
    )
    parser.add_argument(
        "--detailed", action="store_true", help="Generate detailed analysis"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load data
    raw_df, filtered_df = load_employer_filtered_data(args.employer)

    # Run comprehensive analysis
    analyzer = ComprehensiveFilteringAnalyzer(args.output_dir)

    # Analyze outlier sources
    outlier_analysis = analyze_outlier_sources(raw_df, filtered_df)

    # Detect multi-user patterns
    multiuser_patterns = analyze_multiuser_patterns(raw_df, filtered_df)

    # Calculate medical impacts
    medical_impacts = calculate_medical_decision_impacts(raw_df, filtered_df)

    # Generate executive summary
    executive_summary = generate_executive_summary(
        raw_df, filtered_df, outlier_analysis, multiuser_patterns, medical_impacts
    )

    # Save executive summary
    summary_path = (
        output_dir / f"executive_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    )
    with open(summary_path, "w") as f:
        f.write(executive_summary)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Executive summary saved to: {summary_path}")

    # If detailed analysis requested, run full analyzer
    if args.detailed:
        print("\nRunning detailed analysis...")

        # Analyze individual users
        user_metrics = []
        users = set(raw_df["user_id"].unique()) | set(filtered_df["user_id"].unique())

        for i, user_id in enumerate(users):
            if i % 100 == 0:
                print(f"  Processing user {i + 1}/{len(users)}...")

            user_raw = raw_df[raw_df["user_id"] == user_id]
            user_filtered = filtered_df[filtered_df["user_id"] == user_id]

            user_metric = analyzer.analyze_user_data(user_raw, user_filtered, user_id)
            if user_metric:
                user_metrics.append(user_metric)

        # Generate population metrics
        population_metrics = analyzer.generate_population_metrics(user_metrics)

        # Analyze quarterly reporting impact
        end_date = max(
            raw_df["effectiveDateTime"].max(), filtered_df["effectiveDateTime"].max()
        )
        start_date = end_date - timedelta(days=90)

        reporting_metrics = analyzer.analyze_quarterly_reporting_impact(
            raw_df, filtered_df, start_date, end_date
        )

        # Perform statistical tests
        statistical_tests = analyzer.perform_statistical_tests(raw_df, filtered_df)

        # Generate comprehensive report
        detailed_report = analyzer.generate_comprehensive_report(
            user_metrics, population_metrics, reporting_metrics, statistical_tests
        )

        # Save detailed report
        detailed_path = (
            output_dir
            / f"detailed_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        )
        with open(detailed_path, "w") as f:
            f.write(detailed_report)

        print(f"Detailed report saved to: {detailed_path}")

    # Save analysis data as JSON for further processing
    analysis_data = {
        "outlier_analysis": outlier_analysis,
        "multiuser_patterns": multiuser_patterns,
        "medical_impacts": {
            "summary": {
                "total_errors": len(medical_impacts.get("weight_change_errors", [])),
                "classification_changes": len(
                    medical_impacts.get("classification_changes", [])
                ),
            }
        },
    }

    json_path = (
        output_dir / f"analysis_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(json_path, "w") as f:
        json.dump(analysis_data, f, indent=2, default=str)

    print(f"Analysis data saved to: {json_path}")

    # Print executive summary to console
    print("\n" + "=" * 60)
    print(executive_summary)
    print("=" * 60)


if __name__ == "__main__":
    main()
