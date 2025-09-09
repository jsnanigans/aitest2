#!/usr/bin/env python3

import json
import time
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
from src.core import get_logger

logger = get_logger(__name__)


class DebugOutputManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.source_type_trust_scores = config.get("source_type_trust_scores", {})
        self.output_individual_files = config.get("output_individual_files", True)
        self.individual_output_dir = None
        
        if self.output_individual_files:
            self.individual_output_dir = Path(config.get("individual_output_dir", "output/users"))
            self.individual_output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Individual user files will be saved to: {self.individual_output_dir}")
            
    def save_individual_user(self, user_id: str, user_stats: Dict[str, Any]):
        if not self.output_individual_files or not self.individual_output_dir:
            return
            
        user_filename = self.individual_output_dir / f"{user_id}.json"
        with open(user_filename, "w", encoding="utf-8") as user_file:
            json.dump(user_stats, user_file, indent=2, default=str)
            
    def calculate_data_quality(self, stats: Dict[str, Any], readings: List[Dict[str, Any]]) -> Dict[str, Any]:
        quality_metrics = {
            "total_readings": stats.get("total_readings", 0),
            "outlier_count": stats.get("outliers", 0),
            "outlier_percentage": round(stats.get("outliers", 0) / stats.get("total_readings", 1) * 100, 2) if stats.get("total_readings", 0) > 0 else 0,
            "low_confidence_count": sum(1 for c in stats.get("confidence_scores", []) if c < 0.5),
            "high_confidence_count": sum(1 for c in stats.get("confidence_scores", []) if c >= 0.9),
        }
        
        if len(readings) > 1:
            gaps = []
            for i in range(1, len(readings)):
                prev_date = self._parse_datetime(readings[i-1]["date"])
                curr_date = self._parse_datetime(readings[i]["date"])
                gap_days = (curr_date - prev_date).days
                if gap_days > 1:
                    gaps.append({
                        "from": readings[i-1]["date"],
                        "to": readings[i]["date"],
                        "gap_days": gap_days
                    })
                    
            quality_metrics["data_gaps"] = gaps[:5] if gaps else []
            quality_metrics["max_gap_days"] = max([g["gap_days"] for g in gaps]) if gaps else 0
            quality_metrics["total_gaps_gt_7days"] = sum(1 for g in gaps if g["gap_days"] > 7)
            
        if "source_type_breakdown" in stats:
            total_readings = sum(stats["source_type_breakdown"].values())
            quality_metrics["source_reliability"] = {}
            for src, count in stats["source_type_breakdown"].items():
                trust_score = self.source_type_trust_scores.get(src, 0.5)
                quality_metrics["source_reliability"][src] = {
                    "count": count,
                    "percentage": round(count / total_readings * 100, 2),
                    "trust_score": trust_score
                }
                
        if len(readings) > 1:
            weights = [r["weight"] for r in readings]
            weight_changes = [abs(weights[i] - weights[i-1]) for i in range(1, len(weights))]
            if weight_changes:
                quality_metrics["avg_absolute_change"] = round(float(np.mean(weight_changes)), 3)
                quality_metrics["max_single_change"] = round(max(weight_changes), 2)
                quality_metrics["sudden_changes_gt_2kg"] = sum(1 for c in weight_changes if c > 2.0)
                quality_metrics["sudden_changes_gt_5kg"] = sum(1 for c in weight_changes if c > 5.0)
                
        return quality_metrics
        
    def _parse_datetime(self, dt_string: str) -> datetime:
        try:
            return datetime.fromisoformat(dt_string.replace("Z", "+00:00"))
        except:
            try:
                return datetime.strptime(dt_string, "%Y-%m-%d %H:%M:%S")
            except:
                return datetime.strptime(dt_string.split(".")[0], "%Y-%m-%dT%H:%M:%S")
                
    def create_comprehensive_output(
        self,
        results: Dict[str, Any],
        metadata: Dict[str, Any],
        processing_stats: Dict[str, Any]
    ) -> Dict[str, Any]:
        debug_output = {
            "metadata": metadata,
            "global_statistics": processing_stats,
            "aggregate_quality_metrics": {},
            "users": results
        }
        
        if results:
            all_outlier_counts = [u.get("outliers", 0) for u in results.values()]
            all_confidence_scores = []
            all_weight_ranges = []
            total_kalman_errors = 0
            
            for user_data in results.values():
                if "average_confidence" in user_data:
                    all_confidence_scores.append(user_data["average_confidence"])
                if "weight_range" in user_data:
                    all_weight_ranges.append(user_data["weight_range"])
                if "kalman_errors" in user_data:
                    total_kalman_errors += len(user_data["kalman_errors"])
                    
            debug_output["aggregate_quality_metrics"] = {
                "avg_outliers_per_user": round(float(np.mean(all_outlier_counts)), 2) if all_outlier_counts else 0,
                "max_outliers_per_user": max(all_outlier_counts) if all_outlier_counts else 0,
                "users_with_outliers": sum(1 for c in all_outlier_counts if c > 0),
                "avg_confidence_across_users": round(float(np.mean(all_confidence_scores)), 3) if all_confidence_scores else 0,
                "avg_weight_range": round(float(np.mean(all_weight_ranges)), 2) if all_weight_ranges else 0,
                "max_weight_range": round(max(all_weight_ranges), 2) if all_weight_ranges else 0,
                "total_kalman_errors": total_kalman_errors
            }
            
        return debug_output
        
    def save_results(self, output_file: str, debug_output: Dict[str, Any]):
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(debug_output, f, indent=2, default=str)


class ProgressReporter:
    def __init__(self):
        self.last_update_time = time.time()
        self.start_time = time.time()
        
    def print_startup_info(self, config: Dict[str, Any], input_file: str, output_dir: Path, timestamp: str, has_matplotlib: bool):
        specific_user_ids = config.get("specific_user_ids", [])
        specific_user_set = set(specific_user_ids) if specific_user_ids else None
        min_readings = config.get("min_readings_per_user", 10)
        max_users = config.get("process_max_users", 0)
        max_viz = config.get("max_visualizations", 20)
        skip_first_users = config.get("skip_first_users", 0)
        
        print(f"\n{'='*80}")
        print("STREAM PROCESSOR - Starting Analysis")
        print(f"{'='*80}")
        print(f"Input file:        {input_file}")
        print(f"Output directory:  {output_dir}")
        print(f"Timestamp:         {timestamp}")
        print(f"{'─'*80}")
        print("Configuration:")
        print(f"  • Kalman filtering:     {'✓' if config.get('enable_kalman', True) else '✗'}")
        if config.get("enable_kalman", True):
            print("  • Filter type:          Custom 2D with trend tracking")
        print(
            f"  • Visualization:        {'✓' if config.get('enable_visualization', True) and has_matplotlib else '✗'}"
        )
        print(f"  • Min readings/user:    {min_readings}")
        if specific_user_set:
            print(f"  • Specific user IDs:    {len(specific_user_ids)} users only")
        else:
            print(f"  • Skip first users:     {skip_first_users if skip_first_users else 'none'}")
        print(f"  • Max users to process: {max_users if max_users else 'unlimited'}")
        print(f"  • Max visualizations:   {max_viz}")
        print(f"{'='*80}")
        print("Processing data...\n")
        
    def update_progress(
        self,
        users_processed: int,
        users_skipped: int,
        total_rows: int,
        users_with_kalman: int,
        current_user: str = None,
        force: bool = False
    ):
        current_time = time.time()
        if not force and current_time - self.last_update_time < 0.5:
            return
            
        elapsed = current_time - self.start_time
        
        if current_user:
            rows_per_sec = total_rows / elapsed if elapsed > 0 else 0
            progress_msg = (
                f"Users: {users_processed:5d} | "
                f"Rows: {total_rows:7d} | "
                f"Current: {current_user[:8]}... | "
                f"Rate: {rows_per_sec:6.0f} r/s"
            )
        else:
            rate = users_processed / elapsed if elapsed > 0 else 0
            kalman_pct = (users_with_kalman / users_processed * 100) if users_processed > 0 else 0
            progress_msg = (
                f"Users: {users_processed:5d} | "
                f"Skipped: {users_skipped:4d} | "
                f"Rows: {total_rows:7d} | "
                f"Kalman: {users_with_kalman:4d} ({kalman_pct:4.1f}%) | "
                f"Rate: {rate:6.1f} u/s"
            )
            
        sys.stdout.write(progress_msg + "\n")
        sys.stdout.flush()
        self.last_update_time = current_time
        
    def print_summary(
        self,
        total_rows: int,
        users_processed: int,
        users_with_kalman: int,
        elapsed: float,
        output_file: str,
        viz_dir: Path,
        viz_count: int,
        individual_output_dir: Path,
        specific_user_set: set = None,
        users_skipped_not_in_list: int = 0
    ):
        print(f"\n{'='*80}")
        print("PROCESSING COMPLETE - Summary")
        print(f"{'='*80}")
        print("Performance Metrics:")
        print(f"  • Total rows processed:  {total_rows:,}")
        print(f"  • Users processed:       {users_processed:,}")
        if specific_user_set:
            print(f"  • Users not in list:     {users_skipped_not_in_list:,} (skipped)")
        print(f"  • Processing time:       {elapsed:.2f} seconds")
        print(f"  • Processing rate:       {users_processed/elapsed:.1f} users/sec")
        print(f"  • Rows per second:       {total_rows/elapsed:.0f} rows/sec")
        print(f"{'─'*80}")
        print("Analysis Results:")
        if users_processed > 0:
            print(
                f"  • Users with Kalman:     {users_with_kalman:,} ({users_with_kalman/users_processed*100:.1f}%)"
            )
        else:
            print(f"  • Users with Kalman:     {users_with_kalman:,}")
            
        if users_processed > 0:
            avg_readings = total_rows / users_processed
            print(f"  • Avg readings/user:     {avg_readings:.1f}")
            
        print(f"{'─'*80}")
        print("Output Files:")
        print(f"  • JSON results:          {output_file}")
        
        if viz_dir:
            print(f"  • Visualizations:        {viz_count} dashboards in {viz_dir}")
            
        print("  • Application log:       output/app.log")
        if individual_output_dir:
            print(f"  • Individual user files: {individual_output_dir}/ ({users_processed} files)")
        print(f"{'='*80}\n")