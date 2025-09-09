#!/usr/bin/env python3
"""
Unified streaming data processor with configurable features:
- Kalman filtering with rate tracking
- Comprehensive visualization (single image per user)
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict

from src.core import get_logger, load_config, setup_logging
from src.processing import (
    StreamingCSVReader,
    UserFilteredReader,
    UserProcessor,
    KalmanProcessor,
    VisualizationManager,
    DebugOutputManager,
    ProgressReporter
)
from src.filters.enhanced_validation_gate import EnhancedValidationGate
from src.filters.multimodal_detector import MultimodalDetector

setup_logging()
logger = get_logger(__name__)

timestamp = "yes"


class UnifiedStreamProcessor:
    """Unified processor with Kalman filter and visualization."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.user_processor = UserProcessor(config)
        self.kalman_processor = KalmanProcessor(config)
        self.current_user_id = None
        self.kalman_initialized_with_baseline = False
        
        # Initialize enhanced validation if enabled
        self.enable_enhanced_validation = config.get('enable_enhanced_validation', True)
        if self.enable_enhanced_validation:
            validation_config = config.get('validation_gate', {
                'max_deviation_pct': 0.20,
                'same_day_threshold_kg': 10.0,
                'rapid_change_threshold_pct': 0.20,
                'rapid_change_hours': 24,
                'outlier_z_score': 3.0,
                'min_readings_for_stats': 10,
                'future_date_tolerance_days': 1,
                'duplicate_threshold_kg': 0.5
            })
            # Add gap threshold to match baseline gap detection
            validation_config['gap_threshold_days'] = config.get('baseline_gap_threshold_days', 14)
            self.validation_gate = EnhancedValidationGate(validation_config)
            logger.info(f"Enhanced validation enabled with config: {validation_config}")
        else:
            self.validation_gate = None
            logger.info("Enhanced validation disabled")
        
        # Initialize multimodal detector if enabled
        self.enable_multimodal_detection = config.get('enable_multimodal_detection', True)
        if self.enable_multimodal_detection:
            multimodal_config = config.get('multimodal_detector', {
                'min_readings_for_detection': 30,
                'cluster_gap_kg': 15.0,
                'min_cluster_size': 5,
                'max_components': 3,
                'enable_auto_split': True
            })
            self.multimodal_detector = MultimodalDetector(multimodal_config)
        else:
            self.multimodal_detector = None
        
        self.validation_stats = {
            'readings_validated': 0,
            'readings_rejected': 0,
            'readings_deduplicated': 0,
            'raw_readings_seen': 0
        }
        self.all_readings = []  # Track ALL readings for visualization

    def reset(self):
        """Reset processor state for new user."""
        self.user_processor.reset()
        # Reset Kalman processor for new user
        self.kalman_processor.filter_initialized = False
        self.kalman_processor.kalman_filter = None
        self.kalman_processor.kalman_results = []
        self.validation_stats = {
            'readings_validated': 0,
            'readings_rejected': 0,
            'readings_deduplicated': 0,
            'raw_readings_seen': 0
        }
        self.all_readings = []  # Reset all readings tracker
        if self.validation_gate and self.current_user_id:
            self.validation_gate.reset_user_context(self.current_user_id)

    def start_user(self, user_id: str):
        """Initialize processing for a new user."""
        self.reset()
        self.current_user_id = user_id
        self.user_processor.start_user(user_id)
        # Start Kalman immediately with default initialization
        self.kalman_processor.initialize_filter()
        self.kalman_initialized_with_baseline = True  # Allow immediate processing

    def process_reading(self, reading: Dict[str, Any]) -> bool:
        """Process a single reading with all enabled features."""
        
        # Track raw reading count
        self.validation_stats['raw_readings_seen'] += 1
        
        # Apply enhanced validation if enabled
        if self.validation_gate and self.current_user_id:
            # Convert raw CSV row to validation format
            # Handle NULL weights gracefully
            weight_str = reading.get('weight', 0)
            try:
                weight_val = float(weight_str) if weight_str and weight_str != 'NULL' else 0
            except (ValueError, TypeError):
                weight_val = 0
            
            # Handle NULL weights gracefully
            weight_str = reading.get('weight', 0)
            try:
                weight_val = float(weight_str) if weight_str and weight_str != 'NULL' else 0
            except (ValueError, TypeError):
                weight_val = 0
            
            validation_reading = {
                'date': reading.get('effectivDateTime', reading.get('date_str', reading.get('date'))),
                'weight': weight_val,
                'source': reading.get('source_type', reading.get('source', 'unknown'))
            }
            
            # Track this reading with its validation status
            reading_record = {
                'date': validation_reading['date'],
                'weight': validation_reading['weight'],
                'source': validation_reading['source'],
                'validated': False,
                'rejection_reason': None
            }
            
            # Check for duplicates
            if self.validation_gate.should_deduplicate(self.current_user_id, validation_reading):
                self.validation_stats['readings_deduplicated'] += 1
                reading_record['rejection_reason'] = 'duplicate'
                self.all_readings.append(reading_record)
                logger.debug(f"Deduplicated reading for {self.current_user_id}: {validation_reading['date']}")
                return False
            
            # Validate reading
            is_valid, reason = self.validation_gate.validate_reading(self.current_user_id, validation_reading)
            if not is_valid:
                self.validation_stats['readings_rejected'] += 1
                reading_record['rejection_reason'] = reason
                self.all_readings.append(reading_record)
                logger.debug(f"Rejected reading for {self.current_user_id}: {reason} - weight={validation_reading['weight']}kg")
                
                # Still process through Kalman for prediction/residual calculation
                # but it won't update the Kalman state
                if self.kalman_processor and self.kalman_processor.filter_initialized:
                    # Get prediction without updating filter
                    kalman_prediction = self.kalman_processor.get_current_prediction(
                        reading.get('effectivDateTime', reading.get('date_str', reading.get('date')))
                    )
                    if kalman_prediction:
                        # Calculate residual for rejected reading
                        # Safe weight conversion
                        weight_str = reading.get('weight', 0)
                        try:
                            weight_val = float(weight_str) if weight_str and weight_str != 'NULL' else 0
                        except (ValueError, TypeError):
                            weight_val = 0
                        residual = weight_val - kalman_prediction.get('predicted_weight', 0)
                        reading_record['kalman_prediction'] = kalman_prediction.get('predicted_weight')
                        reading_record['kalman_residual'] = residual
                        reading_record['measurement_accepted'] = False
                        
                        # Add to kalman results for visualization
                        # Include all required fields for rejected measurements
                        self.kalman_processor.kalman_results.append({
                            'date': reading_record['date'],
                            'measurement': weight_val,
                            'filtered_weight': kalman_prediction.get('predicted_weight'),
                            'innovation': residual,
                            'normalized_innovation': abs(residual) / kalman_prediction.get('uncertainty', 1.0),
                            'uncertainty': kalman_prediction.get('uncertainty', 1.0),
                            'measurement_accepted': False,
                            'rejection_reason': reason
                        })
                
                return False
            
            # Mark as validated
            reading_record['validated'] = True
            self.all_readings.append(reading_record)
            self.validation_stats['readings_validated'] += 1
        
        # If validation is disabled, track the reading anyway for visualization
        if not self.validation_gate:
            reading_record = {
                'date': reading.get('effectivDateTime', reading.get('date_str', reading.get('date'))),
                'weight': float(reading.get('weight', 0)) if reading.get('weight') else 0,
                'source': reading.get('source_type', reading.get('source', 'unknown')),
                'validated': True,  # All readings are "valid" if validation is disabled
                'rejection_reason': None
            }
            self.all_readings.append(reading_record)
        
        processed = self.user_processor.process_reading(reading)
        if processed is None:
            return False

        # Check if a new baseline was established (gap detected)
        if processed.get('new_baseline'):
            baseline_params = processed['new_baseline']
            if self.kalman_processor.reinitialize_filter(baseline_params):
                logger.info(f"Kalman filter reinitialized for user {self.current_user_id} after gap")
                # Also reset enhanced validation gate context to prevent stale baseline rejections
                if self.validation_gate:
                    self.validation_gate.reset_user_context(self.current_user_id)
                    logger.info(f"Enhanced validation gate context reset for user {self.current_user_id}")
        
        # Also check for large gaps even without baseline re-establishment
        elif processed.get('gap_days', 0) > 60:
            # Force Kalman reinitialization for very large gaps
            logger.debug(f"Very large gap ({processed['gap_days']} days) - forcing Kalman reset")
            if self.kalman_processor.kalman_filter:
                # Force reinitialization with current weight
                baseline_params = {
                    'success': True,
                    'baseline_weight': processed['weight'],
                    'measurement_variance': 2.0  # Higher variance for uncertainty
                }
                self.kalman_processor.reinitialize_filter(baseline_params)

        # Always process with Kalman (no more waiting for baseline)

        kalman_result = self.kalman_processor.process_measurement(
            processed["weight"],
            processed["date"],
            processed["date_str"],
            processed["source_type"]
        )

        if kalman_result:
            last_reading = self.user_processor.readings[-1]
            last_reading["kalman_filtered"] = kalman_result["filtered_weight"]
            last_reading["kalman_uncertainty"] = kalman_result.get(
                "uncertainty_weight", kalman_result.get("uncertainty")
            )
            if "trend_kg_per_day" in kalman_result:
                last_reading["kalman_trend"] = kalman_result["trend_kg_per_day"]

            if "kalman_errors" not in self.user_processor.stats:
                self.user_processor.stats["kalman_errors"] = []

        return True

    def finish_user(self) -> Dict[str, Any]:
        """Complete processing for current user and prepare final stats."""
        if not self.current_user_id:
            return {"total_readings": 0}

        stats = self.user_processor.finalize_stats()
        
        # Add raw reading count to stats for filtering decision
        stats['raw_readings_count'] = self.validation_stats.get('raw_readings_seen', 0)
        
        # Add all readings (including rejected) for visualization
        if self.all_readings:
            stats['all_readings_with_validation'] = self.all_readings
        
        # Add validation statistics
        if self.validation_gate:
            stats['data_validation'] = {
                'enabled': True,
                'raw_readings_seen': self.validation_stats['raw_readings_seen'],
                'readings_validated': self.validation_stats['readings_validated'],
                'readings_rejected': self.validation_stats['readings_rejected'],
                'readings_deduplicated': self.validation_stats['readings_deduplicated'],
                'rejection_rate': (self.validation_stats['readings_rejected'] / 
                                 (self.validation_stats['readings_validated'] + 
                                  self.validation_stats['readings_rejected'] + 
                                  self.validation_stats['readings_deduplicated'])) * 100
                                 if (self.validation_stats['readings_validated'] + 
                                     self.validation_stats['readings_rejected'] +
                                     self.validation_stats['readings_deduplicated']) > 0 else 0
            }
            
            # Get detailed validation stats
            validation_details = self.validation_gate.get_stats(self.current_user_id)
            if validation_details:
                stats['data_validation']['details'] = validation_details
            logger.debug(f"Validation details for {self.current_user_id}: {validation_details}")
        
        # Apply multimodal detection if enabled and enough data
        if self.multimodal_detector and len(self.user_processor.readings) >= 30:
            weights = [r['weight'] for r in self.user_processor.readings]
            multimodal_result = self.multimodal_detector.detect_multimodal(self.current_user_id, weights)
            
            if multimodal_result.get('is_multimodal'):
                stats['multimodal_detection'] = {
                    'detected': True,
                    'num_clusters': multimodal_result['num_clusters'],
                    'cluster_stats': multimodal_result.get('cluster_stats', []),
                    'detection_method': multimodal_result.get('detection_method', 'unknown'),
                    'virtual_users': []
                }
                
                # Create virtual user summaries
                for cluster_id, cluster_stat in enumerate(multimodal_result.get('cluster_stats', [])):
                    virtual_id = f"{self.current_user_id}_cluster_{cluster_id}"
                    stats['multimodal_detection']['virtual_users'].append({
                        'virtual_id': virtual_id,
                        'readings_count': cluster_stat['count'],
                        'mean_weight': cluster_stat['mean'],
                        'std_deviation': cluster_stat['std'],
                        'weight_range': (cluster_stat['min'], cluster_stat['max'])
                    })
                
                logger.info(f"Multimodal detection: User {self.current_user_id} has {multimodal_result['num_clusters']} distinct weight clusters")
            else:
                stats['multimodal_detection'] = {
                    'detected': False,
                    'reason': multimodal_result.get('reason', 'Single distribution')
                }
        
        # If baseline was established at the end, update Kalman's state
        # Note: Kalman has been running all along, but we can improve it with baseline
        if self.kalman_processor.enable_kalman and stats.get("baseline_established"):
            baseline_params = {
                'success': True,
                'baseline_weight': stats['baseline_weight'],
                'measurement_variance': stats['baseline_variance']
            }
            # Update the final Kalman state with better baseline info
            # This improves the final statistics but doesn't require reprocessing
            stats['baseline_used_for_kalman'] = True
            stats['kalman_baseline_weight'] = stats['baseline_weight']
            stats['kalman_baseline_variance'] = stats['baseline_variance']

        stats["kalman_filter_initialized"] = self.kalman_processor.enable_kalman
        stats["config_used"] = {
            "enable_kalman": self.kalman_processor.enable_kalman,
            "enable_visualization": self.config.get("enable_visualization", True),
            "enable_enhanced_validation": self.enable_enhanced_validation,
            "enable_multimodal_detection": self.enable_multimodal_detection,
            "confidence_thresholds": self.user_processor.confidence_thresholds,
            "weight_validation": self.user_processor.weight_validation,
            "source_type_trust_scores": self.user_processor.source_type_trust_scores
        }

        for i, ts_point in enumerate(stats.get("time_series", [])):
            if i < len(self.user_processor.readings) and self.user_processor.readings[i].get("kalman_filtered"):
                reading = self.user_processor.readings[i]
                ts_point["kalman_filtered"] = round(reading["kalman_filtered"], 2)
                ts_point["kalman_residual"] = round(reading["weight"] - reading["kalman_filtered"], 3)
                if reading.get("kalman_uncertainty") is not None:
                    ts_point["kalman_uncertainty"] = round(reading["kalman_uncertainty"], 3)
                if reading.get("kalman_trend") is not None:
                    ts_point["kalman_trend_kg_per_day"] = round(reading["kalman_trend"], 4)

        # Gap filling removed - no predictions added

        if self.kalman_processor.enable_kalman and self.kalman_processor.kalman_results:
            stats["kalman_summary"] = self.kalman_processor.create_summary()
            stats["kalman_time_series"] = self.kalman_processor.kalman_results
            stats.update(self.kalman_processor.get_filter_debug_state())

        debug_manager = DebugOutputManager(self.config)
        stats["data_quality"] = debug_manager.calculate_data_quality(
            stats, self.user_processor.readings
        )

        return stats


def process_csv_unified(config: Dict[str, Any]):
    """Main processing function with unified pipeline."""

    input_file = config["source_file"]

    if config.get("use_test_output_names", False):
        output_file = "output/results_test.json"
    else:
        pattern = config.get("output_file_pattern", "results_{timestamp}.json")
        output_file = f"output/{pattern.format(timestamp=timestamp)}"

    output_dir = Path(config.get("output_folder", "output"))
    output_dir.mkdir(exist_ok=True)

    debug_manager = DebugOutputManager(config)
    viz_manager = VisualizationManager(config, output_dir)
    progress_reporter = ProgressReporter()

    processor = UnifiedStreamProcessor(config)

    # Pass max_date to CSV reader to skip future-dated rows entirely
    max_date = config.get("max_date", None)
    csv_reader = StreamingCSVReader(input_file, max_date=max_date)

    specific_user_ids = config.get("specific_user_ids", config.get("user_id_list", []))
    specific_user_set = set(specific_user_ids) if specific_user_ids else None

    user_filter = UserFilteredReader(
        csv_reader,
        specific_user_set,
        config.get("skip_first_users", 0),
        config.get("process_max_users", 0)
    )

    min_readings = config.get("min_readings_per_user", 10)

    progress_reporter.print_startup_info(
        config, input_file, output_dir, timestamp,
        viz_manager.enable_visualization
    )

    logger.info(f"Starting unified processing of {input_file}")
    logger.info(
        f"Features: Kalman={config.get('enable_kalman', True)}, "
        f"Visualization={config.get('enable_visualization', True)}"
    )

    results = {}
    current_user_id = None
    users_processed = 0
    users_skipped = 0
    users_skipped_not_in_list = 0
    users_with_kalman = 0
    total_rows = 0
    users_seen = set()

    start_time = time.time()

    for row in csv_reader.read_rows():
        total_rows += 1
        user_id = row["user_id"]

        if user_id != current_user_id:
            if current_user_id is not None and processor.current_user_id is not None:
                summary = processor.finish_user()

                should_process = True

                if specific_user_set and current_user_id not in specific_user_set:
                    should_process = False
                    users_skipped_not_in_list += 1
                    logger.debug(f"Skipping user {current_user_id} (not in specific_user_ids)")
                elif not specific_user_set and len(users_seen) <= user_filter.skip_first_users:
                    users_skipped += 1
                    should_process = False
                    logger.info(
                        f"Skipping user {current_user_id} ({users_skipped}/{user_filter.skip_first_users})"
                    )
                elif summary.get("raw_readings_count", summary["total_readings"]) < min_readings:
                    should_process = False
                    logger.info(f"Skipping user {current_user_id} (only {summary.get('raw_readings_count', summary['total_readings'])} raw readings < {min_readings})")

                if should_process:
                    results[current_user_id] = summary
                    users_processed += 1

                    debug_manager.save_individual_user(current_user_id, summary)

                    if summary.get("kalman_filter_initialized"):
                        users_with_kalman += 1

                    viz_manager.create_visualization(current_user_id, summary)

                    progress_reporter.update_progress(
                        users_processed, users_skipped, total_rows,
                        users_with_kalman
                    )

                    if user_filter.max_users and users_processed >= user_filter.max_users:
                        break

            users_seen.add(user_id)

            if specific_user_set and user_id not in specific_user_set:
                current_user_id = user_id
                processor.reset()
            elif not specific_user_set and len(users_seen) <= user_filter.skip_first_users:
                current_user_id = user_id
                processor.reset()
            else:
                current_user_id = user_id
                processor.start_user(user_id)

        if processor.current_user_id is None:
            continue

        if specific_user_set and current_user_id not in specific_user_set:
            continue

        result = processor.process_reading(row)
        if not result:
            continue

        if total_rows % 1000 == 0:
            progress_reporter.update_progress(
                users_processed, users_skipped, total_rows,
                users_with_kalman, current_user=user_id
            )

    if current_user_id and processor.current_user_id and (not user_filter.max_users or users_processed < user_filter.max_users):
        if not specific_user_set or current_user_id in specific_user_set:
            summary = processor.finish_user()

            should_process = True

            if not specific_user_set and len(users_seen) <= user_filter.skip_first_users:
                users_skipped += 1
                should_process = False
                logger.info(f"Skipping user {current_user_id} ({users_skipped}/{user_filter.skip_first_users})")
            elif summary.get("raw_readings_count", summary["total_readings"]) < min_readings:
                should_process = False
                logger.info(f"Skipping user {current_user_id} (only {summary.get('raw_readings_count', summary['total_readings'])} raw readings < {min_readings})")

            if should_process:
                results[current_user_id] = summary
                users_processed += 1

                debug_manager.save_individual_user(current_user_id, summary)
                viz_manager.create_visualization(current_user_id, summary)

    elapsed = time.time() - start_time

    # Calculate aggregate validation statistics
    total_validated = 0
    total_rejected = 0
    total_deduplicated = 0
    users_with_multimodal = 0
    
    for user_id, user_stats in results.items():
        if 'data_validation' in user_stats:
            total_validated += user_stats['data_validation'].get('readings_validated', 0)
            total_rejected += user_stats['data_validation'].get('readings_rejected', 0)
            total_deduplicated += user_stats['data_validation'].get('readings_deduplicated', 0)
        if 'multimodal_detection' in user_stats and user_stats['multimodal_detection'].get('detected'):
            users_with_multimodal += 1
    
    metadata = {
        "processing_timestamp": datetime.now().isoformat(),
        "input_file": input_file,
        "output_file": output_file,
        "processor_version": "2.1-enhanced",
        "total_rows_processed": total_rows,
        "total_users_processed": users_processed,
        "total_users_skipped": users_skipped,
        "processing_time_seconds": round(elapsed, 2),
        "data_quality_summary": {
            "total_readings_validated": total_validated,
            "total_readings_rejected": total_rejected,
            "total_readings_deduplicated": total_deduplicated,
            "overall_rejection_rate": round((total_rejected + total_deduplicated) / 
                                           (total_validated + total_rejected + total_deduplicated) * 100, 2)
                                           if (total_validated + total_rejected + total_deduplicated) > 0 else 0,
            "users_with_multimodal": users_with_multimodal,
            "multimodal_detection_rate": round(users_with_multimodal / users_processed * 100, 2) if users_processed > 0 else 0
        },
        "config": {
            "enable_kalman": config.get("enable_kalman", True),
            "enable_visualization": config.get("enable_visualization", True),
            "enable_enhanced_validation": config.get("enable_enhanced_validation", True),
            "enable_multimodal_detection": config.get("enable_multimodal_detection", True),
            "min_readings_per_user": min_readings,
            "max_users": user_filter.max_users if user_filter.max_users else "unlimited",
            "skip_first_users": user_filter.skip_first_users,
            "specific_user_ids": len(specific_user_ids) if specific_user_ids else None
        }
    }

    processing_stats = {
        "users_with_kalman": users_with_kalman,
        "kalman_percentage": round(users_with_kalman / users_processed * 100, 2) if users_processed > 0 else 0,
        "users_with_multimodal": users_with_multimodal,
        "multimodal_percentage": round(users_with_multimodal / users_processed * 100, 2) if users_processed > 0 else 0,
        "avg_readings_per_user": round(total_rows / users_processed, 2) if users_processed > 0 else 0,
        "avg_validated_per_user": round(total_validated / users_processed, 2) if users_processed > 0 else 0,
        "processing_rate_users_per_sec": round(users_processed / elapsed, 2),
        "processing_rate_rows_per_sec": round(total_rows / elapsed, 0)
    }

    debug_output = debug_manager.create_comprehensive_output(results, metadata, processing_stats)
    debug_manager.save_results(output_file, debug_output)

    viz_count = viz_manager.visualizations_created
    progress_reporter.print_summary(
        total_rows, users_processed, users_with_kalman, elapsed,
        output_file, viz_manager.get_viz_directory(), viz_count,
        debug_manager.individual_output_dir, specific_user_set,
        users_skipped_not_in_list
    )

    logger.info(f"\n{'='*60}")
    logger.info("Processing Complete")
    logger.info(f"{'='*60}")
    logger.info(f"Total rows processed: {total_rows:,}")
    if csv_reader.skipped_future_dates > 0:
        logger.info(f"Rows skipped (future dates): {csv_reader.skipped_future_dates:,}")
    logger.info(f"Users processed: {users_processed:,}")
    logger.info(
        f"Users with Kalman: {users_with_kalman:,} "
        f"({users_with_kalman/users_processed*100:.1f}%)" if users_processed > 0 else ""
    )
    if config.get('enable_enhanced_validation', True):
        logger.info(
            f"Data validation: {total_rejected + total_deduplicated:,} readings removed "
            f"({(total_rejected + total_deduplicated)/(total_validated + total_rejected + total_deduplicated)*100:.1f}%)" 
            if (total_validated + total_rejected + total_deduplicated) > 0 else ""
        )
    if config.get('enable_multimodal_detection', True) and users_with_multimodal > 0:
        logger.info(
            f"Multimodal users: {users_with_multimodal:,} "
            f"({users_with_multimodal/users_processed*100:.1f}%)" if users_processed > 0 else ""
        )
    logger.info(f"Processing time: {elapsed:.2f} seconds")
    logger.info(f"Processing rate: {users_processed/elapsed:.1f} users/sec" if elapsed > 0 else "")
    logger.info(f"\nOutput file: {output_file}")
    if debug_manager.output_individual_files:
        logger.info(f"Individual user files: {debug_manager.individual_output_dir}/ ({users_processed} files)")

    if viz_manager.viz_dir:
        logger.info(f"Visualizations created: {viz_count} dashboards in {viz_manager.viz_dir}")

    logger.info(f"{'='*60}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Stream processor with configurable features")
    parser.add_argument(
        "--enable-viz", action="store_true", help="Enable visualizations even if disabled in config"
    )
    args = parser.parse_args()

    config = load_config()

    if args.enable_viz:
        config["enable_visualization"] = True
        print("CLI override: Visualizations enabled via --enable-viz flag")

    process_csv_unified(config)


if __name__ == "__main__":
    main()
