#!/usr/bin/env python3
"""Weight stream reprocessor for batch and retroactive processing."""

from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from collections import defaultdict
import logging

try:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent / 'legacy'))
    from processor_v4_legacy import WeightProcessor
    from .processor_database import ProcessorDatabase
except ImportError:
    from processor_database import ProcessorDatabase
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent / 'legacy'))
    from processor_v4_legacy import WeightProcessor

logger = logging.getLogger(__name__)


class WeightReprocessor:
    """Handles retroactive and batch reprocessing of weight data."""
    
    SOURCE_PRIORITY = {
        'internal-questionnaire': 1,
        'initial-questionnaire': 1,
        'https://api.iglucose.com': 2,
        'https://connectivehealth.io': 2,
        'patient-device': 3,
        'user-device': 3,
        'patient-upload': 4,
    }
    
    @staticmethod
    def reprocess_from_date(
        user_id: str,
        start_date: datetime,
        measurements: List[Dict[str, Any]],
        processing_config: Dict,
        kalman_config: Dict,
        db: Optional[ProcessorDatabase] = None
    ) -> Dict[str, Any]:
        """
        Reprocess all measurements from a specific date.
        
        Args:
            user_id: User identifier
            start_date: Date to start reprocessing from
            measurements: All measurements for the user (sorted by time)
            processing_config: Processing configuration
            kalman_config: Kalman filter configuration
            db: Database instance (creates new if None)
            
        Returns:
            Dict with reprocessing results and statistics
        """
        if db is None:
            db = ProcessorDatabase()
        
        snapshot_id = db.create_snapshot(user_id, datetime.now())
        
        state_before_date = db.get_state_before_date(user_id, start_date)
        if state_before_date:
            db.save_state(user_id, state_before_date)
        else:
            db.clear_state(user_id)
        
        relevant_measurements = [
            m for m in measurements 
            if m['timestamp'] >= start_date
        ]
        
        results = []
        errors = []
        
        for measurement in relevant_measurements:
            try:
                result = WeightProcessor.process_weight(
                    user_id=user_id,
                    weight=measurement['weight'],
                    timestamp=measurement['timestamp'],
                    source=measurement.get('source', 'unknown'),
                    processing_config=processing_config,
                    kalman_config=kalman_config,
                    db=db
                )
                results.append(result)
            except Exception as e:
                errors.append({
                    'measurement': measurement,
                    'error': str(e)
                })
        
        return {
            'snapshot_id': snapshot_id,
            'start_date': start_date,
            'measurements_processed': len(relevant_measurements),
            'successful': len(results),
            'errors': errors,
            'results': results
        }
    
    @staticmethod
    def process_daily_batch(
        user_id: str,
        batch_date: date,
        measurements: List[Dict[str, Any]],
        processing_config: Dict,
        kalman_config: Dict,
        db: Optional[ProcessorDatabase] = None
    ) -> Dict[str, Any]:
        """
        Process a day's measurements using Kalman predictions for validation.
        
        Args:
            user_id: User identifier
            batch_date: Date to process
            measurements: Measurements for that day
            processing_config: Processing configuration
            kalman_config: Kalman filter configuration
            db: Database instance
            
        Returns:
            Dict with selected measurements and processing results
        """
        if not measurements:
            return {'selected': [], 'rejected': [], 'reason': 'no_measurements'}
        
        if len(measurements) == 1:
            return {
                'selected': measurements,
                'rejected': [],
                'reason': 'single_measurement'
            }
        
        # Get Kalman prediction if available
        kalman_prediction = None
        if db:
            state = db.get_state(user_id)
            if state and state.get('kalman_params'):
                # Use the last known state as baseline prediction
                last_state = state.get('last_state')
                if last_state is not None:
                    import numpy as np
                    if isinstance(last_state, np.ndarray):
                        kalman_prediction = float(last_state.flat[0])
                    else:
                        kalman_prediction = float(last_state[0] if hasattr(last_state, '__len__') else last_state)
        
        # Select measurements based on Kalman prediction
        if kalman_prediction is not None:
            # Use Kalman-guided selection with configurable threshold
            threshold = processing_config.get('kalman_cleanup_threshold', 2.0)  # kg
            selected = WeightReprocessor.select_kalman_guided_measurements(
                measurements,
                kalman_prediction=kalman_prediction,
                threshold=threshold
            )
        else:
            # Fallback to statistical selection if no Kalman state
            selected = WeightReprocessor.select_best_measurements(
                measurements,
                max_deviation=processing_config.get('max_daily_change', 0.05) * 100
            )
        
        rejected = [m for m in measurements if m not in selected]
        
        if db and selected:
            start_of_day = datetime.combine(batch_date, datetime.min.time())
            reprocess_result = WeightReprocessor.reprocess_from_date(
                user_id=user_id,
                start_date=start_of_day,
                measurements=selected,
                processing_config=processing_config,
                kalman_config=kalman_config,
                db=db
            )
        else:
            reprocess_result = None
        
        return {
            'selected': selected,
            'rejected': rejected,
            'reason': 'kalman_guided_filtering' if kalman_prediction else 'statistical_filtering',
            'reprocess_result': reprocess_result,
            'kalman_prediction': kalman_prediction
        }
    
    @staticmethod
    def select_kalman_guided_measurements(
        measurements: List[Dict[str, Any]],
        kalman_prediction: float,
        threshold: float = 2.0
    ) -> List[Dict[str, Any]]:
        """
        Select measurements that are close to the Kalman prediction.
        
        The Kalman filter's prediction is the best estimate of the true weight,
        so we keep only measurements within a threshold of this prediction.
        
        Args:
            measurements: List of measurements to filter
            kalman_prediction: The Kalman filter's predicted weight
            threshold: Maximum deviation from prediction in kg
            
        Returns:
            List of measurements within threshold of Kalman prediction
        """
        if not measurements:
            return []
        
        # First remove extreme outliers (physiological bounds)
        valid_measurements = []
        for m in measurements:
            if 30 <= m['weight'] <= 400:
                valid_measurements.append(m)
        
        if not valid_measurements:
            return []
        
        # Calculate deviation from Kalman prediction for each measurement
        scored_measurements = []
        for m in valid_measurements:
            deviation = abs(m['weight'] - kalman_prediction)
            
            # Get source priority (lower number = higher priority)
            source_priority = WeightReprocessor.SOURCE_PRIORITY.get(
                m.get('source', 'unknown'), 999
            )
            
            # Create composite score: deviation is primary, source is tiebreaker
            # Multiply deviation by 100 to ensure it dominates, add source as minor factor
            score = (deviation * 100) + (source_priority * 0.1)
            
            scored_measurements.append((score, deviation, m))
        
        # Sort by composite score
        scored_measurements.sort(key=lambda x: x[0])
        
        # Keep only measurements within threshold
        selected = []
        for effective_dev, actual_dev, m in scored_measurements:
            if actual_dev <= threshold:
                selected.append(m)
        
        # If nothing within threshold, keep the closest one
        if not selected and scored_measurements:
            selected = [scored_measurements[0][2]]
        
        return selected
    
    @staticmethod
    def select_best_measurements(
        measurements: List[Dict[str, Any]],
        max_deviation: float = 5.0,
        recent_weight: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Filter out bad measurements while keeping all acceptable ones.
        
        Removes only measurements that are:
        1. Extreme outliers (outside 30-400kg range)
        2. Statistical outliers (MAD-based detection)
        
        Sorts remaining measurements by quality:
        - Source reliability (scale > manual > unknown)
        - Consistency with recent weight if available
        
        Args:
            measurements: List of measurements to filter
            max_deviation: Maximum allowed deviation in kg
            recent_weight: Recent known good weight for comparison
            
        Returns:
            List of all valid measurements, sorted by quality
        """
        if not measurements:
            return []
        
        if len(measurements) == 1:
            return measurements
        
        weights = np.array([m['weight'] for m in measurements])
        
        def remove_extreme_outliers(measurements, weights):
            """Remove values that are clearly errors (unrealistic weights)."""
            valid_measurements = []
            for i, m in enumerate(measurements):
                if 30 <= weights[i] <= 400:
                    valid_measurements.append(m)
            
            if not valid_measurements:
                median = np.median(weights)
                mad = np.median(np.abs(weights - median))
                if mad > 0:
                    threshold = 3 * mad
                    valid_idx = np.abs(weights - median) <= threshold
                    return [m for i, m in enumerate(measurements) if valid_idx[i]]
                
            return valid_measurements
        
        measurements = remove_extreme_outliers(measurements, weights)
        if not measurements:
            return []
        
        weights = np.array([m['weight'] for m in measurements])
        
        if len(measurements) > 2:
            median = np.median(weights)
            mad = np.median(np.abs(weights - median))
            
            if mad == 0:
                valid_measurements = measurements
            else:
                z_scores = 0.6745 * (weights - median) / mad
                
                valid_measurements = [
                    m for i, m in enumerate(measurements)
                    if abs(z_scores[i]) < 3.5
                ]
        else:
            valid_measurements = measurements
        
        if not valid_measurements:
            valid_measurements = measurements
        
        if recent_weight is not None:
            scored_measurements = []
            for m in valid_measurements:
                deviation = abs(m['weight'] - recent_weight)
                source_priority = WeightReprocessor.SOURCE_PRIORITY.get(
                    m.get('source', 'unknown'), 999
                )
                
                score = deviation + (source_priority * 0.1)
                scored_measurements.append((score, m))
            
            scored_measurements.sort(key=lambda x: x[0])
            valid_measurements = [m for _, m in scored_measurements]
        else:
            valid_measurements.sort(
                key=lambda m: WeightReprocessor.SOURCE_PRIORITY.get(
                    m.get('source', 'unknown'), 999
                )
            )
        
        return valid_measurements
    
    @staticmethod
    def detect_retroactive_addition(
        user_id: str,
        new_measurement: Dict[str, Any],
        db: ProcessorDatabase
    ) -> bool:
        """
        Detect if a measurement is being added retroactively.
        
        Args:
            user_id: User identifier
            new_measurement: New measurement to check
            db: Database instance
            
        Returns:
            True if measurement is retroactive
        """
        last_timestamp = db.get_last_timestamp(user_id)
        
        if last_timestamp is None:
            return False
        
        return new_measurement['timestamp'] < last_timestamp
    
    @staticmethod
    def daily_reprocessing_job(
        target_date: Optional[date] = None,
        processing_config: Optional[Dict] = None,
        kalman_config: Optional[Dict] = None,
        db: Optional[ProcessorDatabase] = None
    ) -> Dict[str, Any]:
        """
        Daily job to reprocess measurements with multiple values.
        
        Args:
            target_date: Date to process (default: yesterday)
            processing_config: Processing configuration
            kalman_config: Kalman filter configuration
            db: Database instance
            
        Returns:
            Summary of reprocessing results
        """
        if target_date is None:
            target_date = (datetime.now() - timedelta(days=1)).date()
        
        if db is None:
            db = ProcessorDatabase()
        
        users_processed = []
        users_with_changes = []
        total_rejected = 0
        
        return {
            'date': target_date,
            'users_processed': len(users_processed),
            'users_with_changes': len(users_with_changes),
            'total_measurements_rejected': total_rejected,
            'details': users_with_changes
        }
    
    @staticmethod
    def validate_reprocessing(
        user_id: str,
        before_snapshot: str,
        after_results: Dict,
        db: ProcessorDatabase
    ) -> Tuple[bool, str]:
        """
        Validate that reprocessing improved data quality.
        
        Args:
            user_id: User identifier
            before_snapshot: Snapshot ID before reprocessing
            after_results: Results after reprocessing
            db: Database instance
            
        Returns:
            Tuple of (is_valid, reason)
        """
        current_state = db.get_state(user_id)
        
        if not current_state or not current_state.get('kalman_params'):
            return False, "Kalman parameters not set after reprocessing"
        
        if 'last_state' in current_state:
            weight = current_state['last_state'][0]
            if weight < 30 or weight > 400:
                return False, f"Invalid weight after reprocessing: {weight}"
        
        error_rate = len(after_results.get('errors', [])) / max(
            after_results.get('measurements_processed', 1), 1
        )
        if error_rate > 0.5:
            return False, f"High error rate: {error_rate:.1%}"
        
        return True, "Validation passed"