"""
Stateless Weight Processor V2 - Cleaner Interface
Processor handles database interaction internally with user_id
"""

from datetime import datetime
from typing import Any, Dict, Optional, Tuple
import numpy as np
from pykalman import KalmanFilter
from processor_database import ProcessorStateDB, get_state_db


class WeightProcessor:
    """
    Stateless weight processor with clean interface.
    All computation is pure functional, state is managed via database.
    """

    @staticmethod
    def process_weight(
        user_id: str,
        weight: float,
        timestamp: datetime,
        source: str,
        processing_config: Dict,
        kalman_config: Dict,
        db=None,
    ) -> Optional[Dict]:
        """
        Process a single weight measurement for a user.

        This is the main entry point for the stateless processor.
        State is loaded from database, updated, and saved back.

        Args:
            user_id: User identifier
            weight: Weight measurement in kg
            timestamp: Measurement timestamp
            source: Data source identifier
            processing_config: Processing configuration dict
            kalman_config: Kalman filter configuration dict
            db: Optional database instance (creates new if None)

        Returns:
            Result dictionary or None if buffering for initialization
        """
        # Get database instance
        if db is None:
            db = get_state_db()

        # Load or create state
        state = db.get_state(user_id)
        if state is None:
            state = db.create_initial_state()

        # Process the weight
        result, updated_state = WeightProcessor._process_weight_internal(
            weight, timestamp, source, state, processing_config, kalman_config
        )

        # Save updated state if changed
        if updated_state:
            db.save_state(user_id, updated_state)

        return result

    @staticmethod
    def _process_weight_internal(
        weight: float,
        timestamp: datetime,
        source: str,
        state: Dict[str, Any],
        processing_config: dict,
        kalman_config: dict
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Internal processing logic - pure functional.

        Returns:
            Tuple of (result, updated_state)
            updated_state is None if state wasn't modified
        """
        # Basic validation
        if not WeightProcessor._validate_weight(
            weight,
            processing_config,
            state.get('last_state')
        ):
            return {
                "timestamp": timestamp,
                "raw_weight": weight,
                "accepted": False,
                "reason": "Basic validation failed",
                "source": source,
            }, None

        # Check if initialized
        if not state.get('initialized', False):
            # Buffer for initialization
            new_state = state.copy()
            init_buffer = new_state.get('init_buffer', [])
            init_buffer.append((weight, timestamp))
            new_state['init_buffer'] = init_buffer

            if len(init_buffer) >= processing_config["min_init_readings"]:
                # Initialize Kalman filter
                kalman_state = WeightProcessor._initialize_kalman(
                    init_buffer,
                    processing_config,
                    kalman_config
                )
                new_state.update(kalman_state)
                new_state['initialized'] = True
                new_state['init_buffer'] = []  # Clear buffer

                # Process all buffered measurements
                for w, t in init_buffer:
                    new_state = WeightProcessor._update_kalman_state(
                        new_state, w, t, source, processing_config
                    )

                # Return result for last measurement
                result = WeightProcessor._create_result(
                    new_state, weight, timestamp, source, True
                )
                return result, new_state
            else:
                # Still buffering
                return None, new_state

        # Process with existing Kalman filter
        new_state = state.copy()

        # Calculate time delta
        time_delta_days = 1.0
        if new_state.get('last_timestamp'):
            last_timestamp = new_state['last_timestamp']
            if isinstance(last_timestamp, str):
                last_timestamp = datetime.fromisoformat(last_timestamp)
            delta = (timestamp - last_timestamp).total_seconds() / 86400.0
            time_delta_days = max(0.1, min(30.0, delta))

            # Check for reset condition
            reset_gap_days = kalman_config.get("reset_gap_days", 30)
            if new_state.get('adapted_params'):
                reset_gap_days = new_state['adapted_params'].get("reset_gap_days", reset_gap_days)

            if delta > reset_gap_days:
                new_state = WeightProcessor._reset_kalman_state(new_state, weight)

        # Check for extreme deviation
        if new_state.get('last_state') is not None:
            last_state = new_state['last_state']
            if len(last_state.shape) > 1:
                current_state = last_state[-1]
            else:
                current_state = last_state

            predicted_weight = current_state[0] + current_state[1] * time_delta_days
            deviation = abs(weight - predicted_weight) / predicted_weight

            extreme_threshold = processing_config["extreme_threshold"]
            if new_state.get('adapted_params'):
                extreme_threshold = new_state['adapted_params'].get(
                    'extreme_threshold', extreme_threshold
                )

            if deviation > extreme_threshold:
                # Reject measurement
                pseudo_normalized_innovation = (deviation / extreme_threshold) * 3.0
                confidence = WeightProcessor._calculate_confidence(
                    pseudo_normalized_innovation
                )

                return {
                    "timestamp": timestamp,
                    "raw_weight": weight,
                    "filtered_weight": float(predicted_weight),
                    "trend": float(current_state[1]),
                    "accepted": False,
                    "reason": f"Extreme deviation: {deviation:.1%}",
                    "confidence": confidence,
                    "source": source,
                }, None  # State not updated for rejected measurements

        # Update Kalman state
        new_state = WeightProcessor._update_kalman_state(
            new_state, weight, timestamp, source, processing_config
        )

        # Create result
        result = WeightProcessor._create_result(
            new_state, weight, timestamp, source, True
        )

        return result, new_state

    @staticmethod
    def _validate_weight(
        weight: float,
        config: dict,
        last_state: Optional[np.ndarray]
    ) -> bool:
        """Validate weight measurement."""
        if weight < config["min_weight"] or weight > config["max_weight"]:
            return False

        if last_state is not None:
            if isinstance(last_state, np.ndarray):
                last_weight = last_state[-1][0] if len(last_state.shape) > 1 else last_state[0]
            else:
                last_weight = last_state[0] if isinstance(last_state, (list, tuple)) else last_state

            change = abs(weight - last_weight) / last_weight
            if change > 0.5:  # 50% change is definitely an error
                return False

        return True

    @staticmethod
    def _initialize_kalman(
        init_buffer: list,
        processing_config: dict,
        kalman_config: dict
    ) -> Dict[str, Any]:
        """Initialize Kalman filter and return initial state."""
        weights = [w for w, _ in init_buffer]

        # Compute adapted parameters
        adapted_params = WeightProcessor._adapt_parameters(
            init_buffer, processing_config, kalman_config
        )

        # Clean weights for baseline
        weights_clean = []
        for w in weights:
            if not weights_clean:
                weights_clean.append(w)
            else:
                median = np.median(weights_clean)
                if abs(w - median) / median < 0.1:
                    weights_clean.append(w)

        if len(weights_clean) < 3:
            weights_clean = weights

        baseline = np.median(weights_clean)

        # Calculate initial variance
        mad = np.median([abs(w - baseline) for w in weights_clean])
        estimated_std = mad * 1.4826
        variance = estimated_std ** 2 if estimated_std > 0 else kalman_config["initial_variance"]

        # Store Kalman parameters (not the filter object)
        kalman_params = {
            'initial_state_mean': [baseline, 0],
            'initial_state_covariance': [[variance, 0], [0, 0.001]],
            'transition_covariance': [
                [adapted_params.get("transition_covariance_weight",
                                   kalman_config["transition_covariance_weight"]), 0],
                [0, adapted_params.get("transition_covariance_trend",
                                      kalman_config["transition_covariance_trend"])]
            ],
            'observation_covariance': [[adapted_params["observation_covariance"]]],
        }

        return {
            'kalman_params': kalman_params,
            'adapted_params': adapted_params,
            'last_state': np.array([[baseline, 0]]),
            'last_covariance': np.array([[[variance, 0], [0, 0.001]]]),
            'last_timestamp': init_buffer[-1][1] if init_buffer else None,
        }

    @staticmethod
    def _update_kalman_state(
        state: Dict[str, Any],
        weight: float,
        timestamp: datetime,
        source: str,
        processing_config: dict
    ) -> Dict[str, Any]:
        """Update Kalman filter state with new measurement."""
        # Calculate time delta
        time_delta_days = 1.0
        if state.get('last_timestamp'):
            last_timestamp = state['last_timestamp']
            if isinstance(last_timestamp, str):
                last_timestamp = datetime.fromisoformat(last_timestamp)
            delta = (timestamp - last_timestamp).total_seconds() / 86400.0
            time_delta_days = max(0.1, min(30.0, delta))

        # Create Kalman filter from stored parameters
        kalman_params = state['kalman_params']
        kalman = KalmanFilter(
            transition_matrices=np.array([[1, time_delta_days], [0, 1]]),
            observation_matrices=np.array([[1, 0]]),
            initial_state_mean=np.array(kalman_params['initial_state_mean']),
            initial_state_covariance=np.array(kalman_params['initial_state_covariance']),
            transition_covariance=np.array(kalman_params['transition_covariance']),
            observation_covariance=np.array(kalman_params['observation_covariance']),
        )

        # Update state
        observation = np.array([[weight]])

        if state.get('last_state') is None:
            filtered_state_means, filtered_state_covariances = kalman.filter(observation)
            new_last_state = filtered_state_means
            new_last_covariance = filtered_state_covariances
        else:
            last_state = state['last_state']
            last_covariance = state['last_covariance']

            # Get the most recent state
            if len(last_state.shape) > 1:
                current_state = last_state[-1]
                current_covariance = last_covariance[-1]
            else:
                current_state = last_state
                current_covariance = last_covariance[0] if len(last_covariance.shape) > 2 else last_covariance

            filtered_state_mean, filtered_state_covariance = kalman.filter_update(
                current_state,
                current_covariance,
                observation=observation[0],
            )

            # Keep only last 2 states for efficiency
            if len(last_state.shape) == 1:
                new_last_state = np.array([last_state, filtered_state_mean])
                new_last_covariance = np.array([last_covariance, filtered_state_covariance])
            else:
                new_last_state = np.array([last_state[-1], filtered_state_mean])
                new_last_covariance = np.array([last_covariance[-1], filtered_state_covariance])

        # Update and return state
        state['last_state'] = new_last_state
        state['last_covariance'] = new_last_covariance
        state['last_timestamp'] = timestamp

        return state

    @staticmethod
    def _reset_kalman_state(state: Dict[str, Any], new_weight: float) -> Dict[str, Any]:
        """Reset Kalman state after long gap."""
        state['last_state'] = np.array([[new_weight, 0]])
        state['last_covariance'] = np.array([[[1.0, 0], [0, 0.001]]])

        # Update initial state mean in kalman params
        if state.get('kalman_params'):
            state['kalman_params']['initial_state_mean'] = [new_weight, 0]

        return state

    @staticmethod
    def _adapt_parameters(
        user_data: list,
        processing_config: dict,
        kalman_config: dict
    ) -> Dict[str, Any]:
        """Compute adapted parameters based on user data."""
        adapted = kalman_config.copy()
        adapted['extreme_threshold'] = processing_config["extreme_threshold"]

        if len(user_data) < 3:
            return adapted

        weights = [w for w, _ in user_data] if isinstance(user_data[0], tuple) else user_data

        # Calculate variance using MAD
        median_weight = np.median(weights)
        mad = np.median([abs(w - median_weight) for w in weights])
        variance = mad * 1.4826

        # Adapt parameters based on variance
        if variance < 0.5:
            adapted['observation_covariance'] = 0.5
            adapted['extreme_threshold'] = 0.20
        elif variance < 1.5:
            adapted['observation_covariance'] = 1.5
            adapted['extreme_threshold'] = 0.25
        else:
            adapted['observation_covariance'] = 3.0
            adapted['extreme_threshold'] = 0.35

        # Check for trends
        if len(weights) > 7:
            first_half = np.mean(weights[:len(weights)//2])
            second_half = np.mean(weights[len(weights)//2:])
            trend_magnitude = abs(second_half - first_half) / len(weights)

            if trend_magnitude > 0.05:
                adapted['transition_covariance_trend'] = 0.001
            else:
                adapted['transition_covariance_trend'] = 0.0005

        return adapted

    @staticmethod
    def _calculate_confidence(normalized_innovation: float) -> float:
        """Calculate confidence using smooth exponential decay."""
        alpha = 0.5
        return np.exp(-alpha * normalized_innovation ** 2)

    @staticmethod
    def _create_result(
        state: Dict[str, Any],
        weight: float,
        timestamp: datetime,
        source: str,
        accepted: bool
    ) -> Dict[str, Any]:
        """Create result dictionary from current state."""
        if state.get('last_state') is None:
            return None

        # Get current state
        last_state = state['last_state']
        if len(last_state.shape) > 1:
            current_state = last_state[-1]
        else:
            current_state = last_state

        filtered_weight = current_state[0]
        trend = current_state[1]

        # Calculate innovation
        innovation = weight - filtered_weight

        # Get covariance for confidence calculation
        last_covariance = state.get('last_covariance')
        if last_covariance is not None:
            if len(last_covariance.shape) > 2:
                current_covariance = last_covariance[-1]
            else:
                current_covariance = last_covariance

            obs_covariance = state['kalman_params']['observation_covariance'][0][0]
            innovation_variance = current_covariance[0, 0] + obs_covariance
            normalized_innovation = (
                abs(innovation) / np.sqrt(innovation_variance)
                if innovation_variance > 0
                else 0
            )
        else:
            normalized_innovation = 0

        confidence = WeightProcessor._calculate_confidence(normalized_innovation)

        return {
            "timestamp": timestamp,
            "raw_weight": weight,
            "filtered_weight": float(filtered_weight),
            "trend": float(trend),
            "trend_weekly": float(trend * 7),
            "accepted": accepted,
            "confidence": float(confidence),
            "innovation": float(innovation),
            "normalized_innovation": float(normalized_innovation),
            "source": source,
        }

    @staticmethod
    def get_user_state(user_id: str) -> Optional[Dict[str, Any]]:
        """Get the current state for a user (for debugging/inspection)."""
        db = get_state_db()
        return db.get_state(user_id)

    @staticmethod
    def reset_user(user_id: str) -> bool:
        """Reset a user's state (delete from database)."""
        db = get_state_db()
        return db.delete_state(user_id)
