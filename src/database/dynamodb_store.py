"""DynamoDB implementation of StateStore."""

import json
import logging
import os
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Any, Optional, List

import numpy as np

from .base import StateStore

logger = logging.getLogger(__name__)

# Import boto3 only when actually needed
try:
    import boto3
    from botocore.exceptions import ClientError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    logger.warning("boto3 not available - DynamoDB store will not work")


class DynamoDBStateStore(StateStore):
    """DynamoDB-based state storage for AWS deployment."""

    def __init__(self, table_name: str = None, region: str = None):
        """
        Initialize DynamoDB state store.

        Args:
            table_name: DynamoDB table name
            region: AWS region
        """
        if not BOTO3_AVAILABLE:
            raise ImportError("boto3 is required for DynamoDB store. Install with: pip install boto3")

        self.table_name = table_name or os.getenv('DYNAMODB_TABLE_NAME', 'weight-processor-state')
        self.region = region or os.getenv('AWS_REGION', 'us-east-1')

        # Initialize DynamoDB client
        self.dynamodb = boto3.resource('dynamodb', region_name=self.region)
        self.table = self.dynamodb.Table(self.table_name)

    def get_state(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve state for a user from DynamoDB."""
        try:
            response = self.table.get_item(
                Key={
                    'userId': user_id,
                    'stateType': 'current'
                }
            )

            if 'Item' not in response:
                return None

            # Deserialize the state
            state = self._deserialize_state(response['Item'])
            return state

        except ClientError as e:
            logger.error(f"Error getting state from DynamoDB: {e}")
            return None

    def save_state(self, user_id: str, state: Dict[str, Any]) -> bool:
        """Save state to DynamoDB."""
        try:
            # Serialize the state
            item = self._serialize_state(state)
            item.update({
                'userId': user_id,
                'stateType': 'current',
                'updatedAt': datetime.utcnow().isoformat(),
                'version': state.get('version', 0) + 1
            })

            # Save to DynamoDB
            self.table.put_item(Item=item)
            return True

        except ClientError as e:
            logger.error(f"Error saving state to DynamoDB: {e}")
            return False

    def delete_state(self, user_id: str) -> bool:
        """Delete state from DynamoDB."""
        try:
            # Delete current state
            self.table.delete_item(
                Key={
                    'userId': user_id,
                    'stateType': 'current'
                }
            )

            # Delete all snapshots
            self._delete_user_snapshots(user_id)
            return True

        except ClientError as e:
            logger.error(f"Error deleting state from DynamoDB: {e}")
            return False

    def create_initial_state(self) -> Dict[str, Any]:
        """Create an empty initial state."""
        return {
            'kalman_params': None,
            'last_state': None,
            'last_covariance': None,
            'last_timestamp': None,
            'last_accepted_timestamp': None,
            'last_source': None,
            'last_raw_weight': None,
            'measurement_history': [],
            'reset_events': [],
            'measurements_since_reset': 0,
            'adaptation_state': {},
            'version': 0
        }

    def save_state_snapshot(self, user_id: str, timestamp: datetime) -> bool:
        """Save a snapshot to DynamoDB."""
        try:
            # Get current state
            current_state = self.get_state(user_id)
            if not current_state:
                return False

            # Create snapshot item
            snapshot = self._serialize_state(current_state)
            snapshot.update({
                'userId': user_id,
                'stateType': f'snapshot_{timestamp.isoformat()}',
                'snapshotTime': timestamp.isoformat(),
                'ttl': int((timestamp + timedelta(days=7)).timestamp())  # 7-day retention
            })

            # Save to DynamoDB
            self.table.put_item(Item=snapshot)
            return True

        except ClientError as e:
            logger.error(f"Error saving snapshot to DynamoDB: {e}")
            return False

    def restore_state_snapshot(self, user_id: str) -> bool:
        """Restore state from the latest snapshot."""
        try:
            # Query for the latest snapshot
            response = self.table.query(
                KeyConditionExpression='userId = :uid AND begins_with(stateType, :st)',
                ExpressionAttributeValues={
                    ':uid': user_id,
                    ':st': 'snapshot_'
                },
                ScanIndexForward=False,  # Descending order
                Limit=1
            )

            if not response.get('Items'):
                return False

            snapshot_item = response['Items'][0]
            # Restore the snapshot as current state
            state = self._deserialize_state(snapshot_item)
            return self.save_state(user_id, state)

        except ClientError as e:
            logger.error(f"Error restoring snapshot from DynamoDB: {e}")
            return False

    def get_snapshot(self, user_id: str, timestamp: datetime) -> Optional[Dict[str, Any]]:
        """Get the nearest snapshot before the given timestamp."""
        try:
            # Query snapshots
            response = self.table.query(
                KeyConditionExpression='userId = :uid AND stateType < :st',
                ExpressionAttributeValues={
                    ':uid': user_id,
                    ':st': f'snapshot_{timestamp.isoformat()}'
                },
                ScanIndexForward=False,  # Descending order
                Limit=1
            )

            if not response.get('Items'):
                return None

            return self._deserialize_state(response['Items'][0])

        except ClientError as e:
            logger.error(f"Error getting snapshot from DynamoDB: {e}")
            return None

    def check_and_restore_snapshot(self, user_id: str, buffer_start_time: datetime) -> dict:
        """Check if a snapshot exists and restore it atomically."""
        snapshot = self.get_snapshot(user_id, buffer_start_time)
        if snapshot:
            # Restore the snapshot as current state
            if self.save_state(user_id, snapshot):
                return {
                    'success': True,
                    'snapshot': snapshot,
                    'snapshot_timestamp': snapshot.get('last_timestamp', buffer_start_time),
                    'user_id': user_id
                }

        return {
            'success': False,
            'error': f'No snapshot found for user {user_id}',
            'user_id': user_id
        }

    def export_to_csv(self, filepath: str) -> int:
        """Export all states to CSV (not implemented for DynamoDB)."""
        logger.warning("CSV export not implemented for DynamoDB store")
        return 0

    def _delete_user_snapshots(self, user_id: str):
        """Delete all snapshots for a user."""
        try:
            # Query all snapshots
            response = self.table.query(
                KeyConditionExpression='userId = :uid AND begins_with(stateType, :st)',
                ExpressionAttributeValues={
                    ':uid': user_id,
                    ':st': 'snapshot_'
                }
            )

            # Delete each snapshot
            for item in response.get('Items', []):
                self.table.delete_item(
                    Key={
                        'userId': user_id,
                        'stateType': item['stateType']
                    }
                )

        except ClientError as e:
            logger.error(f"Error deleting snapshots: {e}")

    def _serialize_state(self, state: Dict[str, Any], depth: int = 0) -> Dict[str, Any]:
        """Serialize state for DynamoDB storage with depth protection."""
        MAX_DEPTH = 10  # Prevent infinite recursion

        if depth > MAX_DEPTH:
            logger.warning(f"Max serialization depth {MAX_DEPTH} reached")
            return {"__truncated__": "Max depth exceeded"}

        serialized = {}

        for key, value in state.items():
            if value is None:
                continue  # Skip None values

            if isinstance(value, np.ndarray):
                # Convert numpy arrays to lists
                serialized[key] = value.tolist()
            elif isinstance(value, datetime):
                # Convert datetime to ISO string
                serialized[key] = value.isoformat()
            elif isinstance(value, float):
                # Convert float to Decimal for DynamoDB
                serialized[key] = Decimal(str(value))
            elif isinstance(value, dict):
                # Recursively serialize nested dicts with depth tracking
                serialized[key] = self._serialize_state(value, depth + 1)
            elif isinstance(value, list):
                # Handle lists with depth tracking
                serialized[key] = [
                    self._serialize_value(item, depth + 1) for item in value
                ]
            else:
                serialized[key] = value

        return serialized

    def _deserialize_state(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Deserialize state from DynamoDB."""
        state = {}

        for key, value in item.items():
            if key in ['userId', 'stateType', 'ttl', 'updatedAt', 'version']:
                continue  # Skip DynamoDB metadata

            if key in ['last_timestamp', 'last_accepted_timestamp']:
                # Convert ISO strings back to datetime
                if isinstance(value, str):
                    state[key] = datetime.fromisoformat(value)
                else:
                    state[key] = value
            elif key == 'kalman_params' and value:
                # Handle Kalman parameters
                state[key] = self._deserialize_kalman_params(value)
            elif isinstance(value, Decimal):
                # Convert Decimal back to float
                state[key] = float(value)
            elif isinstance(value, dict):
                # Recursively deserialize nested dicts
                state[key] = self._deserialize_state(value)
            elif isinstance(value, list):
                # Handle lists
                state[key] = [self._deserialize_value(item) for item in value]
            else:
                state[key] = value

        return state

    def _deserialize_kalman_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Deserialize Kalman filter parameters."""
        deserialized = {}

        for key, value in params.items():
            if key in ['x', 'P'] and isinstance(value, list):
                # Convert lists back to numpy arrays
                deserialized[key] = np.array(value)
            elif isinstance(value, Decimal):
                deserialized[key] = float(value)
            else:
                deserialized[key] = value

        return deserialized

    def _serialize_value(self, value: Any, depth: int = 0) -> Any:
        """Serialize a single value with depth protection."""
        MAX_DEPTH = 10

        if depth > MAX_DEPTH:
            logger.warning(f"Max serialization depth {MAX_DEPTH} reached in value")
            return "__truncated__"

        if isinstance(value, float):
            return Decimal(str(value))
        elif isinstance(value, datetime):
            return value.isoformat()
        elif isinstance(value, np.ndarray):
            return value.tolist()
        elif isinstance(value, dict):
            return self._serialize_state(value, depth + 1)
        return value

    def _deserialize_value(self, value: Any) -> Any:
        """Deserialize a single value."""
        if isinstance(value, Decimal):
            return float(value)
        elif isinstance(value, str):
            # Try to parse as datetime
            try:
                return datetime.fromisoformat(value)
            except (ValueError, AttributeError):
                return value
        elif isinstance(value, dict):
            return self._deserialize_state(value)
        return value