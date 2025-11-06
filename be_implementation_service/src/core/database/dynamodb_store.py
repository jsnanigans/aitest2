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
    from botocore.config import Config

    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    logger.warning("boto3 not available - DynamoDB store will not work")


class DynamoDBStateStore(StateStore):
    """DynamoDB-based state storage for AWS deployment."""

    # Class-level session for connection reuse
    _session = None

    def __init__(self, table_name: str = None, region: str = None):
        """
        Initialize DynamoDB state store.

        Args:
            table_name: DynamoDB table name
            region: AWS region
        """
        if not BOTO3_AVAILABLE:
            raise ImportError(
                "boto3 is required for DynamoDB store. Install with: pip install boto3"
            )

        self.table_name = table_name or os.getenv(
            "DYNAMODB_TABLE_NAME", "weight-processor-state"
        )
        self.region = region or os.getenv("AWS_REGION", "us-east-1")

        # Check if we're running against DynamoDB Local
        endpoint_url = os.getenv("DYNAMODB_ENDPOINT")

        # Configure boto3 client with optimized settings
        boto_config = Config(
            region_name=self.region,
            retries={
                'max_attempts': 3,  # Reduced from default 9 for faster failure
                'mode': 'adaptive'  # Better backoff strategy
            },
            max_pool_connections=50,  # Increase connection pool size
            connect_timeout=5,  # 5 second connection timeout
            read_timeout=10,  # 10 second read timeout
        )

        # Reuse session for better connection pooling
        if DynamoDBStateStore._session is None:
            DynamoDBStateStore._session = boto3.Session()

        # Initialize DynamoDB client
        if endpoint_url:
            # Local development with DynamoDB Local
            logger.info(f"Connecting to DynamoDB Local at {endpoint_url}")
            # Use environment credentials if set, otherwise use dummy for local
            access_key = os.getenv("AWS_ACCESS_KEY_ID", "dummy")
            secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", "dummy")
            self.dynamodb = DynamoDBStateStore._session.resource(
                "dynamodb",
                config=boto_config,
                endpoint_url=endpoint_url,
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
            )
        else:
            # Production AWS DynamoDB
            logger.info(f"Connecting to AWS DynamoDB in region {self.region}")
            self.dynamodb = DynamoDBStateStore._session.resource("dynamodb", config=boto_config)

        # Initialize table reference (create if necessary on first operation)
        self.table = self.dynamodb.Table(self.table_name)
        self._table_initialized = False

        # Try to initialize table but don't fail if it doesn't exist yet
        try:
            self._init_table()
            self._table_initialized = True
        except ConnectionError:
            # Re-raise connection errors
            raise
        except Exception as e:
            # Log but don't fail - table will be created on first operation
            logger.info(f"Table initialization deferred: {e}")

    def _ensure_table_exists(self):
        """Ensure table exists before operations."""
        if not self._table_initialized:
            try:
                self._init_table()
                self._table_initialized = True
            except Exception as e:
                logger.debug(f"Table check: {e}")
                # Will be created on first write if needed

    def get_state(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve state for a user from DynamoDB."""
        try:
            response = self.table.get_item(
                Key={"userId": user_id, "stateType": "current"}
            )

            if "Item" not in response:
                return None

            # Deserialize the state
            state = self._deserialize_state(response["Item"])
            return state

        except ClientError as e:
            if e.response["Error"]["Code"] == "ResourceNotFoundException":
                # Table doesn't exist yet, try to create it
                logger.warning(
                    f"Table '{self.table_name}' not found, attempting to create..."
                )
                try:
                    self._init_table()
                    # Try again after creating table
                    return None  # Return None for first attempt after table creation
                except Exception as init_error:
                    logger.error(f"Failed to create table: {init_error}")
                    return None
            else:
                logger.error(f"Error getting state from DynamoDB: {e}")
                return None
        except Exception as e:
            logger.error(f"Unexpected error getting state: {e}")
            return None

    def save_state(self, user_id: str, state: Dict[str, Any]) -> bool:
        """Save state to DynamoDB."""
        try:
            # Serialize the state
            item = self._serialize_state(state)
            item.update(
                {
                    "userId": user_id,
                    "stateType": "current",
                    "updatedAt": datetime.utcnow().isoformat(),
                    "version": state.get("version", 0) + 1,
                }
            )

            # Save to DynamoDB
            self.table.put_item(Item=item)
            return True

        except ClientError as e:
            if e.response["Error"]["Code"] == "ResourceNotFoundException":
                # Table doesn't exist yet, try to create it
                logger.warning(
                    f"Table '{self.table_name}' not found, attempting to create..."
                )
                try:
                    self._init_table()
                    # Recreate table reference
                    self.table = self.dynamodb.Table(self.table_name)
                    # Try again after creating table
                    self.table.put_item(Item=item)
                    return True
                except Exception as init_error:
                    logger.error(f"Failed to create table or save state: {init_error}")
                    return False
            else:
                logger.error(f"Error saving state to DynamoDB: {e}")
                return False
        except Exception as e:
            logger.error(f"Unexpected error saving state: {e}")
            return False

    def delete_state(self, user_id: str) -> bool:
        """Delete state from DynamoDB."""
        try:
            # Delete current state
            self.table.delete_item(Key={"userId": user_id, "stateType": "current"})

            # Delete all snapshots
            self._delete_user_snapshots(user_id)
            return True

        except ClientError as e:
            logger.error(f"Error deleting state from DynamoDB: {e}")
            return False

    def create_initial_state(self) -> Dict[str, Any]:
        """Create an empty initial state."""
        return {
            "kalman_params": None,
            "last_state": None,
            "last_covariance": None,
            "last_timestamp": None,
            "last_accepted_timestamp": None,
            "last_source": None,
            "last_raw_weight": None,
            "measurement_history": [],
            "reset_events": [],
            "measurements_since_reset": 0,
            "adaptation_state": {},
            "version": 0,
        }

    def save_state_snapshot(self, user_id: str, timestamp: datetime) -> bool:
        """Save a snapshot to DynamoDB."""
        try:
            # Get current state
            current_state = self.get_state(user_id)
            if not current_state:
                # No state to snapshot yet
                return True  # Not an error, just nothing to snapshot

            # Create snapshot item
            snapshot = self._serialize_state(current_state)
            snapshot.update(
                {
                    "userId": user_id,
                    "stateType": f"snapshot_{timestamp.isoformat()}",
                    "snapshotTime": timestamp.isoformat(),
                    "ttl": int(
                        (timestamp + timedelta(days=10)).timestamp()
                    ),  # 10-day retention for replay support
                }
            )

            # Save to DynamoDB
            self.table.put_item(Item=snapshot)
            return True

        except ClientError as e:
            if e.response["Error"]["Code"] == "ResourceNotFoundException":
                # Table doesn't exist yet, try to create it
                logger.warning(
                    f"Table '{self.table_name}' not found, attempting to create..."
                )
                try:
                    self._init_table()
                    # Recreate table reference
                    self.table = self.dynamodb.Table(self.table_name)
                    # For now, just return True since there's no state to snapshot yet
                    return True
                except Exception as init_error:
                    logger.error(f"Failed to create table: {init_error}")
                    return False
            else:
                logger.error(f"Error saving snapshot to DynamoDB: {e}")
                return False
        except Exception as e:
            logger.error(f"Unexpected error saving snapshot: {e}")
            return False

    def restore_state_snapshot(self, user_id: str) -> bool:
        """Restore state from the latest snapshot."""
        try:
            # Query for the latest snapshot
            response = self.table.query(
                KeyConditionExpression="userId = :uid AND begins_with(stateType, :st)",
                ExpressionAttributeValues={":uid": user_id, ":st": "snapshot_"},
                ScanIndexForward=False,  # Descending order
                Limit=1,
            )

            if not response.get("Items"):
                return False

            snapshot_item = response["Items"][0]
            # Restore the snapshot as current state
            state = self._deserialize_state(snapshot_item)
            return self.save_state(user_id, state)

        except ClientError as e:
            logger.error(f"Error restoring snapshot from DynamoDB: {e}")
            return False

    def get_snapshot(
        self, user_id: str, timestamp: datetime
    ) -> Optional[Dict[str, Any]]:
        """Get the nearest snapshot before the given timestamp."""
        try:
            # Query snapshots
            response = self.table.query(
                KeyConditionExpression="userId = :uid AND stateType < :st",
                ExpressionAttributeValues={
                    ":uid": user_id,
                    ":st": f"snapshot_{timestamp.isoformat()}",
                },
                ScanIndexForward=False,  # Descending order
                Limit=1,
            )

            if not response.get("Items"):
                return None

            return self._deserialize_state(response["Items"][0])

        except ClientError as e:
            logger.error(f"Error getting snapshot from DynamoDB: {e}")
            return None

    def get_latest_snapshot(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the most recent snapshot for a user.

        Used by periodic snapshot logic to determine when to create next snapshot.

        Args:
            user_id: User identifier

        Returns:
            Latest snapshot dict or None if no snapshots exist
        """
        try:
            # Query for the latest snapshot
            response = self.table.query(
                KeyConditionExpression="userId = :uid AND begins_with(stateType, :st)",
                ExpressionAttributeValues={":uid": user_id, ":st": "snapshot_"},
                ScanIndexForward=False,  # Descending order (newest first)
                Limit=1,
            )

            if not response.get("Items"):
                return None

            return self._deserialize_state(response["Items"][0])

        except ClientError as e:
            logger.error(f"Error getting latest snapshot from DynamoDB: {e}")
            return None

    def check_and_restore_snapshot(
        self, user_id: str, buffer_start_time: datetime
    ) -> dict:
        """Check if a snapshot exists and restore it atomically."""
        snapshot = self.get_snapshot(user_id, buffer_start_time)
        if snapshot:
            # Restore the snapshot as current state
            if self.save_state(user_id, snapshot):
                return {
                    "success": True,
                    "snapshot": snapshot,
                    "snapshot_timestamp": snapshot.get(
                        "last_timestamp", buffer_start_time
                    ),
                    "user_id": user_id,
                }

        return {
            "success": False,
            "error": f"No snapshot found for user {user_id}",
            "user_id": user_id,
        }


    def _init_table(self):
        """Create DynamoDB table if it doesn't exist (mainly for local development)."""
        try:
            # Check if table exists
            self.dynamodb.Table(self.table_name).load()
            logger.info(f"DynamoDB table '{self.table_name}' exists")
            return  # Table exists, we're done
        except ClientError as e:
            if e.response["Error"]["Code"] == "ResourceNotFoundException":
                # Table doesn't exist, create it
                logger.info(f"Creating DynamoDB table '{self.table_name}'...")
                try:
                    table = self.dynamodb.create_table(
                        TableName=self.table_name,
                        KeySchema=[
                            {"AttributeName": "userId", "KeyType": "HASH"},
                            {"AttributeName": "stateType", "KeyType": "RANGE"},
                        ],
                        AttributeDefinitions=[
                            {"AttributeName": "userId", "AttributeType": "S"},
                            {"AttributeName": "stateType", "AttributeType": "S"},
                        ],
                        BillingMode="PAY_PER_REQUEST",  # On-demand billing
                    )
                    # Wait for table to be created and active
                    logger.info(
                        f"Waiting for table '{self.table_name}' to be active..."
                    )
                    table.wait_until_exists()
                    # Give it a moment to be fully ready
                    import time

                    time.sleep(1)
                    logger.info(
                        f"DynamoDB table '{self.table_name}' created successfully"
                    )
                except ClientError as create_error:
                    logger.error(f"Failed to create DynamoDB table: {create_error}")
                    raise
            else:
                logger.error(f"Error checking DynamoDB table: {e}")
                raise
        except Exception as e:
            # Check if it's a connection error
            if (
                "Connection" in str(e)
                or "reach" in str(e)
                or "Failed to establish" in str(e)
                or "Could not connect" in str(e)
            ):
                endpoint_url = os.getenv("DYNAMODB_ENDPOINT")
                if endpoint_url:
                    raise ConnectionError(
                        f"Cannot connect to DynamoDB Local at {endpoint_url}. "
                        "Please start DynamoDB Local first: docker-compose up -d dynamodb-local"
                    )
                else:
                    raise ConnectionError(
                        "Cannot connect to AWS DynamoDB. "
                        "Please check your AWS credentials and network connection."
                    )
            else:
                logger.error(f"Unexpected error initializing table: {e}")
                raise

    def export_to_csv(self, filepath: str) -> int:
        """Export all states to CSV (limited implementation for DynamoDB)."""
        import csv

        try:
            # Scan for all current states
            response = self.table.scan(
                FilterExpression="stateType = :st",
                ExpressionAttributeValues={":st": "current"},
            )

            users_exported = 0
            with open(filepath, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["user_id", "last_weight", "last_timestamp", "last_updated"]
                )

                for item in response.get("Items", []):
                    state = self._deserialize_state(item)
                    last_weight = None
                    if state.get("last_state"):
                        try:
                            if (
                                isinstance(state["last_state"], list)
                                and len(state["last_state"]) > 0
                            ):
                                last_weight = float(state["last_state"][0])
                        except (TypeError, ValueError, IndexError):
                            pass

                    writer.writerow(
                        [
                            item["userId"],
                            last_weight,
                            state.get("last_timestamp"),
                            item.get("updatedAt", ""),
                        ]
                    )
                    users_exported += 1

            logger.info(f"Exported {users_exported} users to {filepath}")
            return users_exported

        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
            return 0

    def _delete_user_snapshots(self, user_id: str):
        """Delete all snapshots for a user."""
        try:
            # Query all snapshots
            response = self.table.query(
                KeyConditionExpression="userId = :uid AND begins_with(stateType, :st)",
                ExpressionAttributeValues={":uid": user_id, ":st": "snapshot_"},
            )

            # Delete each snapshot
            for item in response.get("Items", []):
                self.table.delete_item(
                    Key={"userId": user_id, "stateType": item["stateType"]}
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
                # Convert numpy arrays to lists and ensure floats become Decimals
                list_value = value.tolist()
                serialized[key] = self._convert_floats_to_decimal(list_value)
            elif isinstance(value, datetime):
                # Convert datetime to ISO string
                serialized[key] = value.isoformat()
            elif isinstance(value, (float, np.float32, np.float64)):
                # Convert float to Decimal for DynamoDB
                if np.isnan(value) or np.isinf(value):
                    # DynamoDB doesn't support NaN or Inf
                    serialized[key] = None
                else:
                    serialized[key] = Decimal(str(value))
            elif isinstance(value, (int, np.int32, np.int64)):
                # Ensure integers are standard Python ints
                serialized[key] = int(value)
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
            if key in ["userId", "stateType", "ttl", "updatedAt", "version"]:
                continue  # Skip DynamoDB metadata

            if key in ["last_timestamp", "last_accepted_timestamp"]:
                # Convert ISO strings back to datetime
                if isinstance(value, str):
                    state[key] = datetime.fromisoformat(value)
                else:
                    state[key] = value
            elif key in ["last_state", "last_covariance"] and value is not None:
                # Convert lists back to numpy arrays for Kalman state
                if isinstance(value, list):
                    state[key] = np.array(self._convert_decimals_to_float(value))
                else:
                    state[key] = value
            elif key == "kalman_params" and value:
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
            if key in ["x", "P"] and isinstance(value, list):
                # Convert lists back to numpy arrays, handling Decimals
                deserialized[key] = np.array(self._convert_decimals_to_float(value))
            elif isinstance(value, Decimal):
                deserialized[key] = float(value)
            else:
                deserialized[key] = value

        return deserialized

    def _convert_floats_to_decimal(self, obj: Any) -> Any:
        """Recursively convert all floats to Decimal in a nested structure."""
        if isinstance(obj, (float, np.float32, np.float64)):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return Decimal(str(obj))
        elif isinstance(obj, (int, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, list):
            return [self._convert_floats_to_decimal(item) for item in obj]
        elif isinstance(obj, tuple):
            return [self._convert_floats_to_decimal(item) for item in obj]
        elif isinstance(obj, dict):
            return {
                key: self._convert_floats_to_decimal(value)
                for key, value in obj.items()
            }
        return obj

    def _serialize_value(self, value: Any, depth: int = 0) -> Any:
        """Serialize a single value with depth protection."""
        MAX_DEPTH = 10

        if depth > MAX_DEPTH:
            logger.warning(f"Max serialization depth {MAX_DEPTH} reached in value")
            return "__truncated__"

        if isinstance(value, (float, np.float32, np.float64)):
            if np.isnan(value) or np.isinf(value):
                return None
            return Decimal(str(value))
        elif isinstance(value, (int, np.int32, np.int64)):
            return int(value)
        elif isinstance(value, datetime):
            return value.isoformat()
        elif isinstance(value, np.ndarray):
            # Convert to list and handle floats
            list_value = value.tolist()
            return self._convert_floats_to_decimal(list_value)
        elif isinstance(value, dict):
            return self._serialize_state(value, depth + 1)
        elif isinstance(value, list):
            return [self._serialize_value(item, depth + 1) for item in value]
        return value

    def _convert_decimals_to_float(self, obj: Any) -> Any:
        """Recursively convert all Decimals to float in a nested structure."""
        if isinstance(obj, Decimal):
            return float(obj)
        elif isinstance(obj, list):
            return [self._convert_decimals_to_float(item) for item in obj]
        elif isinstance(obj, dict):
            return {
                key: self._convert_decimals_to_float(value)
                for key, value in obj.items()
            }
        return obj

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
        elif isinstance(value, list):
            return [self._deserialize_value(item) for item in value]
        return value

    def close_connections(self):
        """
        Close all connections and reset session.
        Called when resetting the database instance.
        """
        try:
            # Close any active connections
            if hasattr(self, 'dynamodb') and self.dynamodb:
                # DynamoDB resource doesn't have explicit close, but we can
                # reset the session to force new connections
                pass

            # Reset the class-level session to force new connections
            DynamoDBStateStore._session = None
            logger.debug("DynamoDB connections reset")
        except Exception as e:
            logger.warning(f"Error closing DynamoDB connections: {e}")

    @classmethod
    def reset_session(cls):
        """Reset the class-level session (useful for testing)."""
        cls._session = None
