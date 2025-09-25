#!/usr/bin/env python3
"""Test that all critical imports work correctly."""

import sys
import traceback
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_imports():
    """Test all critical imports."""

    failed = []
    passed = []

    imports_to_test = [
        # Core modules
        ("Core Constants", "from src.core.constants import KALMAN_DEFAULTS"),
        ("Core Exceptions", "from src.core.exceptions import StateValidationError"),
        ("Core Utils", "from src.core.processing.type_conversion import ensure_float"),
        # Processing modules
        ("Kalman Filter", "from src.core.processing.kalman import KalmanFilterManager"),
        ("Processor", "from src.core.processing.processor import process_measurement"),
        (
            "Quality Scorer",
            "from src.core.processing.unified_quality_scorer import UnifiedQualityScorer",
        ),
        (
            "Outlier Detection",
            "from src.core.processing.outlier_detection import OutlierDetector",
        ),
        (
            "Validation",
            "from src.core.processing.validation import DataQualityPreprocessor",
        ),
        # Database modules
        ("Database Base", "from src.core.database.base import StateStore"),
        ("Database", "from src.core.database import get_state_db"),
        (
            "DynamoDB Store",
            "from src.core.database.dynamodb_store import DynamoDBStateStore",
        ),
        # Replay modules
        ("Replay Manager", "from src.core.replay.replay_manager import ReplayManager"),
        ("Replay Buffer", "from src.core.replay.replay_buffer import ReplayBuffer"),
        # AWS modules
        ("Lambda Handler", "from src.aws.lambda_handler import handler"),
        ("API Models", "from src.aws.api.models import Measurement, ProcessRequest"),
        ("Config Manager", "from src.aws.config.config_manager import ConfigManager"),
        (
            "Weight Processor Service",
            "from src.aws.services.weight_processor_service import WeightProcessorService",
        ),
        # Local modules (optional, may not be needed for deployment)
        ("Local Main", "from src.local.main import load_config"),
    ]

    print("Testing imports...")
    print("=" * 50)

    for name, import_statement in imports_to_test:
        try:
            exec(import_statement)
            passed.append(name)
            print(f"✅ {name}")
        except ImportError as e:
            failed.append((name, str(e)))
            print(f"❌ {name}: {e}")
        except Exception as e:
            failed.append((name, f"Unexpected error: {e}"))
            print(f"❌ {name}: Unexpected error: {e}")

    print("=" * 50)
    print(f"\n📊 Results:")
    print(f"  Passed: {len(passed)}")
    print(f"  Failed: {len(failed)}")

    if failed:
        print(f"\n❌ Failed imports:")
        for name, error in failed:
            print(f"  - {name}: {error}")
        return False
    else:
        print(f"\n✅ All imports successful!")
        return True


if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
