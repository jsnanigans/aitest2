#!/usr/bin/env python3
"""
Local Lambda testing runner.
Tests Lambda functions without deploying to AWS.
"""

import json
import os
import sys
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import traceback
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Set up test environment
os.environ['DB_BACKEND'] = 'memory'  # Use in-memory database for testing
os.environ['LOG_LEVEL'] = 'DEBUG'
os.environ['AWS_REGION'] = 'us-east-1'


class LambdaTester:
    """Local Lambda function tester."""

    def __init__(self, use_mock_dynamodb: bool = True):
        """
        Initialize tester.

        Args:
            use_mock_dynamodb: If True, use moto to mock DynamoDB
        """
        self.use_mock_dynamodb = use_mock_dynamodb
        self.handler = None
        self.mock_context = None

        # Setup mocks if requested
        if use_mock_dynamodb:
            self._setup_mocks()

    def _setup_mocks(self):
        """Set up AWS service mocks."""
        try:
            import boto3
            from moto import mock_dynamodb

            # Start DynamoDB mock
            self.dynamodb_mock = mock_dynamodb()
            self.dynamodb_mock.start()

            # Create mock table
            dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
            table = dynamodb.create_table(
                TableName='weight-processor-state',
                KeySchema=[
                    {'AttributeName': 'userId', 'KeyType': 'HASH'},
                    {'AttributeName': 'stateType', 'KeyType': 'RANGE'}
                ],
                AttributeDefinitions=[
                    {'AttributeName': 'userId', 'AttributeType': 'S'},
                    {'AttributeName': 'stateType', 'AttributeType': 'S'}
                ],
                BillingMode='PAY_PER_REQUEST'
            )

            print(f"✓ Created mock DynamoDB table: {table.table_name}")

        except ImportError:
            print("⚠️  Moto not installed. Run: pip install moto")
            print("   Continuing without DynamoDB mocks...")

    def load_handler(self):
        """Load the Lambda handler."""
        try:
            from src.lambda_handler import handler
            self.handler = handler
            print("✓ Lambda handler loaded successfully")
        except ImportError as e:
            print(f"✗ Failed to load handler: {e}")
            print("\nMake sure you've created src/lambda_handler.py")
            sys.exit(1)

        # Create mock Lambda context
        self.mock_context = MagicMock()
        self.mock_context.function_name = "weight-processor-test"
        self.mock_context.function_version = "$LATEST"
        self.mock_context.invoked_function_arn = "arn:aws:lambda:us-east-1:123456789012:function:test"
        self.mock_context.memory_limit_in_mb = "1024"
        self.mock_context.aws_request_id = "test-request-id"
        self.mock_context.log_group_name = "/aws/lambda/test"
        self.mock_context.log_stream_name = "test-stream"

        def get_remaining_time_in_millis():
            return 300000  # 5 minutes

        self.mock_context.get_remaining_time_in_millis = get_remaining_time_in_millis

    def test_endpoint(self, event: Dict[str, Any], description: str) -> Dict[str, Any]:
        """
        Test a single endpoint.

        Args:
            event: Lambda event
            description: Test description

        Returns:
            Response from handler
        """
        print(f"\n{'='*60}")
        print(f"Testing: {description}")
        print(f"{'='*60}")

        # Extract request info
        method = event.get('httpMethod', 'GET')
        path = event.get('path', '/')
        user_id = event.get('pathParameters', {}).get('userId', 'unknown')

        print(f"Request: {method} {path}")
        print(f"User ID: {user_id}")

        if event.get('body'):
            try:
                body = json.loads(event['body'])
                if 'measurements' in body:
                    print(f"Measurements: {len(body['measurements'])}")
            except:
                print("Body: <invalid JSON>")

        # Call handler
        try:
            start_time = datetime.now()
            response = self.handler(event, self.mock_context)
            duration = (datetime.now() - start_time).total_seconds() * 1000

            # Parse response
            status_code = response.get('statusCode', 0)
            body = response.get('body', '')

            # Print results
            print(f"\nResponse Status: {status_code}")
            print(f"Duration: {duration:.2f}ms")

            # Parse and display body
            if body:
                try:
                    body_data = json.loads(body)
                    print(f"Response Body:")

                    # Pretty print based on content
                    if 'error' in body_data:
                        print(f"  ❌ Error: {body_data['error']}")
                    elif 'measurements' in body_data:
                        accepted = body_data.get('acceptedCount', 0)
                        rejected = body_data.get('rejectedCount', 0)
                        print(f"  ✓ Accepted: {accepted}")
                        print(f"  ✗ Rejected: {rejected}")
                        if 'stateUpdate' in body_data:
                            update = body_data['stateUpdate']
                            print(f"  State: {update.get('previousWeight')} → {update.get('currentWeight')}")
                    else:
                        # Generic pretty print
                        for key, value in body_data.items():
                            if isinstance(value, dict):
                                print(f"  {key}:")
                                for k, v in value.items():
                                    print(f"    {k}: {v}")
                            elif isinstance(value, list):
                                print(f"  {key}: [{len(value)} items]")
                            else:
                                print(f"  {key}: {value}")

                except json.JSONDecodeError:
                    print(f"  Raw: {body[:200]}...")

            # Color-coded status
            if status_code < 300:
                print(f"✅ Test passed")
            elif status_code < 500:
                print(f"⚠️  Client error (expected for error test)")
            else:
                print(f"❌ Server error")

            return response

        except Exception as e:
            print(f"\n❌ Handler crashed: {e}")
            print(traceback.format_exc())
            return {"statusCode": 500, "body": json.dumps({"error": str(e)})}

    def run_all_tests(self):
        """Run all standard tests."""
        from tests.local.mock_events import (
            get_process_event_single,
            get_process_event_batch,
            get_cleanup_event,
            get_state_event,
            get_delete_state_event,
            get_invalid_weight_event,
            get_process_event_historical_conflict
        )

        tests = [
            (get_process_event_single(), "Process single measurement"),
            (get_process_event_batch(), "Process batch of measurements"),
            (get_cleanup_event(), "Cleanup all historical data"),
            (get_state_event(), "Get user state"),
            (get_delete_state_event(), "Delete user state"),
            (get_invalid_weight_event(), "Invalid weight (error test)"),
            (get_process_event_historical_conflict(), "Historical conflict test")
        ]

        results = {
            "passed": 0,
            "failed": 0,
            "errors": 0
        }

        for event, description in tests:
            response = self.test_endpoint(event, description)
            status = response.get('statusCode', 500)

            if status < 300:
                results["passed"] += 1
            elif status < 500:
                # Client errors might be expected
                if "error test" in description.lower() or "conflict" in description.lower():
                    results["passed"] += 1
                else:
                    results["failed"] += 1
            else:
                results["errors"] += 1

        # Summary
        print(f"\n{'='*60}")
        print(f"Test Summary")
        print(f"{'='*60}")
        print(f"✅ Passed: {results['passed']}")
        print(f"❌ Failed: {results['failed']}")
        print(f"💥 Errors: {results['errors']}")
        print(f"Total: {sum(results.values())}")

        return results["failed"] == 0 and results["errors"] == 0

    def test_custom_event(self, event_file: str):
        """Test with a custom event file."""
        with open(event_file) as f:
            event = json.load(f)

        return self.test_endpoint(event, f"Custom event from {event_file}")

    def interactive_mode(self):
        """Run in interactive mode."""
        print("\n🚀 Lambda Tester - Interactive Mode")
        print("Commands:")
        print("  1. Process single measurement")
        print("  2. Process batch")
        print("  3. Cleanup")
        print("  4. Get state")
        print("  5. Run all tests")
        print("  6. Load custom event file")
        print("  q. Quit")

        from tests.local.mock_events import (
            get_process_event_single,
            get_process_event_batch,
            get_cleanup_event,
            get_state_event
        )

        while True:
            choice = input("\nSelect option: ").strip()

            if choice == '1':
                self.test_endpoint(get_process_event_single(), "Process single")
            elif choice == '2':
                self.test_endpoint(get_process_event_batch(), "Process batch")
            elif choice == '3':
                self.test_endpoint(get_cleanup_event(), "Cleanup")
            elif choice == '4':
                user_id = input("Enter user ID [test-user-001]: ").strip() or "test-user-001"
                event = get_state_event()
                event['pathParameters']['userId'] = user_id
                self.test_endpoint(event, f"Get state for {user_id}")
            elif choice == '5':
                self.run_all_tests()
            elif choice == '6':
                filepath = input("Enter event file path: ").strip()
                if Path(filepath).exists():
                    self.test_custom_event(filepath)
                else:
                    print(f"File not found: {filepath}")
            elif choice.lower() == 'q':
                break
            else:
                print("Invalid option")

    def cleanup(self):
        """Clean up mocks."""
        if self.use_mock_dynamodb and hasattr(self, 'dynamodb_mock'):
            self.dynamodb_mock.stop()
            print("✓ Cleaned up DynamoDB mock")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test Lambda functions locally")
    parser.add_argument(
        "--event",
        help="Path to custom event JSON file"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--no-mock",
        action="store_true",
        help="Don't use mocked AWS services"
    )
    parser.add_argument(
        "--dynamodb",
        choices=["memory", "local", "mock"],
        default="memory",
        help="DynamoDB backend (memory, local, or mock)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    # Configure environment
    if args.dynamodb == "memory":
        os.environ['DB_BACKEND'] = 'memory'
        use_mock = False
    elif args.dynamodb == "local":
        os.environ['DB_BACKEND'] = 'dynamodb'
        os.environ['DYNAMODB_ENDPOINT'] = 'http://localhost:8000'
        use_mock = False
    else:
        os.environ['DB_BACKEND'] = 'dynamodb'
        use_mock = True

    if args.verbose:
        os.environ['LOG_LEVEL'] = 'DEBUG'

    # Create tester
    tester = LambdaTester(use_mock_dynamodb=use_mock and not args.no_mock)

    try:
        # Load handler
        tester.load_handler()

        # Run tests
        if args.event:
            # Test custom event
            tester.test_custom_event(args.event)
        elif args.interactive:
            # Interactive mode
            tester.interactive_mode()
        else:
            # Run all tests
            success = tester.run_all_tests()
            sys.exit(0 if success else 1)

    finally:
        tester.cleanup()


if __name__ == "__main__":
    main()