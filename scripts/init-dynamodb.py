#!/usr/bin/env python3
"""
Initialize DynamoDB table for weight processor.
Run this before processing to ensure table exists.
"""

import os
import sys
import time
import boto3
from botocore.exceptions import ClientError

def create_table():
    """Create the DynamoDB table if it doesn't exist."""

    # Configuration
    table_name = os.getenv('DYNAMODB_TABLE_NAME', 'weight-processor-state')
    endpoint_url = os.getenv('DYNAMODB_ENDPOINT', 'http://localhost:8000')
    region = os.getenv('AWS_REGION', 'us-east-1')

    print(f"Initializing DynamoDB table '{table_name}'...")

    # Create DynamoDB resource
    if endpoint_url:
        print(f"Using DynamoDB Local at {endpoint_url}")
        dynamodb = boto3.resource(
            'dynamodb',
            region_name=region,
            endpoint_url=endpoint_url,
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID', 'dummy'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY', 'dummy')
        )
    else:
        print(f"Using AWS DynamoDB in region {region}")
        dynamodb = boto3.resource('dynamodb', region_name=region)

    try:
        # Check if table exists
        table = dynamodb.Table(table_name)
        table.load()
        print(f"✓ Table '{table_name}' already exists")
        print(f"  Status: {table.table_status}")
        print(f"  Item count: {table.item_count}")
        return True

    except ClientError as e:
        if e.response['Error']['Code'] == 'ResourceNotFoundException':
            # Table doesn't exist, create it
            print(f"Creating table '{table_name}'...")

            try:
                table = dynamodb.create_table(
                    TableName=table_name,
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

                # Wait for table to be created
                print("Waiting for table to become active...")
                table.wait_until_exists()

                # Give it a moment to be fully ready
                time.sleep(2)

                # Verify table is active
                table.reload()
                if table.table_status == 'ACTIVE':
                    print(f"✓ Table '{table_name}' created successfully")
                    return True
                else:
                    print(f"⚠ Table status: {table.table_status}")
                    return False

            except Exception as create_error:
                print(f"✗ Failed to create table: {create_error}")
                return False
        else:
            print(f"✗ Error checking table: {e}")
            return False

    except Exception as e:
        # Check if it's a connection error
        if 'Connection' in str(e) or 'reach' in str(e) or 'Failed to establish' in str(e):
            print(f"✗ Cannot connect to DynamoDB")
            if endpoint_url:
                print(f"  Make sure DynamoDB Local is running at {endpoint_url}")
                print("  Run: docker-compose up -d dynamodb-local")
            else:
                print("  Check your AWS credentials and network connection")
            return False
        else:
            print(f"✗ Unexpected error: {e}")
            return False

if __name__ == "__main__":
    # Set up environment for local development if not already set
    if not os.getenv('DYNAMODB_ENDPOINT') and not os.getenv('AWS_LAMBDA_FUNCTION_NAME'):
        os.environ['DYNAMODB_ENDPOINT'] = 'http://localhost:8000'
        os.environ['AWS_ACCESS_KEY_ID'] = 'local'
        os.environ['AWS_SECRET_ACCESS_KEY'] = 'local'

    success = create_table()
    sys.exit(0 if success else 1)