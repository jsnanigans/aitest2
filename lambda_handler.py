"""
AWS Lambda handler for weight processing service.
"""

import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def lambda_handler(event, context):
    """
    AWS Lambda handler function.

    Args:
        event: Lambda event containing:
            - csv_path: S3 path or local path to CSV file
            - output_bucket: S3 bucket for output (optional)
            - config: Processing configuration (optional)
        context: Lambda context

    Returns:
        Response with processing status and statistics
    """
    try:
        # Import here to ensure proper initialization in Lambda environment
        from main import stream_process, load_config
        from src.database import get_state_db, reset_db_instance

        # Ensure DynamoDB is configured for Lambda
        import os
        if not os.getenv('DYNAMODB_TABLE_NAME'):
            os.environ['DYNAMODB_TABLE_NAME'] = 'weight-processor-state'

        # Get configuration
        config_override = event.get('config', {})
        config = load_config()

        # Merge event config with default config
        if config_override:
            for key, value in config_override.items():
                if isinstance(value, dict) and key in config:
                    config[key].update(value)
                else:
                    config[key] = value

        # Disable visualizations in Lambda (no GUI support)
        config['visualization']['enabled'] = False

        # Use /tmp for output in Lambda
        output_dir = '/tmp/output'
        Path(output_dir).mkdir(exist_ok=True)

        # Get CSV path from event
        csv_path = event.get('csv_path')
        if not csv_path:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'csv_path is required'})
            }

        # Handle S3 paths (requires additional S3 download logic)
        if csv_path.startswith('s3://'):
            # This would require boto3 and S3 download logic
            # For now, return an error
            return {
                'statusCode': 501,
                'body': json.dumps({'error': 'S3 paths not yet implemented'})
            }

        # Check if file exists
        if not Path(csv_path).exists():
            return {
                'statusCode': 404,
                'body': json.dumps({'error': f'File not found: {csv_path}'})
            }

        # Process the data
        logger.info(f"Processing CSV: {csv_path}")
        start_time = datetime.now()

        try:
            user_results, stats = stream_process(
                csv_path=csv_path,
                output_dir=output_dir,
                config=config,
                filtered_output=event.get('filtered_output'),
                debug=event.get('debug', False)
            )

            # Calculate summary statistics
            elapsed = (datetime.now() - start_time).total_seconds()

            response = {
                'statusCode': 200,
                'body': json.dumps({
                    'success': True,
                    'statistics': {
                        'total_rows': stats.get('total_rows', 0),
                        'users_processed': len(user_results),
                        'accepted': stats.get('accepted', 0),
                        'rejected': stats.get('rejected', 0),
                        'processing_time_seconds': elapsed,
                        'rows_per_second': stats.get('total_rows', 0) / elapsed if elapsed > 0 else 0
                    },
                    'message': f"Successfully processed {len(user_results)} users"
                }, default=str)
            }

            # Clean up database connections
            db = get_state_db()
            if hasattr(db, 'close_connections'):
                db.close_connections()

            return response

        except Exception as e:
            logger.error(f"Processing error: {str(e)}", exc_info=True)

            # Clean up on error
            try:
                reset_db_instance()
            except:
                pass

            return {
                'statusCode': 500,
                'body': json.dumps({
                    'success': False,
                    'error': str(e),
                    'message': 'Processing failed'
                })
            }

    except Exception as e:
        logger.error(f"Handler error: {str(e)}", exc_info=True)
        return {
            'statusCode': 500,
            'body': json.dumps({
                'success': False,
                'error': str(e),
                'message': 'Handler initialization failed'
            })
        }


def process_s3_event(event, context):
    """
    Process S3 event triggers (when CSV is uploaded to S3).

    Args:
        event: S3 event
        context: Lambda context

    Returns:
        Processing response
    """
    try:
        import boto3

        s3 = boto3.client('s3')

        # Extract S3 information from event
        bucket = event['Records'][0]['s3']['bucket']['name']
        key = event['Records'][0]['s3']['object']['key']

        # Download file to /tmp
        local_path = f'/tmp/{os.path.basename(key)}'
        s3.download_file(bucket, key, local_path)

        logger.info(f"Downloaded s3://{bucket}/{key} to {local_path}")

        # Process using main handler
        processing_event = {
            'csv_path': local_path,
            'output_bucket': bucket,
            'output_prefix': f"processed/{os.path.splitext(key)[0]}/"
        }

        result = lambda_handler(processing_event, context)

        # Clean up downloaded file
        try:
            os.remove(local_path)
        except:
            pass

        return result

    except Exception as e:
        logger.error(f"S3 processing error: {str(e)}", exc_info=True)
        return {
            'statusCode': 500,
            'body': json.dumps({
                'success': False,
                'error': str(e),
                'message': 'S3 processing failed'
            })
        }