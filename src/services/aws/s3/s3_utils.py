import boto3

from src.services.logger.logger_config import logger


def list_files(bucket_name, prefix):
    s3 = boto3.client('s3')
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    return response.get('Contents', [])


def upload_file(bucket_name, file_name, object_name=None):
    if object_name is None:
        object_name = file_name

    s3 = boto3.client('s3')
    try:
        response = s3.upload_file(file_name, bucket_name, object_name)
    except Exception as e:
        logger.error(
            f"Failed to upload file {file_name} to bucket {bucket_name} with error: {e}")
        return False
    return True
