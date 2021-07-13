import logging
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def is_s3_url(url):
    """
    Check if url is an s3, s3n, or s3a url
    """
    try:
        return urlparse(url).scheme in ["s3", "s3n", "s3a"]
    except ValueError:
        return False


def create_s3_bucket_if_not_exists(bucket_name, region):
    import boto3
    from botocore.exceptions import ClientError

    s3_client = boto3.client('s3', region)
    try:
        s3_client.get_bucket_acl(Bucket=bucket_name)
        logger.debug("Found bucket %s in region %s already exist", bucket_name, region)
    except ClientError as error:
        if error.response and error.response['Error']['Code'] == 'NoSuchBucket':
            logger.debug('Creating s3 bucket: %s in region %s', bucket_name, region)

            # NOTE: boto3 will raise ClientError(InvalidLocationConstraint) if
            # `LocationConstraint` is set to `us-east-1` region.
            # https://github.com/boto/boto3/issues/125.
            # This issue still show up in  boto3 1.13.4(May 6th 2020)
            try:
                s3_client.create_bucket(
                    Bucket=bucket_name,
                    CreateBucketConfiguration={'LocationConstraint': region},
                )
            except ClientError as s3_error:
                if (
                    s3_error.response
                    and s3_error.response['Error']['Code']
                    == 'InvalidLocationConstraint'
                ):
                    logger.debug(
                        'Special s3 region: %s, will attempt create bucket without '
                        '`LocationConstraint`',
                        region,
                    )
                    s3_client.create_bucket(Bucket=bucket_name)
                else:
                    raise s3_error
        else:
            raise error
