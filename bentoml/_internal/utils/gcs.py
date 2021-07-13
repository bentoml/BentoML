import logging
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def is_gcs_url(url):
    """
    Check if the url is a gcs url
    'gs://' is the standard way for Google Cloud URI
    Source: https://cloud.google.com/storage/docs/gsutil
    """
    try:
        return urlparse(url).scheme in ["gs"]
    except ValueError:
        return False
