"""Define AWS storage backends for media files."""

from storages.backends.s3boto3 import S3Boto3Storage


def MediaBackend():
    """Media storage backend."""
    return S3Boto3Storage(location="media")
