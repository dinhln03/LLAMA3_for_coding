from typing import TypedDict

from cff.models.cloudfront_event import CloudFrontEvent


class Record(TypedDict):
    """Record of an event that raised a Lambda event."""

    cf: CloudFrontEvent
    """The CloudFront event that raised this Lambda event."""
