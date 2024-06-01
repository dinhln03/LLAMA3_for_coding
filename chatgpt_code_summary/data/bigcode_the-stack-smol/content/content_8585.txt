"""Database exceptions."""


class BaseError(Exception):
    """The base exception."""


class NotFoundError(BaseError):
    """When an item was not found in the database."""
