"""Define endpoints related to user reports."""
import logging
from typing import Any, Dict

from .helpers.report import Report

_LOGGER: logging.Logger = logging.getLogger(__name__)


class UserReport(Report):
    """Define a user report object."""

    async def status_by_coordinates(
        self, latitude: float, longitude: float
    ) -> Dict[str, Any]:
        """Get symptom data for the location nearest to the user's lat/lon."""
        return await self.nearest_by_coordinates(latitude, longitude)

    async def status_by_zip(self, zip_code: str) -> Dict[str, Any]:
        """Get symptom data for the provided ZIP code."""
        try:
            location = next(
                (d for d in await self.user_reports() if d["zip"] == zip_code)
            )
        except StopIteration:
            return {}

        return await self.status_by_coordinates(
            float(location["latitude"]), float(location["longitude"])
        )
