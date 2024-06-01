"""
IGN Instituto Geográfico Nacional Sismología Feed.

Fetches GeoRSS feed from IGN Instituto Geográfico Nacional Sismología.
"""
from datetime import datetime
from typing import Optional

import dateparser as dateparser
from georss_client import FeedEntry, GeoRssFeed
from georss_client.consts import CUSTOM_ATTRIBUTE
from georss_client.feed_manager import FeedManagerBase

ATTRIBUTION = "Instituto Geográfico Nacional"

IMAGE_URL_PATTERN = (
    "http://www.ign.es/web/resources/sismologia/www/"
    "dir_images_terremotos/detalle/{}.gif"
)

REGEXP_ATTR_MAGNITUDE = r"magnitud (?P<{}>[^ ]+) ".format(CUSTOM_ATTRIBUTE)
REGEXP_ATTR_REGION = r"magnitud [^ ]+ en (?P<{}>[A-ZÁÉÓÜÑ0-9 \-\.]+) en".format(
    CUSTOM_ATTRIBUTE
)
REGEXP_ATTR_PUBLISHED_DATE = r"-Info.terremoto: (?P<{}>.+)$".format(CUSTOM_ATTRIBUTE)
REGEXP_ATTR_SHORT_ID = (
    r"http:\/\/www\.ign\.es\/web\/ign\/portal\/"
    r"sis-catalogo-terremotos\/-\/catalogo-terremotos\/"
    r"detailTerremoto\?evid=(?P<{}>\w+)$".format(CUSTOM_ATTRIBUTE)
)

URL = "http://www.ign.es/ign/RssTools/sismologia.xml"


class IgnSismologiaFeedManager(FeedManagerBase):
    """Feed Manager for IGN Sismología feed."""

    def __init__(
        self,
        generate_callback,
        update_callback,
        remove_callback,
        coordinates,
        filter_radius=None,
        filter_minimum_magnitude=None,
    ):
        """Initialize the IGN Sismología Feed Manager."""
        feed = IgnSismologiaFeed(
            coordinates,
            filter_radius=filter_radius,
            filter_minimum_magnitude=filter_minimum_magnitude,
        )
        super().__init__(feed, generate_callback, update_callback, remove_callback)


class IgnSismologiaFeed(GeoRssFeed):
    """IGN Sismología feed."""

    def __init__(
        self, home_coordinates, filter_radius=None, filter_minimum_magnitude=None
    ):
        """Initialise this service."""
        super().__init__(home_coordinates, URL, filter_radius=filter_radius)
        self._filter_minimum_magnitude = filter_minimum_magnitude

    def __repr__(self):
        """Return string representation of this feed."""
        return "<{}(home={}, url={}, radius={}, magnitude={})>".format(
            self.__class__.__name__,
            self._home_coordinates,
            self._url,
            self._filter_radius,
            self._filter_minimum_magnitude,
        )

    def _new_entry(self, home_coordinates, rss_entry, global_data):
        """Generate a new entry."""
        return IgnSismologiaFeedEntry(home_coordinates, rss_entry)

    def _filter_entries(self, entries):
        """Filter the provided entries."""
        entries = super()._filter_entries(entries)
        if self._filter_minimum_magnitude:
            # Return only entries that have an actual magnitude value, and
            # the value is equal or above the defined threshold.
            return list(
                filter(
                    lambda entry: entry.magnitude
                    and entry.magnitude >= self._filter_minimum_magnitude,
                    entries,
                )
            )
        return entries


class IgnSismologiaFeedEntry(FeedEntry):
    """IGN Sismología feed entry."""

    def __init__(self, home_coordinates, rss_entry):
        """Initialise this service."""
        super().__init__(home_coordinates, rss_entry)

    @property
    def attribution(self) -> str:
        """Return the attribution of this entry."""
        return ATTRIBUTION

    @property
    def published(self) -> Optional[datetime]:
        """Return the published date of this entry."""
        published_date = self._search_in_title(REGEXP_ATTR_PUBLISHED_DATE)
        if published_date:
            published_date = dateparser.parse(published_date)
        return published_date

    @property
    def magnitude(self) -> Optional[float]:
        """Return the magnitude of this entry."""
        magnitude = self._search_in_description(REGEXP_ATTR_MAGNITUDE)
        if magnitude:
            magnitude = float(magnitude)
        return magnitude

    @property
    def region(self) -> Optional[float]:
        """Return the region of this entry."""
        return self._search_in_description(REGEXP_ATTR_REGION)

    def _short_id(self) -> Optional[str]:
        """Return the short id of this entry."""
        return self._search_in_external_id(REGEXP_ATTR_SHORT_ID)

    @property
    def image_url(self) -> Optional[str]:
        """Return the image url of this entry."""
        short_id = self._short_id()
        if short_id:
            return IMAGE_URL_PATTERN.format(short_id)
        return None
