"""
Contains Catalyst DAO implementations.
"""

from django.conf import settings
from restclients.mock_http import MockHTTP
from restclients.dao_implementation import get_timeout
from restclients.dao_implementation.live import get_con_pool, get_live_url
from restclients.dao_implementation.mock import get_mockdata_url
import datetime
import hashlib
import pytz


class File(object):
    """
    The File DAO implementation returns generally static content.  Use this
    DAO with this configuration:

    RESTCLIENTS_CANVAS_DAO_CLASS =
    'restclients.dao_implementation.catalyst.File'
    """
    def getURL(self, url, headers):
        return get_mockdata_url("catalyst", "file", url, headers)


class Live(object):
    """
    This DAO provides real data.  It requires further configuration, e.g.

    For cert auth:
    RESTCLIENTS_CATALYST_CERT_FILE='/path/to/an/authorized/cert.cert',
    RESTCLIENTS_CATALYST_KEY_FILE='/path/to/the/certs_key.key',

    SolAuth Authentication (personal only):
    RESTCLIENTS_CATALYST_SOL_AUTH_PUBLIC_KEY='public_key'
    RESTCLIENTS_CATALYST_SOL_AUTH_PRIVATE_KEY='12345'

    SolAuth tokens are available at https://catalyst.uw.edu/rest_user

    For an alternate host:
    RESTCLIENTS_CATALYST_HOST = 'https://my-dev-server/'

    """
    pool = None

    def getURL(self, url, headers):
        host = settings.RESTCLIENTS_CATALYST_HOST

        if hasattr(settings, "RESTCLIENTS_CATALYST_CERT_FILE"):
            Live.pool = get_con_pool(host,
                                     settings.RESTCLIENTS_CATALYST_KEY_FILE,
                                     settings.RESTCLIENTS_CATALYST_CERT_FILE,
                                     socket_timeout=get_timeout("catalyst"))

        else:
            Live.pool = get_con_pool(host,
                                     socket_timeout=get_timeout("catalyst"))

        if hasattr(settings, "RESTCLIENTS_CATALYST_SOL_AUTH_PRIVATE_KEY"):
            # Use js_rest instead of rest, to avoid certificate issues
            url = url.replace("/rest/", "/js_rest/")
            now_with_tz = datetime.datetime.now(pytz.utc).strftime(
                "%a %b %d %H:%M:%S %Z %Y")
            header_base = "%s\nGET\n%s\n%s\n" % (
                settings.RESTCLIENTS_CATALYST_SOL_AUTH_PRIVATE_KEY,
                url,
                now_with_tz
            )

            public_key = settings.RESTCLIENTS_CATALYST_SOL_AUTH_PUBLIC_KEY
            hashed = hashlib.sha1(header_base).hexdigest()
            headers["Authorization"] = "SolAuth %s:%s" % (public_key, hashed)
            headers["Date"] = now_with_tz

        return get_live_url(Live.pool, "GET", host, url, headers=headers)
