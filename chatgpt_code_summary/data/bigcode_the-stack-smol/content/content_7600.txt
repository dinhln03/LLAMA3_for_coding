# MIT License, Copyright (c) 2020 Bob van den Heuvel
# https://github.com/bheuvel/transip/blob/main/LICENSE
"""Interface with the TransIP API, specifically DNS record management."""
import logging
from enum import Enum
from pathlib import Path
from time import sleep
from typing import Dict, Union

import requests

from transip_dns import __project__, __version__
from transip_dns.accesstoken import AccessToken

logger = logging.getLogger(__name__)

DNS_RECORD_TYPES = ["A", "AAAA", "CNAME", "MX", "NS", "TXT", "SRV", "SSHFP", "TLSA"]


class DnsEntry(object):
    """Class matching the TransIP dnsEntry."""

    def __init__(
        self,
        content: str = None,
        expire: int = None,
        name: str = None,
        rtype: str = None,
    ):
        """Initialize the DnsEntry object.

        Closely represent the TransIP dnsEntry object

        :param content: content (rdata) corresponding to the record type
                        (e.g. ip), defaults to None
        :type content: str, optional
        :param expire: Time To Live (TTL) of the record, defaults to None
        :type expire: int, optional
        :param name: name of the record, defaults to None
        :type name: str, optional
        :param rtype: one of the (allowed) record types (see DNS_RECORD_TYPES),
                      defaults to None
        :type rtype: str, optional
        """
        self.content = content
        self.expire = expire
        self.name = name
        self.rtype = None
        self.rtype = rtype

    def __repr__(self) -> str:
        """Represent the TransIP definition of a dnsEntry object.

        The dnsEntry object is specified as a JSON object

        :return: JSON representation of the record according to the dnsEntry
        :rtype: str
        """
        return {
            "dnsEntry": {
                "name": self.name,
                "expire": self.expire,
                "type": self.rtype,
                "content": self.content,
            }
        }


class RecordState(Enum):
    """Enumeration of record states.

    When searching for records, these are the possible states.

    NOTFOUND: The record is not present
    FOUND_SAME: Record is present and the content is (already) the same
    FOUND_DIFFERENT: Record is present, but with different content
    FOUND_NO_REQUEST_DATA: If the content of the (requested) dns_record is empty.
                           This may occur when deleting a record (just) by name.

    :param Enum: Parent class to create an enumeration
    :type Enum: Enum
    """

    FOUND_SAME = 1
    FOUND_DIFFERENT = 2
    FOUND_NO_REQUEST_DATA = 4
    NOTFOUND = 3


class DnsRecord(DnsEntry):
    """DNS Record encapsulation with ip query and data checking.

    Initializes the object, potentially search for the IP address and
    check if the record type is allowed.

    :param DnsEntry: Parent class to enhance
    :type DnsEntry: DnsEntry
    """

    def __init__(
        self,
        name: str,
        rtype: str,
        expire: str,
        content: str,
        zone: str,
        query_data: Union[str, None] = None,
    ) -> None:
        """Initialize the DnsRecord object with safety checks.

        :param name: name of the DNS record
        :type name: str
        :param rtype: type of the DNS record
        :type rtype: str
        :param expire: TTL of the DNS record
        :type expire: str
        :param content: content of the DNS record
        :type content: str
        :param zone: Zone or domain of the DNS record
        :type zone: str
        :param query_data: url which produces the exact data to be used as
                           content, defaults to None
        :type query_data: Union[str, None], optional
        :raises ValueError: Raise an error if an invalid record type is specified
        """
        if rtype is not None:
            if not rtype.upper() in DNS_RECORD_TYPES:
                raise ValueError(
                    f"Type '{rtype}' is not one of the "
                    f"allowed record types ({DNS_RECORD_TYPES})"
                )
        super().__init__(content=content, expire=expire, name=name, rtype=rtype)

        self.zone = zone
        self.fqdn = f"{self.name}.{self.zone}"
        if query_data:
            self.content = DnsRecord.query_for_content(query_data)
            logger.info(f"Resolved record data to be used: '{self.content}'")
        self.record_state = None

    @property
    def dnsentry(self):
        """Return the TransIP representation of the dnsEntry object."""
        return super().__repr__()

    @staticmethod
    def query_for_content(query_url: str) -> str:
        """Retrieve the ip address from the "current" location.

        By default it will query for an ip (v4/v6) address,
        but may be used for other data as well

        :param query_url: url which produces the exact data
                             to be used as content
        :type query_url: str
        :raises RequestsRaisedException: raised for connection errors with the server
        :raises Non200Response: raised when server does not respond "OK" (200)
        :return: the resolved ip address, or whatever may be
                 returned by a custom provided url
        :rtype: str
        """
        my_ip = None
        try:
            ip_query = requests.get(query_url)
        except Exception as e:
            raise RequestsRaisedException(
                "Error in request for Internet ip address; "
            ) from e

        if ip_query.status_code == 200:
            my_ip = ip_query.text.strip()
        else:
            raise Non200Response(
                (
                    "Could not resolve Internet ip address (non 200 response); "
                    f"{ip_query.status_code}: {ip_query.reason}"
                )
            )

        return my_ip


class KeyFileLoadException(Exception):
    """Provided private_key is is not a valid path, nor a valid key format."""

    pass


class RequestsRaisedException(Exception):
    """Error occurred in requesting an url for the Internet ip address."""

    pass


class Non200Response(Exception):
    """Request for the Internet ip address resulted in a non 200 response."""

    pass


class TransipInterface:
    """Encapsulation of connection with TransIP."""

    def __init__(
        self,
        login: str = None,
        private_key_pem: str = None,
        private_key_pem_file: Path = None,
        access_token: str = None,
        expiration_time: int = 60,
        read_only: bool = False,
        global_key: bool = False,
        label: str = f"{__project__} {__version__}",
        authentication_url: str = "https://api.transip.nl/v6/auth",
        root_endpoint: str = "https://api.transip.nl/v6",
        connection_timeout: int = 30,
        retry: int = 3,
        retry_delay: float = 5,
    ):
        """Initialize the interface with TransIP.

        :param login: the TransIP login name, defaults to None
        :type login: str, optional
        :param private_key_pem: the private key as string, defaults to None
        :type private_key_pem: str, optional
        :param private_key_pem_file: file location of the private key, defaults to None
        :type private_key_pem_file: Path, optional
        :param access_token: JSON Web Token, defaults to None
        :type access_token: str, optional
        :param expiration_time: expiration time (TTL) of the access token,
                                defaults to 60
        :type expiration_time: int, optional
        :param read_only: key/token allows to change objects or only read,
                          defaults to False
        :type read_only: bool, optional
        :param global_key: key may only be used from whitelisted ip addresses,
                           defaults to False
        :type global_key: bool, optional
        :param label: textual identifier for the access token,
                      defaults to "__project__ __version__"
        :type label: str, optional
        :param authentication_url: TransIP authentication url,
                                   defaults to "https://api.transip.nl/v6/auth"
        :type authentication_url: str, optional
        :param root_endpoint: TransIP root of endpoints,
                              defaults to "https://api.transip.nl/v6"
        :type root_endpoint: str, optional
        :param connection_timeout: timeout for the network response, defaults to 30
        :type connection_timeout: int, optional
        :param retry: retry when the call fails due to zone
                      being saved or locked (409), defaults to 3
        :type retry: int, optional
        :param retry_delay: time in seconds to wait between retries,
                            defaults to 5
        :type retry_delay: float, optional
        """
        if login is not None and access_token is not None:
            raise ValueError(
                "Either login and private_key or access token must be used, not both."
            )

        self.attempts = retry + 1
        self.retry_delay = retry_delay
        self.root_endpoint = root_endpoint
        self.connection_timeout = connection_timeout
        if access_token is None:
            self._token = AccessToken(
                login=login,
                private_key=private_key_pem,
                private_key_file=private_key_pem_file,
                expiration_time=expiration_time,
                read_only=read_only,
                global_key=global_key,
                label=label,
                authentication_url=authentication_url,
                connection_timeout=connection_timeout,
            )
        else:
            self._token = access_token

    @property
    def headers(self) -> Dict:
        """Generate the default headers.

        Note the the reference to "self._token" will allways
        provide a valid (and renewed if needed) token

        :return: default headers, including the authentication token
        :rtype: Dict
        """
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._token}",
            "User-Agent": f"{__project__} {__version__}",
        }

    def execute_dns_entry(self, method: str, rest_path: str, dnsentry: dict):
        """Execute the requested action, with retry on 409.

        409: ~ "DNS Entries are currently being saved"
        409: ~ "is locked"


        :param method: get, post, patch, delete
        :type method: str
        :param zone_name: respective DNS zone
        :type zone_name: str
        :param dnsentry: DNS entry to manage
        :type dnsentry: dict
        :raises requests.exceptions.HTTPError: Raise an error
                                               if a 400 or 500 response is returned
        :return: the requests response
        :rtype: requests.models.Response
        """
        endpoint = f"{self.root_endpoint}{rest_path}"

        request = getattr(requests, method)
        response = None
        for attempt in range(1, self.attempts + 1):

            response = request(
                url=endpoint,
                json=dnsentry,
                headers=self.headers,
                timeout=self.connection_timeout,
            )
            if response.status_code != 409:
                response.raise_for_status()
                logger.debug(f"API request returned {response.status_code}")
                return response

            logger.debug(
                (
                    f"API request returned {response.status_code}: "
                    f"{response.text}, atttempt {attempt} of {self.attempts}"
                )
            )

            sleep(self.retry_delay)

        # raises requests.exceptions.HTTPError
        response.raise_for_status()

    def domains(self) -> list:
        """Get a listing of all available domains.

        [extended_summary]

        :return: List of available domains
        :rtype: list
        """
        return self.execute_dns_entry("get", "/domains", None)

    def get_dns_entry(self, dns_zone_name: str) -> Dict:
        """Get a listing of the respective domain."""
        response = self.execute_dns_entry(
            "get", rest_path=f"/domains/{dns_zone_name}/dns", dnsentry=None
        )
        return response

    def post_dns_entry(self, dns_record: DnsRecord):
        """Add a dnsEntry to the respective domain."""
        return self.execute_dns_entry(
            "post",
            rest_path=f"/domains/{dns_record.zone}/dns",
            dnsentry=dns_record.dnsentry,
        )

    def patch_dns_entry(self, dns_record: DnsRecord):
        """Adjust a record in the respective domain."""
        return self.execute_dns_entry(
            "patch",
            rest_path=f"/domains/{dns_record.zone}/dns",
            dnsentry=dns_record.dnsentry,
        )

    def delete_dns_entry(self, dns_record: DnsRecord):
        """Delete an entry in the respective domain."""
        return self.execute_dns_entry(
            "delete",
            rest_path=f"/domains/{dns_record.zone}/dns",
            dnsentry=dns_record.dnsentry,
        )
