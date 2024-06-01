# -*- coding: utf-8 -*-

"""Access to FAIRsharing via its API.

.. seealso:: https://beta.fairsharing.org/API_doc
"""

from typing import Any, Iterable, Mapping, MutableMapping, Optional

import pystow
import requests
import yaml
from tqdm import tqdm

__all__ = [
    "ensure_fairsharing",
    "load_fairsharing",
    "FairsharingClient",
]

PATH = pystow.join("bio", "fairsharing", name="fairsharing.yaml")


def load_fairsharing(force_download: bool = False, use_tqdm: bool = True, **kwargs):
    """Get the FAIRsharing registry."""
    path = ensure_fairsharing(force_download=force_download, use_tqdm=use_tqdm, **kwargs)
    with path.open() as file:
        return yaml.safe_load(file)


def ensure_fairsharing(force_download: bool = False, use_tqdm: bool = True, **kwargs):
    """Get the FAIRsharing registry."""
    if PATH.exists() and not force_download:
        return PATH

    client = FairsharingClient(**kwargs)
    # As of 2021-12-13, there are a bit less than 4k records that take about 3 minutes to download
    rv = {
        row["prefix"]: row
        for row in tqdm(
            client.iter_records(),
            unit_scale=True,
            unit="record",
            desc="Downloading FAIRsharing",
            disable=not use_tqdm,
        )
    }
    with PATH.open("w") as file:
        yaml.safe_dump(rv, file, allow_unicode=True, sort_keys=True)
    return PATH


# These fields are the same in each record
REDUNDANT_FIELDS = {
    "fairsharing-licence",
}


class FairsharingClient:
    """A client for programmatic access to the FAIRsharing private API."""

    def __init__(
        self,
        login: Optional[str] = None,
        password: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """Instantiate the client and get an appropriate JWT token.

        :param login: FAIRsharing username
        :param password: Corresponding FAIRsharing password
        :param base_url: The base URL
        """
        self.base_url = base_url or "https://api.fairsharing.org"
        self.signin_url = f"{self.base_url}/users/sign_in"
        self.records_url = f"{self.base_url}/fairsharing_records"
        self.username = pystow.get_config(
            "fairsharing", "login", passthrough=login, raise_on_missing=True
        )
        self.password = pystow.get_config(
            "fairsharing", "password", passthrough=password, raise_on_missing=True
        )
        self.jwt = self.get_jwt()
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.jwt}",
            }
        )

    def get_jwt(self) -> str:
        """Get the JWT."""
        payload = {
            "user": {
                "login": self.username,
                "password": self.password,
            },
        }
        res = requests.post(self.signin_url, json=payload).json()
        return res["jwt"]

    def iter_records(self) -> Iterable[Mapping[str, Any]]:
        """Iterate over all FAIRsharing records."""
        yield from self._iter_records_helper(self.records_url)

    def _preprocess_record(
        self, record: MutableMapping[str, Any]
    ) -> Optional[MutableMapping[str, Any]]:
        if "type" in record:
            del record["type"]
        record = {"id": record["id"], **record["attributes"]}

        doi = record.get("doi")
        if doi is None:
            # Records without a DOI can't be resolved
            url = record["url"]
            if not url.startswith("https://fairsharing.org/fairsharing_records/"):
                tqdm.write(f"{record['id']} has no DOI: {record['url']}")
            return None
        elif doi.startswith("10.25504/"):
            record["prefix"] = record.pop("doi")[len("10.25504/") :]
        else:
            tqdm.write(f"DOI has unexpected prefix: {record['doi']}")

        record["description"] = _removeprefix(
            record.get("description"), "This FAIRsharing record describes: "
        )
        record["name"] = _removeprefix(record.get("name"), "FAIRsharing record for: ")
        for key in REDUNDANT_FIELDS:
            if key in record:
                del record[key]
        return record

    def _iter_records_helper(self, url: str) -> Iterable[Mapping[str, Any]]:
        res = self.session.get(url).json()
        for record in res["data"]:
            yv = self._preprocess_record(record)
            if yv:
                yield yv
        next_url = res["links"].get("next")
        if next_url:
            yield from self._iter_records_helper(next_url)


def _removeprefix(s: Optional[str], prefix) -> Optional[str]:
    if s is None:
        return None
    if s.startswith(prefix):
        return s[len(prefix) :]
    return s


if __name__ == "__main__":
    ensure_fairsharing(force_download=True)
