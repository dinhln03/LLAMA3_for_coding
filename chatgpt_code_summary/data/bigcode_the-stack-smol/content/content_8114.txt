"""
Data Structures shared by both the detectors and the server
"""

import datetime
import sys
import traceback
from typing import *
from dataclasses import dataclass, field
from dataclasses_jsonschema import JsonSchemaMixin


@dataclass
class ConfigMessage(JsonSchemaMixin):
    cat_identifiers: Dict[str, str] # service_id -> cat_name
    sampling_period: int = 15 # How often to sample, in seconds
    api_uri: str = r"http://tesla:5058/kitbit/api"


@dataclass
class ScanObservationMessage(JsonSchemaMixin):
    detector_uuid: str
    cat_rssi: Dict[str, float]


@dataclass
class ErrorMessage(JsonSchemaMixin):
    traceback: str
    exception: str
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)

    @staticmethod
    def from_last_exception():
        return ErrorMessage(
            traceback=traceback.format_exc(),
            exception=str(sys.exc_info()[0])
        )

