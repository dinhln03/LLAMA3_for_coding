from dataclasses import dataclass, field
from typing import List
from bindings.wfs.range import Range

__NAMESPACE__ = "http://www.opengis.net/ows/1.1"


@dataclass
class AllowedValues:
    """List of all the valid values and/or ranges of values for this quantity.

    For numeric quantities, signed values should be ordered from
    negative infinity to positive infinity.
    """

    class Meta:
        namespace = "http://www.opengis.net/ows/1.1"

    value: List[str] = field(
        default_factory=list,
        metadata={
            "name": "Value",
            "type": "Element",
        },
    )
    range: List[Range] = field(
        default_factory=list,
        metadata={
            "name": "Range",
            "type": "Element",
        },
    )
