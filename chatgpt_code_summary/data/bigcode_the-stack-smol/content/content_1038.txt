from typing import List, Literal, Union, Callable, Tuple
from dataclasses import dataclass, replace

from .config_types import (
    TWInterface,
    TWSettingStorage, TWSettingBool, TWSetting,

    WrapType, Fields, AnkiModel, LabelText, WhichField, Tags, Falsifiable,
)

ScriptKeys = Literal[
    'enabled',
    'name',
    'version',
    'description',
    'conditions',
    'code',
]

def __list_to_tw_bool(prototype, vals: List[ScriptKeys]):
    return replace(
        prototype,
        **dict([(key, True) for key in vals])
    )

def make_interface(
    # name for the type of the interface
    tag: str,
    prototype:  WrapType,
    getter: Callable[[str, TWSettingStorage], TWSetting],
    # result is used for storing,
    setter: Callable[[str, TWSetting], Union[bool, TWSetting]],
    wrapper: Callable[[str, TWSettingStorage, AnkiModel, Fields, WhichField, slice, Tags], Tuple[Fields, Tags]],
    label: Falsifiable(Callable[[str, TWSettingStorage], LabelText]),
    reset: Falsifiable(Callable[[str, TWSettingStorage], TWSetting]),
    deletable: Falsifiable(Callable[[str, TWSettingStorage], bool]),

    # list of values that are readonly,,
    readonly: TWSettingBool,
    # list of values or stored in `storage` field,
    store: TWSettingBool,
) -> TWInterface:
    return TWInterface(
        tag,
        prototype,
        getter,
        setter,
        wrapper,
        label,
        reset,
        deletable,
        readonly,
        store,
    )
