from dataclasses import dataclass

from beebole.interfaces.entities import IdEntity, CountEntity


@dataclass
class Group(IdEntity):
    name: str
    groups: CountEntity


@dataclass
class ParentedGroup(IdEntity):
    name: str
    parent: IdEntity
