from __future__ import annotations

from typing import List, Tuple, Optional

from network_simulator.BloodType import BloodType
from network_simulator.compatibility_markers import OrganType

path_structure = Optional[List[Optional[int]]]
shortest_path_structure = Tuple[path_structure, float]


class Organ:
    """
    A class representing a given organ which is available for transplant.

    Each organ has a name, a unique ID, lifetime (a maximum out of body duration),
    type matching, and a location.
    """

    organ_count = 0

    def __init__(self, organ_type: OrganType, blood_type: BloodType,
                 location: int, organ_list: 'OrganList' = None) -> None:  # type: ignore
        Organ.organ_count = Organ.organ_count + 1

        self.organ_id: int = Organ.organ_count
        self.organ_type: OrganType = organ_type
        self.blood_type: BloodType = blood_type
        self.viability: float = Organ.get_viability(self.organ_type)
        self.origin_location: int = location
        self.current_location: int = location
        self.path: path_structure = [location]

        if organ_list:
            organ_list.add_organ(self)

    def move_organ(self, new_location: int, cost: float,
                   shortest_path: shortest_path_structure) -> None:
        """
        This function allows an organ's attributes to be altered to represent it's
        transportation across the network. This is intended to be used with
        Dijkstra.shortest_path (this will be the source of the cost parameter)

        :param int new_location: node id representing the destination location
        :param cost: weight/cost associated with then most efficient path
        :param list shortest_path: transit path taken when moving organ
        """
        if self.viability < cost:
            print('ERROR: organ no longer viable!')
            return
        path, weight = shortest_path
        self.path = path
        self.current_location = new_location
        self.viability -= cost

    @staticmethod
    def get_viability(organ_type: OrganType) -> float:
        """
        Gets viability rating for each organ individually

        Viability is represented by hours an organ can be out of body * 10

        :param int organ_type: constant corresponding to an organ type
        :return: int viability rating (used in __init__())
        """
        viability = {
            OrganType.Heart.value:      60,
            OrganType.Kidney.value:     300,
            OrganType.Liver.value:      120,
            OrganType.Lungs.value:      60,
            OrganType.Pancreas.value:   120,
            OrganType.Intestines.value: 80}

        return viability[organ_type.value]

    def __str__(self) -> str:
        """
        Builds an easily readable string representing an organ

        :return: str
        """
        return f'Organ:\n' \
            f'\tOrgan ID: {"{:05d}".format(self.organ_id)}\n' \
            f'\tOrgan type: {OrganType(self.organ_type).name}\n' \
            f'\tBlood type: {self.blood_type}\n' \
            f'\tViability: {self.viability}\n' \
            f'\tOrigin location: {self.origin_location}\n' \
            f'\tCurrent location: {self.current_location}\n' \
            f'\tTransit path: {self.path}\n'
