#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
.. py:currentmodule:: trim.montecarlo.source

.. moduleauthor:: Hendrix Demers <hendrix.demers@mail.mcgill.ca>


"""

# Copyright 2019 Hendrix Demers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Standard library modules.

# Third party modules.

# Local modules.

# Project modules.
from trim.montecarlo.math import Point

# Globals and constants variables.
GROUP_SOURCE = "source"
GROUP_POSITIONS = "position (nm)"
GROUP_DIRECTION = "direction"
ATTRIBUTE_KINETIC_ENERGY = "kinetic energy (keV)"
ATTRIBUTE_MASS = "mass (amu)"
ATTRIBUTE_ATOMIC_NUMBER = "atomic number"


class Source:
    def __init__(self):
        # Default to Ar at 6 keV
        self.position_nm = Point(0.0, 0.0, 0.0)
        self.direction = Point(0.0, 0.0, -1.0)
        self.kinetic_energy_keV = 6.0
        self.mass_amu = 39.962
        self.atomic_number = 18

    def write(self, parent):
        group = parent.require_group(GROUP_SOURCE)

        position_group = group.require_group(GROUP_POSITIONS)
        self.position_nm.write(position_group)

        direction_group = group.require_group(GROUP_DIRECTION)
        self.direction.write(direction_group)

        group.attrs[ATTRIBUTE_KINETIC_ENERGY] = self.kinetic_energy_keV
        group.attrs[ATTRIBUTE_MASS] = self.mass_amu
        group.attrs[ATTRIBUTE_ATOMIC_NUMBER] = self.atomic_number

    def read(self, parent):
        group = parent.require_group(GROUP_SOURCE)

        position_group = group.require_group(GROUP_POSITIONS)
        self.position_nm.read(position_group)

        direction_group = group.require_group(GROUP_DIRECTION)
        self.direction.read(direction_group)

        self.kinetic_energy_keV = group.attrs[ATTRIBUTE_KINETIC_ENERGY]
        self.mass_amu = group.attrs[ATTRIBUTE_MASS]
        self.atomic_number = group.attrs[ATTRIBUTE_ATOMIC_NUMBER]
