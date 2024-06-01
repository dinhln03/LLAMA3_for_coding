# -*- coding: utf-8 -*-

"""
/***************************************************************************
 QGIS plugin 'Graphium'
/***************************************************************************
 *
 * Copyright 2020 Simon Gröchenig @ Salzburg Research
 * eMail     graphium@salzburgresearch.at
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 ***************************************************************************/
"""

from enum import Enum


class OsmHighwayTypes(Enum):
    MOTORWAY = 'motorway'
    MOTORWAY_LINK = 'motorway_link'
    TRUNK = 'trunk'
    TRUNK_LINK = 'trunk_link'
    PRIMARY = 'primary'
    PRIMARY_LINK = 'primary_link'
    SECONDARY = 'secondary'
    SECONDARY_LINK = 'secondary_link'
    TERTIARY = 'tertiary'
    TERTIARY_LINK = 'tertiary_link'
    UNCLASSIFIED = 'unclassified'
    RESIDENTIAL = 'residential'
    LIVING_STREET = 'living_street'
    SERVICE = 'service'
    PEDESTRIAN = 'pedestrian'
    TRACK = 'track'
    BUS_GUIDEWAY = 'bus_guideway'
    FOOTWAY = 'footway'
    BRIDLEWAY = 'bridleway'
    STEPS = 'steps'
    CORRIDOR = 'dorridor'
    PATH = 'path'
    SIDEWALK = 'sidewalk'
    CYCLEWAY = 'cycleway'
