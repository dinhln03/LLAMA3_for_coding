# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2015-2016 European Synchrotron Radiation Facility
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ###########################################################################*/

from __future__ import absolute_import

__authors__ = ["D. Naudet"]
__license__ = "MIT"
__date__ = "15/09/2016"


from ...io.XsocsH5 import ScanPositions

from .ProjectItem import ProjectItem
from .ProjectDef import ItemClassDef


@ItemClassDef('ScanPositionsItem')
class ScanPositionsItem(ProjectItem):
    def _createItem(self):
        with self.xsocsH5 as h5f:
            entries = h5f.entries()
            entry = entries[0]
            scan_positions = h5f.scan_positions(entry)
            pathTpl = self.path + '/' + '{0}'
            with self:
                itemPath = pathTpl.format('pos_0')
                self._set_array_data(itemPath, scan_positions.pos_0)
                itemPath = pathTpl.format('pos_1')
                self._set_array_data(itemPath, scan_positions.pos_1)
                itemPath = pathTpl.format('motor_0')
                self._set_scalar_data(itemPath, scan_positions.motor_0)
                itemPath = pathTpl.format('motor_1')
                self._set_scalar_data(itemPath, scan_positions.motor_1)
                itemPath = pathTpl.format('n_0')
                self._set_scalar_data(itemPath, scan_positions.shape[0])
                itemPath = pathTpl.format('n_1')
                self._set_scalar_data(itemPath, scan_positions.shape[1])

    def positions(self):
        pathTpl = self.path + '/' + '{0}'
        with self:
            itemPath = pathTpl.format('pos_0')
            pos_0 = self._get_array_data(itemPath)
            itemPath = pathTpl.format('pos_1')
            pos_1 = self._get_array_data(itemPath)
            itemPath = pathTpl.format('motor_0')
            motor_0 = self._get_scalar_data(itemPath)
            itemPath = pathTpl.format('motor_1')
            motor_1 = self._get_scalar_data(itemPath)
            itemPath = pathTpl.format('n_0')
            n_0 = self._get_scalar_data(itemPath)
            itemPath = pathTpl.format('n_1')
            n_1 = self._get_scalar_data(itemPath)
        return ScanPositions(motor_0=motor_0,
                             pos_0=pos_0,
                             motor_1=motor_1,
                             pos_1=pos_1,
                             shape=(n_0, n_1))
