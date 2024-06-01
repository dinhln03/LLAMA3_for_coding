"""
Load volumes into vpv from a toml config file. Just load volumes and no overlays

Examples
--------

Example toml file

orientation = 'sagittal'


[top]
specimens = [
'path1.nrrd',
'path2.nrrd',
'path3.nrrd']

[bottom]
specimens = [
'path1.nrrd',
'path2.nrrd',
'path3.nrrd']

"""
import sys
from pathlib import Path
from itertools import chain

import toml
from PyQt5 import QtGui

from vpv.vpv import Vpv
from vpv.common import Layers

from typing import Dict

def load(config: Dict):



    top_vols = config['top']['specimens']


    bottom = config['bottom']['specimens']
    if bottom:
        bottom_vols =  config['bottom']['specimens']
    else: # We allow only top vier visible
        bottom_specs = []
        bottom_vols = []
        bottom_labels = []

    app = QtGui.QApplication([])
    ex = Vpv()

    p2s = lambda x: [str(z) for z in x]

    all_vols = top_vols + bottom_vols
    ex.load_volumes(chain(p2s(top_vols), p2s(bottom_vols)), 'vol')


    # Set the top row of views
    for i in range(3):
        try:
            vol_id = Path(top_vols[i]).stem
            ex.views[i].layers[Layers.vol1].set_volume(vol_id)
        except IndexError:
            continue

    if bottom:
        # Set the top row of views
        for i in range(3):
            try:
                vol_id = Path(bottom_vols[i]).stem
                ex.views[i + 3].layers[Layers.vol1].set_volume(vol_id)
            except IndexError:
                continue

    print('Finished loading')

    # Show two rows
    ex.data_manager.show2Rows(True if bottom else False)

    # Set orientation
    ex.data_manager.on_orientation(config['orientation'])

    sys.exit(app.exec_())


if __name__ == '__main__':
    file_ = sys.argv[1]
    config = toml.load(file_)
    load(config)