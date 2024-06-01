import os

import PyQt5.QtCore as qc

DATA_DIR = 'MiptCisDocs'
WRITABLE_LOCATION = qc.QStandardPaths.writableLocation(
    qc.QStandardPaths.StandardLocation.AppDataLocation
)


def get_data_dir() -> str:
    data_dir = os.path.abspath(os.path.join(WRITABLE_LOCATION, DATA_DIR))
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    return data_dir
