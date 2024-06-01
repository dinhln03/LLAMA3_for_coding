# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from pathlib import Path

import pandas as pd
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from .processing import process_daily


class FileWatcher(FileSystemEventHandler):
    def __init__(self, path, sheets):
        super().__init__()
        self._path = Path(path)
        self._sheets = sheets
        self._observer = Observer()
        self._observer.schedule(self, str(self._path.parent))

    def _update_sheets(self):
        self._sheets.clear()
        sheets = pd.read_excel(self._path, None)
        for key, val in sheets.items():
            if key[:2] == '20':
                sheets[key] = process_daily(val)
        self._sheets.update(sheets)

    def on_created(self, event):
        if Path(event.src_path) == self._path:
            self._update_sheets()

    def run(self):
        self._update_sheets()
        self._observer.start()
        print('running...')

    def stop(self):
        self._observer.stop()
        self._observer.join()
        print('stopped')
