"""
utils.py
"""
import pathlib
import tempfile
import shutil

import curio.io as io


async def atomic_write(p, data):
    p = pathlib.Path(p)
    with tempfile.NamedTemporaryFile(dir=p.parent, delete=False) as f:
        af = io.FileStream(f)
        res = await af.write(data)
    shutil.move(f.name, p)
    return res
