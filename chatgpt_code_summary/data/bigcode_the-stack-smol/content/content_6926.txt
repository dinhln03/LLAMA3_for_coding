__version__ = "0.1.0"

import mmap
import os

# from   .ext import load_file, parse

#-------------------------------------------------------------------------------

def parse_file(path, **kw_args):
    fd = os.open(path, os.O_RDONLY)
    try:
        map = mmap.mmap(fd, 0, prot=mmap.PROT_READ)
        return parse(map, **kw_args)
    finally:
        os.close(fd)


