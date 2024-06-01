import sys
import os
from cx_Freeze import setup, Executable

# because of how namespace packages work, cx-freeze isn't finding zope.interface
# the following will import it, find the path of zope, and add a new empty
# file name _init__.py at the /site-packages/python2.7/zope path.
# this was found here:
# https://bitbucket.org/anthony_tuininga/cx_freeze/issues/47/cannot-import-zopeinterface

import zope
path = zope.__path__[0]
open(os.path.join(path, '__init__.py'), 'wb')  # create __init__.py file

# Dependencies are automatically detected, but it might need fine tuning.
build_exe_options = {"packages": ["zope.interface"]}

base = None
if sys.platform == "win32":
    base = "Win32GUI"

setup(  name = "tw-cs",
        version = "0.1",
        description = "My twisted application!",
        options = {"build_exe": build_exe_options},
        executables = [Executable("echoserv.py", base=base)])
