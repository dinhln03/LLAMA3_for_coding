import os
import platform
from datetime import datetime


# TODO test counter
# def test_count():
#    return 0
# f"Count: {test_count()}\n"\

def serve_info():
    return f"Stats\n" \
           f"UTC: {datetime.utcnow().isoformat()}\n" \
           f"\nMachine\n" \
           f"Architecture: {platform.machine()}\n" \
           f"Name: {platform.node()}\n" \
           f"Platform: {platform.platform()}\n" \
           f"CPU Model: {platform.processor()}\n" \
           f"CPU Count: {os.cpu_count()}\n" \
           f"Release: {platform.release()}\n" \
           f"System: {platform.system()}\n" \
           f"Version: {platform.version()}\n" \
           f"\nPython\n" \
           f"Branch: {platform.python_branch()}\n" \
           f"Build: {platform.python_build()}\n" \
           f"Compiler: {platform.python_compiler()}\n" \
           f"Implementation: {platform.python_implementation()}\n" \
           f"Revision: {platform.python_revision()}\n"
