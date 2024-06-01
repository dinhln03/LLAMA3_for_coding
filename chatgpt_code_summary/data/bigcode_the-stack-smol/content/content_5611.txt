"""
    Builder for web assembly
"""

import subprocess
import sys
from SCons.Script import AlwaysBuild, Default, DefaultEnvironment

try:
    subprocess.check_output(["em++", "--version"])
except FileNotFoundError:
    print(
        "Could not find emscripten. Maybe install it? (e.g. `brew install emscripten` on macOS. See also: https://emscripten.org/docs/getting_started/downloads.html)",
        file=sys.stderr,
    )
    exit(1)

env = DefaultEnvironment()

env.Append(
    LINKFLAGS=["--bind"],
)

env.Replace(
    CXX="em++",
    CC="emcc",
    AR="emar",
    RANLIB="emranlib",
    PROGSUFFIX=".html"
)

#
# Target: Build wasm
#
target_bin = env.BuildProgram()

#
# Default targets
#
Default([target_bin])
