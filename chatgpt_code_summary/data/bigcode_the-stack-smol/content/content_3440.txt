# /usr/bin/env python3

"""Benchmark of handling PDB files comparing multiple libraries."""

import argparse
import glob
import os
import re
import subprocess
import sys
from pathlib import Path


def gather_libs(selected_libs):
    libs = []
    for path in sorted(glob.iglob("bench/*")):
        lib = os.path.basename(path)
        if not os.path.isdir(path) or (selected_libs and lib not in selected_libs):
            continue
        libs.append(lib)
    return libs


def gather_tests(libs, selected_tests):
    tests = []
    for lib in libs:
        for filepath in sorted(glob.iglob(os.path.join("bench", lib, "*"))):
            test, _ = os.path.splitext(os.path.basename(filepath))
            if test in tests or (selected_tests and test not in selected_tests):
                continue
            tests.append(test)
    return tests


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-t", "--tests", help="Test names to run.")
    parser.add_argument("-l", "--libraries", help="Library names to test.")

    opts = parser.parse_args()
    if opts.tests:
        opts.tests = opts.tests.split(",")
    if opts.libraries:
        opts.libraries = opts.libraries.split(",")

    return vars(opts)


def run_test(filepath, pdbfile, repeats=10):
    *_, dirname, filename = Path(filepath).parts
    basename, _ = os.path.splitext(filename)
    pdbid, _ = os.path.splitext(os.path.basename(pdbfile))
    print(format(f"{dirname}/{basename}/{pdbid}", "<40"), end="", flush=True)

    if "schrodinger" in filepath:
        cmd = [
            os.path.join(os.environ["SCHRODINGER"], "run"),
            filepath,
            pdbfile,
            str(repeats),
        ]
    elif filepath.endswith(".py"):
        cmd = ["python3", filepath, pdbfile, str(repeats)]
    elif filepath.endswith(".cr"):
        cmd = ["crystal", "run", "--release", filepath, "--", pdbfile, str(repeats)]
    elif filepath.endswith(".tcl"):
        cmd = [
            "vmd",
            "-dispdev",
            "none",
            "-e",
            filepath,
            "-args",
            pdbfile,
            str(repeats),
        ]

    try:
        output = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
        output = output.decode(sys.stdout.encoding).strip()
        try:
            elapsed = float(output)
        except ValueError:
            elapsed = float(re.findall(r"elapsed *= *([\d\.e\-]+)", output)[0])
        print(format(elapsed, ".6f"))
    except subprocess.CalledProcessError:
        print("failed")


opts = parse_args(sys.argv[1:])
libs = gather_libs(opts["libraries"])
tests = gather_tests(libs, opts["tests"])
pdbs = list(map(os.path.abspath, glob.glob("data/*.pdb")))

for test in tests:
    for pdbfile in pdbs if test.startswith("parse") else ["data/1ake.pdb"]:
        for lib in libs:
            paths = glob.glob(f"bench/{lib}/{test}.*")
            if not paths:
                continue

            run_test(paths[0], pdbfile, repeats=10 if "1htq" not in pdbfile else 3)

        print("")
