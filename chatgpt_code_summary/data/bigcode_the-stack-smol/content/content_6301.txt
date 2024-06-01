"""
This module contains the cli functions.
Split them out into separate files if required.
"""

import sys
import os
import subprocess
import pickle
from cheapskate_bal import balance as bal

__collector__ = {'exe': "collect_3008", 'samp_rate': 2000}


def csbal_process():
    """
    This method is run when the `csbal` script is called.
    can be used to check a single file (check balance state after adjusting)
    args are file stem, freq (Hz [rpm/60] float), samp_rate (data collector)
    """
    args = sys.argv[1:]

    stem = args[0]
    freq = float(args[1])
    samp_rate = float(args[2])

    df = bal.read_data_files(stem, freq, samp_rate)
    bal.graph_data(df)
    bal.process_data(df, freq, samp_rate, True)


def grab_data(tests, stem):
    for t in tests:
        msg, tag = t
        print("\n\n==================================")
        print(msg)
        print("start DUT now")
        input("Press Enter to start data capture...")
        cp = subprocess.run(["taskset", "-c", "3", "nice", "-20", __collector__['exe'], stem+tag],
                            capture_output=True, universal_newlines=True)
        summary = cp.stdout.splitlines()[-5:]
        print(*summary,sep='\n')


def batch_process(tests, stem, freq):
    results = []
    for t in tests:
        tag = t[1]
        sr = __collector__['samp_rate']
        df = bal.read_data_files(stem+tag, freq, sr)
        results.append(bal.process_data(df, freq, sr))
    return results


def csbal_single():
    """
    This method performs the whole process for a single plane balance
    Four data files are captured, and the results are emitted
    args are file stem, freq(Hz), shift angle of test mass (deg), test mass """

    args = sys.argv[1:]
    if len(args) < 4:
        print("args are stem, freq, shift_ang, test_mass")
    stem = args[0]
    freq = float(args[1])
    shift_ang = float(args[2])
    tmass = float(args[3])

    offset_1_ang = 360
    offset_2_ang = 360  # these should not both be 0, as there is a div by their sum

    if len(args) > 5:
        offset_1_ang = float(args[4])
        offset_2_ang = float(args[5])

    # make sure the stem looks like a directory
    if stem[-1] != os.path.sep:
        stem = stem + os.path.sep

    tests = [('T0: initial unbalanced state', 't0'),
             ('T1: test mass at 0 deg ref', 't1'),
             ('T2: test mass at positive angle', 't2'),
             ('T3: test mass at negative angle', 't3'), ]

    grab_data(tests, stem)
    print("Processing captured data...")

    results = batch_process(tests, stem, freq)

    print("Balace Results:")
    bal.single_balance(results, tmass, shift_ang, offset_1_ang, offset_2_ang)


def csbal_dual_init():
    """
    THis method performs the whole process for a dual plane balance
    Three files are captured and the results are emitted
    args are file stem, freq(Hz), shift angle of test mass (deg), test mass """

    args = sys.argv[1:]
    if len(args) < 4:
        print("args are stem, freq, shift_ang, test_mass")
    stem = args[0]
    freq = float(args[1])
    shift_ang = float(args[2])
    tmass = float(args[3])

    # make sure the stem looks like a directory
    if stem[-1] != os.path.sep:
        stem = stem + os.path.sep

    tests = [('T0: initial unbalanced state', 't0'),
             ('TA: test mass on bearing 1 at shift angle', 'ta'),
             ('TB: test mass on bearing 2 at shift angle', 'tb')]

    grab_data(tests, stem)

    print("Processing captured data...")
    results = batch_process(tests, stem, freq)

    print("Dual Plane Balance Results")
    influence, correction = bal.dual_compute_influence(results, tmass, shift_ang)

    # write the influence params to a file
    inf_file = stem+"influence"
    with open(inf_file, 'wb') as filehandle:
        pickle.dump(influence, filehandle)


def csbal_dual_iter():
    """
    This method performs an iteration of dual plane  balance, once the
    influence params are known. One file is captured and the results
    are emitted
    args are file stem, tag, freq
    """

    args = sys.argv[1:]
    if len(args) < 3:
        print("args are: filestem, tag, freq")
    stem = args[0]
    tag = args[1]
    freq = float(args[2])

    # make sure the stem looks like a directory
    if stem[-1] != os.path.sep:
        stem = stem + os.path.sep

    # get the influence from file
    influence = []
    inf_file = stem+"influence"
    with open(inf_file, 'rb') as filehandle:
        influence = pickle.load(filehandle)

    tests = [('T(curr): initial unbalanced state', 't'+tag)]

    grab_data(tests, stem)

    print("Processing captured data...")
    results = batch_process(tests, stem, freq)

    print("Dual Plane Balance Results")
    correction = bal.dual_compute_weights(results, influence)


