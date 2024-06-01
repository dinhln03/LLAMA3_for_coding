#
# linter.py
# Linter for SublimeLinter3, a code checking framework for Sublime Text 3
#
# Written by David O'Brien
# Copyright (c) 2017 David O'Brien
#
# License: MIT
#

"""Exports the Vcom plugin class."""

from SublimeLinter.lint import Linter


class Vcom(Linter):
    """Provides an interface to vcom (Mentor Modelsim)."""

    syntax = ('vhdl')
    cmd = 'vcom -2008 -work work @'
    tempfile_suffix = 'vhd'

    # SAMPLE ERRORS
    # ** Error: (vcom-13069) .\fname.v(9): near "reg": syntax error, unexpected reg, expecting ';' or ','.
    # ** Error: (vcom-13069) .\fname.v(9): Unknown identifier "var": syntax error, unexpected reg, expecting ';' or ','.
    # ** Error (suppressible): .\fname.sv(46): (vlog-2388) 'var' already declared in this scope (mname).
    # ** Error: (vlog-13069) .\fname.sv(45): near "==": syntax error, unexpected ==, expecting ';' or ','.

    regex = (
        r'^\*\* (((?P<error>Error)|(?P<warning>Warning))'      # Error
        r'( \(suppressible\))?: )'                             # Maybe suppressible
        r'(\([a-z]+-[0-9]+\) )?'                               # Error code - sometimes before
        r'([^\(]*\((?P<line>[0-9]+)\): )'                      # File and line
        r'(\([a-z]+-[0-9]+\) )?'                               # Error code - sometimes after
        r'(?P<message>'                                        # Start of message
        r'(((near|Unknown identifier|Undefined variable):? )?' # Near/Unknown/Unidentified
        r'["\'](?P<near>[\w=:;\.]+)["\']'                      # Identifier
        r'[ :.]*)?'                                            # Near terminator
        r'.*)'                                                 # End of message
    )

    def split_match(self, match):
        """Override this method to prefix the error message with the lint binary name."""
        match, line, col, error, warning, message, near = super().split_match(match)
        if match:
            message = '[vcom] ' + message
        return match, line, col, error, warning, message, near
