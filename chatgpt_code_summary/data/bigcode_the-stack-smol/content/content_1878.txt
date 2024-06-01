#!/usr/bin/python

script = r"""
MD Dir1
MD Dir1\Dir2
CD Dir1\Dir2
MF file2.dat
MD Dir3
CD Dir3
MF file3.dat
MD Dir4
CD Dir4
MF file4.dat
MD Dir5
CD Dir5
MF file5.dat
CD C:
DELTREE Dir1
MD Dir2
CD Dir2
MF a.txt
MF b.txt
CD C:
MD Dir3
COPY Dir2 Dir3
"""


expected = r"""
C:
|_DIR2
|   |_a.txt
|   |_b.txt
|
|_DIR3
    |_DIR2
        |_a.txt
        |_b.txt
"""

import test

test.run(script, expected)

