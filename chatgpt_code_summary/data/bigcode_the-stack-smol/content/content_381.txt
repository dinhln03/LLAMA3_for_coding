#!/usr/bin/python
import hashlib
import sys

v = sys.argv[1]
index = 0
pw = ''
i = 0
while True:
    suffix = str(i)
    h = hashlib.md5(v+suffix).hexdigest()
    if h.startswith("00000"):

        pw += h[5]
        print(v+suffix,h,pw)
        if len(pw) == 8:
            break
    i += 1

print(pw)

