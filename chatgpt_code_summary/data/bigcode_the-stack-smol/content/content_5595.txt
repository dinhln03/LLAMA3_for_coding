#!/usr/bin/env python3

import sys

def gen_freq(freq: float, duration: int):
    tmp = 0.0
    pri = chr(32)
    sec = chr(126)
    for i in range(duration):
        if (tmp >= freq):
            tmp -= freq
            pri, sec = sec, pri
        tmp += 1.0
        print(pri, end='')
    sys.stdout.flush()

if __name__ == '__main__':
    assert(len(sys.argv) == 3)
    gen_freq(float(sys.argv[1]), int(sys.argv[2]))
