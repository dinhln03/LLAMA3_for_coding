data = "cqjxjnds"

import string
import re

lc = string.ascii_lowercase
next = dict(zip(lc[:-1], lc[1:]))

three_seq = ["".join(z) for z in zip(lc[:-2], lc[1:-1], lc[2:])]

def check(pw):
    if "i" in pw or "o" in pw or "l" in pw:
        return False
    three_match = False
    for seq in three_seq:
        if seq in pw:
            three_match = True
    if not three_match:
        return False
    doubles = set(re.findall(r'(.)\1', pw))
    if  len(doubles) < 2:
        return False
    return True

def inc(pw):
    pw = list(pw)
    i = -1
    while pw[i] == 'z':
        pw[i] = 'a'
        i -= 1
    pw[i] = next[pw[i]]
    return "".join(pw)

# TEST
print(check("hijklmmn"))
print(check("abbceffg"))
print(check("abbcegjk"))
print(check("abcdffaa"))
print(check("ghjaabcc"))

# PART 1
pw = data
while not check(pw):
    pw = inc(pw)
print(pw)

# PART 2
pw = inc(pw)
while not check(pw):
    pw = inc(pw)
print(pw)
