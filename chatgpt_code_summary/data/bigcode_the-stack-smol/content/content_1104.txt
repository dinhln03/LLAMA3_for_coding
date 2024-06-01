import twl

wolf = twl.Wolf()


def split(wolf, end=7):
    return [wolf.len(n)() for n in range(2, end+1)]


def spell(ltrs, wild=0):
    return split(wolf.wild(ltrs, wild), len(ltrs)+wild)


def _munge(func, fix, ltrs, wild=0):
    return split(func(fix).wild(fix+ltrs, wild), len(fix+ltrs)+wild)


def starts(fix, ltrs, wild=0):
    return _munge(wolf.starts, fix, ltrs, wild)


def ends(fix, ltrs, wild=0):
    return _munge(wolf.ends, fix, ltrs, wild)


def contains(fix, ltrs, wild=0):
    return _munge(wolf.contains, fix, ltrs, wild)


if __name__ == "__main__":
    # print(wolf.len(2).words)
    # print(wolf.wild('aa')())
    print(contains('a', 'ciodtji'))
