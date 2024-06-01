def repl(elt, req, eq):
    r = req.copy()
    q = r[elt] // eq['qte']
    if r[elt] % eq['qte'] != 0: q += 1

    for i in eq['inp']:
        if i not in r: r[i] = 0
        r[i] += q * eq['inp'][i]
    r[elt] -= q * eq['qte']

    return r

def optimize(need, dest, eqs):
    req = need.copy()
    while any(req[i] > 0 and i != dest for i in req):
        elt = [i for i in req if req[i] > 0 and i != dest][0]
        did = False
        others = []
        req = repl(elt, req, eqs[elt])

    return req[dest]

with open('data/input14.txt') as f:
    l_eq = f.readlines()

eqs = {}    

for eq in l_eq:
    left, right = eq[:-1].split(' => ')
    out = right.split()
    eqs[out[1]] = {'qte': int(out[0])}
    eqs[out[1]]['inp'] = {}
    for elt in left.split(', '):
        item = elt.split()
        eqs[out[1]]['inp'][item[1]] = int(item[0])

req = {'FUEL': 1}
q_ore = optimize(req, 'ORE', eqs)
print(q_ore)

goal = 10**12
low = goal // q_ore
high = low
i = 1

while optimize({'FUEL': high}, 'ORE', eqs) < goal:
    high = low + 10**i
    i += 1

while True:
    m = (high + low) // 2
    if optimize({'FUEL': m}, 'ORE', eqs) > goal:
        high = m
    else:
        low = m
    if high - low < 2:
        print(low)
        break