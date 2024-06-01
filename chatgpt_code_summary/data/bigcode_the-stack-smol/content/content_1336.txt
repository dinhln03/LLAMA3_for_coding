import constraint

coins = [1, 2, 5, 10, 20, 50, 100, 200]

CSP = constraint.Problem()
for coin in coins:
    CSP.addVariable(coin, range(0, 201, coin))
CSP.addConstraint(constraint.ExactSumConstraint(200))
print len(CSP.getSolutions())