from bayesianABTest import sampleSuccessRateForBinomial
from numpy import mean

def bestOfFive(A,B,C,D,E,F):
    return mean( (A > B) & (A > C) & (A > D) & (A > E) & (A > F))

############# Example: Binomial Distribution #############

# Actual data for all cases
installs = [986,1013,959,968,1029,1014]
returns = [340,298,274,287,325,291]


A = sampleSuccessRateForBinomial(installs[0],returns[0])
B = sampleSuccessRateForBinomial(installs[1],returns[1])
C = sampleSuccessRateForBinomial(installs[2],returns[2])
D = sampleSuccessRateForBinomial(installs[3],returns[3])
E = sampleSuccessRateForBinomial(installs[4],returns[4])
F = sampleSuccessRateForBinomial(installs[5],returns[5])

A_best = bestOfFive(A,B,C,D,E,F)
B_best = bestOfFive(B,A,C,D,E,F)
C_best = bestOfFive(C,B,A,D,E,F)
D_best = bestOfFive(D,B,C,A,E,F)
E_best = bestOfFive(E,B,C,D,A,F)
F_best = bestOfFive(F,B,C,D,E,A)

# Get samples from the posterior
print "The probability of 20 being the best choice is {}".format(A_best)
print "The probability of 21 being the best choice is {}".format(B_best)
print "The probability of 22 being the best choice is {}".format(C_best)
print "The probability of 23 being the best choice is {}".format(D_best)
print "The probability of 24 being the best choice is {}".format(E_best)
print "The probability of 25 being the best choice is {}".format(F_best)
