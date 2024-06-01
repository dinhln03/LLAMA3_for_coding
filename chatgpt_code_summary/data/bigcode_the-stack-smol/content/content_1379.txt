# ----------------------------------------------------------------
# ---------- ASSOCIATION RULE MINING : NOTEABLE ATTEMPT 2 ---------
# ----------------------------------------------------------------


# ------------------ DAILY DATASET --------------------

association_rules = apriori(dailyRankedCrimes.values, min_support=0.02, min_confidence=0.95, min_lift=3, min_length=4, use_colnames = True)
association_results = list(association_rules)
print(len(association_results))
# 17


# ------------------ YEARLY DATASET --------------------

association_rules = apriori(yearlyRankedCrimes.values, min_support=0.02, min_confidence=0.95, min_lift=3, min_length=4, use_colnames = True)
association_results = list(association_rules)
print(len(association_results))
# 2

# Not Many Rules, playing with the settings:

association_rules = apriori(yearlyRankedCrimes.values, min_support=0.0045, min_confidence=0.95, min_lift=1, min_length=2, use_colnames = True)
association_results = list(association_rules)
print(len(association_results))
# 41

# This is better

# I printed the Rules using the common commands (found in common-commands.py)