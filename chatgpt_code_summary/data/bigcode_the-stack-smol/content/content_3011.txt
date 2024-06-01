# COMBINATION:
# combination is all the different ways that we can group something where the order does not matter.
# PERMUTATION:
# permutation is all the different ways that we can group something where the order does matter.
import itertools
my_list=[1,2,3]

my_combinations=itertools.combinations(my_list,2)# here first arguement is a list and second arguement is how many items we want in a group. it is r of nCr.
for c in my_combinations:
    print(c)

my_permutations=itertools.permutations(my_list,2)
for c in my_permutations:
    print(c)

# When we should use combination and permutation?
# if the order doent matter we should use combinations.
import itertools
my_list=[1,2,3,4,5,6]

my_combinations=itertools.combinations(my_list,3)
answer=[results for results in my_combinations if sum(results)==10]
for i in answer:
    print(i)

# if the order does matter we should use permutations. 
# word macthing game. 
import itertools
word="sample"
my_letters="pslame"
my_permutations=itertools.permutations(my_letters,len(my_letters))
for p in my_permutations:
    if "".join(p) == word:
        print("Match!")
        break
    else:
        print("No match")


    