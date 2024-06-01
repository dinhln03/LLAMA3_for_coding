import pdb

if __name__ == "__main__":
    with open("21input.txt") as f:
        data = f.read().split("\n")
        data.pop(-1)
        print(data)
    all_food = []
    for food in data:
        allergens = False
        ings = []
        alle = []
        for ingredient in food.split(" "):
            if "(contains" == ingredient:
                allergens = True
            elif allergens:
                alle.append(ingredient[:-1])
            else:
                ings.append(ingredient)
        all_food.append([ings, alle])
    print(all_food)
    alg_dico = {}
    assigned = {}
    for food in all_food:
        for alg in food[1]:
            if alg in alg_dico:
                alg_dico[alg] &= set(food[0])
            else:
                alg_dico[alg] = set(food[0])

    solved = []
    unsolved = []

    for alg, val in alg_dico.items():
        if (len(val) == 1):
            solved.append(alg)
        else:
            unsolved.append(alg)
    for alg in alg_dico.keys():
        alg_dico[alg] = list(alg_dico[alg])

    print(alg_dico, solved, unsolved)
    while (len(unsolved)>0) :
        for alg in solved:
            val = alg_dico[alg][0]
            for algx in unsolved:
                if val in (alg_dico[algx]):
                    alg_dico[algx].remove(val)
                    if len(alg_dico[algx]) == 1:
                        solved.append(algx)
                        unsolved.remove(algx)


    used_ing = list(alg_dico.values())
    used_ing = [x[0] for x in used_ing]
    # for alg, val in alg_dico.items():
    #     if (len(val) == 1):
    #         for valx in alg_dico.values():
    #             if val in valx and valx != val:
    #                 valx.remove(val)


    print(used_ing)
    cpt = 0
    for ings, algs in all_food:
        for ing in ings:
            if ing not in used_ing:
                cpt+=1

    print(cpt)
    algs = list(alg_dico.keys())
    algs.sort()
    used_ing_sorted = []
    for alg in algs:
        used_ing_sorted.append(alg_dico[alg][0])

    print(used_ing_sorted, ",".join(used_ing_sorted))
