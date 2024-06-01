def conv(T,taille):
    # conv (list(list(bool)) * int -> list(list(int)))
    # Convertis un tableau à 2 dimensions contenent des booléens en tableau à 2 dimensions contenant des entiers tel que True = 1 et False = 0
    # T (list(list(bool))) : tableau à 2 dimensions contenant des booléens
    # taille (int) : taille du tableau à 2 dimensions
    
    # Initialisation et traitement
    # tableau (list(list(int))) : tableau à 2 dimensions contenant des entiers
    # En même temps que l'on parcours le tableau T, on construit le tableau tableau en suivant la règle True = 1 et False = 0
    tableau = [[0 if T[i][j] == False else 1 for j in range(taille)] for i in range(taille)]
    return tableau
    
