import pandas as pd
from utils.config import Config
import numpy as np
import pandas as pd


def fun_clean_categogy1(array, keyw1, index, BOW):
    compty = 0
    c = 0
    for elm in array:
        if elm == "oui" or elm == "parfois":
            BOW[c].append(keyw1[index])
            compty += 1
        c += 1
    # print(compty)
    return BOW

#Ajout des keywords de la catégorie 2 ATTENTION, ici j'ajoute tout le contenu des colonnes, donc il peut y avoir
# une grande variété de mots qui sugissent à cause d'ici. De plus, ce sont souvent des mots composés ou des
# séquences de mots. On peut envisager de ne sélectionner que le premier mot par exemple.
def fun_clean_categogy2(array, BOW):
    compty = 0
    c = 0
    for elm in array:
        if not elm == "":
            if not BOW[c].__contains__(elm):
                BOW[c].append(elm)
                compty += 1
        c += 1
    # print(compty)
    return BOW


def fun_clean_categogy3(array, keyw3, index, BOW, list_THR):
    compty = 0
    c = 0
    for elm in array:
        # print(elm)
        if not np.isnan(float(str(elm).replace(",", "."))):
            if float(str(elm).replace(",", ".")) > list_THR[index]:
                if not BOW[c].__contains__(elm):
                    BOW[c].append(keyw3[index])
                    compty += 1
        c += 1
    print(compty)
    return BOW


if __name__ == '__main__':
    # %%
    df = pd.read_csv(Config.csv_files[-1], sep=';', encoding='ISO-8859-1')
    df.columns
    #
    # d = {'col1': [1, 2], 'col2': [3, 4]}
    # df = pd.DataFrame(data=d)


    List_cat1 = ["difficulté endormisst", "fatigue au reveil", "hyperacousie", "surdité", "SDE", "vertiges",
                 "depression", "anxiété"]

    #Keywords à associer aux colonnes de la catégorie 1
    keyw1 = ["endormissement", "fatigue", "hyperacousie", "surdité", "somnolence", "vertige", "dépression", "anxiété"]

    List_cat2 = ["timbre acouphène", "type de douleurs", "type otalgie", "type de vertiges",
                 "caractere particulier", "mode apparition"]

    List_cat3 = ["EVA  depression", "epworth", "EVA anxiété", "EVA douleurs", "EVA hyperac", "EVA hypoac",
                 "EVA Otalgie 1", "EVA SADAM", "EVA vertiges", "ISI", "score khalfa hyperacousie", "EVA  concentration"]

    # Keywords à associer aux colonnes de la catégorie 3
    keyw3 = ["dépression", "somnolence", "anxiété", "douleurs", "hyperacousie", "hypoacousie", "otalgie", "mâchoire",
             "vertige", "sommeil", "hyperacousie", "concentration"]

    # seuils de sélections à associer aux colonnes de la catégorie 3
    List_THR = [5, 10, 5, 5, 5, 5, 4, 3, 3, 12, 20, 5]

    cat4 = ["intensité ac"]

    compt = 0
    #Liste de mots clés associés à chaque patient. Une liste par patient
    BOW = [[] for i in range(len(df[df.columns[0]]))]

    #ajout des keywords de la categorie 1 à la liste des bag of words BOW
    for colname in List_cat1:
        # print(df[colname])  # show value before
        print(colname)
        BOW = fun_clean_categogy1(df[colname], keyw1, compt, BOW)
        compt += 1

    # ajout des keywords de la categorie 2 à la liste des bag of words BOW
    compt=0
    for colname in List_cat2:
        print(colname)
        BOW = fun_clean_categogy2(df[colname], BOW)
        compt += 1

    # ajout des keywords de la categorie 3 à la liste des bag of words BOW
    compt=0
    for colname in List_cat3:
            print(colname)
            BOW = fun_clean_categogy3(df[colname], keyw3, compt, BOW, List_THR)
            compt += 1

    #Nettoyage des valeurs "NaN" copiées par erreur par la catégorie 2
    for elm in BOW:
        if elm.__contains__(np.nan):
            elm.pop(elm.index(np.nan))

    print(BOW[:200])  # petit extrait de la liste des bag of words
    BOW2=[]
    for elm in BOW:
        stri=""
        for st in elm:
            stri = stri + " " + st
        BOW2.append(stri)

    df2 = pd.DataFrame(BOW2)
    df2.to_csv('lettres_persanes.csv', sep=';', encoding='ISO-8859-1')
    print(df2)
