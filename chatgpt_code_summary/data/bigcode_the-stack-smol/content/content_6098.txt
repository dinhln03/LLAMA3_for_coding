#!/bin/python3
"""
https://www.hackerrank.com/challenges/crossword-puzzle/problem?h_l=interview&playlist_slugs%5B%5D=interview-preparation-kit&playlist_slugs%5B%5D=recursion-backtracking&h_r=next-challenge&h_v=zen
"""
# Complete the crossword_puzzle function below.
def crossword_puzzle(crossword, words):
    """resuelve el puzzle"""
    palabras = words.split(";")
    puzzle_y = len(crossword)
    puzzle_x = len(crossword[0])

    pos = []
    for i in palabras:
        pos.append([0, 0, "x", 1])

    cruces = []
    sig = 0
    j = 0
    while j < puzzle_y:
        i = 0
        while i < puzzle_x:
            if (
                    crossword[j][i] == "-"
                    or (
                        i + 1 < puzzle_x
                        and crossword[j][i] == "v"
                        and crossword[j][i + 1] == "-"
                    )
                    or (
                        j + 1 < puzzle_y
                        and crossword[j][i] == "h"
                        and crossword[j + 1][i] == "-"
                    )
            ):
                if crossword[j][i] != "-":
                    cruces.append([sig, i, j])
                crossword[j] = crossword[j][:i] + "i" + crossword[j][i + 1 :]
                pos[sig][0] = i
                pos[sig][1] = j
                sig += 1
                iter_i = i + 1
                iter_j = j + 1
                while iter_i < puzzle_x and (
                        crossword[j][iter_i] == "-"
                        or crossword[j][iter_i] == "v"
                ):
                    pos[sig - 1][2] = "h"
                    pos[sig - 1][3] += 1
                    if crossword[j][iter_i] == "v":
                        crossword[j] = (
                            crossword[j][:iter_i]
                            + "x"
                            + crossword[j][iter_i + 1 :]
                        )
                        cruces.append([sig - 1, iter_i, j])
                    else:
                        crossword[j] = (
                            crossword[j][:iter_i]
                            + "h"
                            + crossword[j][iter_i + 1 :]
                        )
                    iter_i += 1
                while iter_j < puzzle_y and (
                        crossword[iter_j][i] == "-"
                        or crossword[iter_j][i] == "h"
                ):
                    pos[sig - 1][2] = "v"
                    pos[sig - 1][3] += 1
                    if crossword[iter_j][i] == "h":
                        crossword[iter_j] = (
                            crossword[iter_j][:i]
                            + "x"
                            + crossword[iter_j][i + 1 :]
                        )
                        cruces.append([sig - 1, i, iter_j])
                    else:
                        crossword[iter_j] = (
                            crossword[iter_j][:i]
                            + "v"
                            + crossword[iter_j][i + 1 :]
                        )
                    iter_j += 1
            i += 1
        j += 1

    for palabra_aux1 in pos:
        posibles = []
        for pal in palabras:
            if len(pal) == palabra_aux1[3]:
                posibles.append(pal)
        palabra_aux1.append(posibles)

    for cruce in cruces:
        i = 0
        while i < len(pos):
            if pos[i][2] == "h":
                if (
                        pos[i][0] <= cruce[1]
                        and pos[i][0] + pos[i][3] >= cruce[1]
                        and pos[i][1] == cruce[2]
                ):
                    break
            if pos[i][2] == "v":
                if (
                        pos[i][1] <= cruce[2]
                        and pos[i][1] + pos[i][3] >= cruce[2]
                        and pos[i][0] == cruce[1]
                ):
                    break
            i += 1
        letra1 = abs(cruce[1] - pos[i][0] + cruce[2] - pos[i][1])
        letra2 = abs(pos[cruce[0]][0] - cruce[1] + pos[cruce[0]][1] - cruce[2])
        palabra_aux1 = ""
        palabra_aux2 = ""
        for palabra1 in pos[i][4]:
            for palabra2 in pos[cruce[0]][4]:
                if palabra1[letra1] == palabra2[letra2]:
                    palabra_aux1 = palabra1
                    palabra_aux2 = palabra2
                    break
        pos[i][4] = [palabra_aux1]
        pos[cruce[0]][4] = [palabra_aux2]

    for pal in pos:
        if pal[2] == "h":
            crossword[pal[1]] = (
                crossword[pal[1]][: pal[0]]
                + pal[4][0]
                + crossword[pal[1]][pal[0] + pal[3] :]
            )
        else:
            i = 0
            while i < pal[3]:
                crossword[pal[1] + i] = (
                    crossword[pal[1] + i][: pal[0]]
                    + pal[4][0][i]
                    + crossword[pal[1] + i][pal[0] + 1 :]
                )
                i += 1

    return crossword


# ++H+F+++++++++
# +RINOCERONTE++
# ++E+C++++++L++
# ++N+AGUILA+E++
# ++A++++++++F++
# +++++++++++A++
# +++++++++++N++
# +++++++++++T++
# +++++++++++E++

CROSSWORD = [
    "++-+-+++++++++",
    "+-----------++",
    "++-+-++++++-++",
    "++-+------+-++",
    "++-++++++++-++",
    "+++++++++++-++",
    "+++++++++++-++",
    "+++++++++++-++",
    "+++++++++++-++",
]
WORDS = "AGUILA;RINOCERONTE;ELEFANTE;HIENA;FOCA"
for x in crossword_puzzle(CROSSWORD, WORDS):
    print(x)
