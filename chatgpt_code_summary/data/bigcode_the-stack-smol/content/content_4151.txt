import math
numero_turmas = int(input('Qual o número de turmas? '))
for _ in range(numero_turmas):
    numero_alunos = int(input('Qual o número de alunos? '))
    soma = 0
    menor = math.inf
    maior = 0
    for i in range(numero_alunos):
        nota = float(input(f'Qual a nota do aluno {i + 1}? '))
        soma += nota
        if menor > nota:
            menor = nota
        if maior < nota:
            maior = nota

    print(f'A média é {soma / numero_alunos}. A menor nota é {menor}, e a maior é {maior}.')
