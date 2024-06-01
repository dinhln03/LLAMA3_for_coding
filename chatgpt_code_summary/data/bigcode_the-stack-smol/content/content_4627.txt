from time import sleep
from random import randint
numeros = []


def sorteio():
    c = 0
    while True:
        n = randint(0, 20)
        numeros.append(n)
        c = c+1
        if c == 5:
            break
    print('=-'*20)
    print('SORTEANDO OS 5 VALORES DA LISTA:', end=' ')
    for n in numeros:
        sleep(0.5)
        print(n, end=' ')
    print()


def somapar():
    soma = 0
    for n in numeros:
        if n % 2 == 0:
            soma = soma + n
    sleep(2)
    print(f'Somando os valores PARES de {numeros}: {soma}')


sorteio()
somapar()
