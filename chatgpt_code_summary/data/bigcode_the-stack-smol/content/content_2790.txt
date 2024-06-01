from random import randint
from time import sleep

def sorteio(lista):
    print('-=' * 30)
    print('Sorteando 5 valores da lista: ', end='')
    for i in range(0, 5):
        lista.append(randint(1, 10))
        print(f'{lista[i]} ', end='', flush=True)
        sleep(0.3)
    print('PRONTO!')

def somaPar(lista):
    print('-=' * 30)
    pares = 0
    for num in lista:
        if num % 2 == 0:
            pares += num
    print(f'Somando os valores pares de {lista}, temos {pares}')

# Programa principal
numeros = []
sorteio(numeros)
somaPar(numeros)
print('-=' * 30)
