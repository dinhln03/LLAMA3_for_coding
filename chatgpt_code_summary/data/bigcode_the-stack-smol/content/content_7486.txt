numeros = [[], []]
for c in range(0, 7):
    n = int(input(f'Digite o {c+1}o. número: '))
    if n % 2 == 0:
        numeros[0].append(n)
    elif n % 2 == 1:
        numeros[1].append(n)
numeros[0].sort()
numeros[1].sort()
print('='*30)
print(f'Números pares digitados: {numeros[0]}')
print(f'Números ímpares digitados: {numeros[1]}')
