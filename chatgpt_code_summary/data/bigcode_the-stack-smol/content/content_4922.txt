# Crie um programa onde o usuário possa digitar sete valores numéricos
# e cadastre-os em uma lista única que mantenha separados os valores pares e ímpares.
# No final, mostre os valores pares e ímpares em ordem crescente.
lista_unic = [[], []]
print('-=' * 20)
for c in range(0, 7):
    nums = int(input(f'Informe um {c+1}° valor: '))
    if nums%2 == 0:
        lista_unic[0].append(nums)
    else:
        lista_unic[1].append(nums)
print('-=-' * 30)
lista_unic[0].sort()
lista_unic[1].sort()
print(f'Os valores pares foram: {lista_unic[0]}')
print(f'Os valores ímpares foram: {lista_unic[1]}')
print('-=-' * 30)
