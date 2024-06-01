# Refaça o exercicio009, mostrando a tabuada de um número que um usuário escolher utilizando FOR.

print('=-='*3)
print('TABUADA')
print('=-='*3)

m = 0
n = int(input('Digite o número que deseja saber a tabuada: '))
for c in range(1, 11):
    m = n * c
    print('{} x {} = {}.'.format(n, c, m))
