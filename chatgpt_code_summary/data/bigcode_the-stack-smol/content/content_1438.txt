# coding=utf-8
# Exemplos para entendiemnto
"""nome = input('Qual seu nome?' )
if nome == 'Rodrigo' or nome == 'RAYANNE':
    print('Que nome lindo vocé tem!')
else:
    print('Que nome tão normal!!!')
print('Bom dia, {}'.format(nome))"""
n1 = float(input('Digite a primeira nota: '))
n2 = float(input('Digite a segunda nota: '))
m = (n1 + n2) / 2
print('A sua média foi: {:.1f}'.format(m))

print('A sua media foi boa!' if m >= 6.0 else 'Sua media foi ruim,estude mais!')

"""if m >= 6.0:
    print('Sua média foi boa!')
else:
    print('A sua média foi ruim,estude mais!')"""
