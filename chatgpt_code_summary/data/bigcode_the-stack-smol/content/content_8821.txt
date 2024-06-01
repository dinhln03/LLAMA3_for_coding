#crie um tupla com o nome dos produtos, seguidos do preço.
#mostre uma listagem de preços, de forma tabular.

lista = ('Lápis', 1.5, 'Borracha', 2.5, 'Caderno', 10.8,
         'Estojo', 20, 'Mochila', 100.5)

print('\033[31m--'*20)
print(f'{"LISTAGEM DE PREÇOS":^40}')
print('--'*20, '\033[m')

for i in range(0, len(lista), 2):
    print(f'{lista[i]:.<30}R${lista[i+1]:>5.2f}')
print('\033[31m--\033[m'*20)

''' Formatação:
print(f'{"LISTAGEM DE PREÇOS":^40}')
centralizado = {elemento:^quantidade}
à direita = {:<quantidade} > preenche com espaço
à direita = {:.<quantidade} > preenche com ponto
à esquerda = {:>quantidade} > preenche com espaço
à esquerda = {:->quantidade} > preenche com -
'''
