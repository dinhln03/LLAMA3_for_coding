lista = ['item1', 'item2', 'item3', 123, 12.43, 898.34, 00.989]
print(lista)
del lista[0] # pode remover tudo ou apenas um item de um indice permanentemente
popped = lista.pop(0) #pode remover um item pelo indice de uma lista, porem o item tirado pode ser posto em uma variavel
lista.remove('item3') # pode remover um item pelo valor / remove o primeiro valor da lista
print(lista)
print(popped)

valores = list(range(0, 11)) # list cria uma lista
print(valores)

valores_1 = [7, 3, 6, 9, 7, 8, 2, 1, 78, 90, 23, 45, 56, 21, 3]
print(valores_1)
valores_1.sort() # coloca os itens de forma ordenada em uma lista permanentemente
print(valores_1)
valores_1.sort(reverse=True) # faz o mesmo de cima, porem inverte a ordem
print(valores_1)
print(len(valores_1)) # len conta a quantidade de elementos de uma lista ou varavel

valores_1[0] = 'new' #substitui um valor de indice por outro no mesmo indice
valores_1.append('alex') # adiciona um objeto no final da lista
valores_1.insert(4, 'camila') # insere um item em um indice especifico
print(valores_1)
print('\n')

a = [12, 43, 76, 35, 24] # lista
b = a # cria uma ligação com a lista
print(f'{a}\n{b}')
b.remove(43)
print(f'{a}\n{b}')

c = a[:] # [:] usado para criar uma copia de uma lista que pode ser modificado como quiser
c.append('jack')
print(f'{a}\n{b}\n{c}')
