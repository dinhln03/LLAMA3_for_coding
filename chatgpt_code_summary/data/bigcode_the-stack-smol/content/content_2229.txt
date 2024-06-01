n = int(input())
c = int(input())
lista = input().split()
graph = [[0 for i in range(n)] for j in range(n)]
cont = 0
for i in range(n):
    for j in range(n):
        graph[i][j] = int(lista[cont])
        cont += 1
        if i == j:
            graph[i][j] = 0
listaMemoria = [c]
contaminados = []
contaminados.append(c)
k = 1
while True:
    veLinha = listaMemoria[-1]
    check = 0
    for i in range(n):
        if graph[veLinha][i] == 1:
            graph[veLinha][i] = 0
            graph[i][veLinha] = 0
            listaMemoria.append(i)
            contaminados.append(i)
            check = 1
            k += 1
            break
    if check == 0:
        if listaMemoria[-1] == c:
            break
        else:
            listaMemoria.pop()
print(k)