###Titulo: Multiplicação através de repetidas somas
###Função: Este programa realiza a multiplicação de dois números através de sucessivas adições
###Autor: Valmor Mantelli Jr.
###Data: 14/12/2018
###Versão: 0.0.5

# Declaração de variáve

x = 0

y = 0

w = 0

z = 1

# Atribuição de valor a variavel

x = int(input("Diga o primeiro número: "))

y = int(input("Diga por qual número deseja multiplicar: "))

# Processamento

while z <= x:
	w +=  y
	z +=  1
# Saída

print("%d x %d = %d" % (x, y, w))

	
