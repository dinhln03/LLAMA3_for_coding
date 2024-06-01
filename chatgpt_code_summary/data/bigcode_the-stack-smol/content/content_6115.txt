"""
For...in em Python
Iterando strings com for...in
função range recebe esses argumentos (start=0, stop, step=1)
"""

texto = input('informe seu CPF: ')
texto_novo = ''

for letra in range(len(texto)):

    if  letra % 3 == 0:
        texto_novo += '.' + texto[letra]
        continue

    texto_novo += texto[letra]
print(texto_novo[1:])