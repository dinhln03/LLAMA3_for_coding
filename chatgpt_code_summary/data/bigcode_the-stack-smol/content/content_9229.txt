'''entre no sistema com um valor float e saia com a sua
parte inteira'''

cores = {'limpa': '\033[m', 'azul': '\033[1;34m'}
print('{:-^40}'.format('PARTE INTEIRA DE UM VALOR'))
num = float(input('Digite um valor com ponto [Ex: 1.20]: '))
print('{}{}{} - sua parte inteira é - {}{}{} '
	.format(cores['azul'], num, cores['limpa'], cores['azul'], round(num), cores['limpa']))
print('{:-^40}'.format('FIM'))
