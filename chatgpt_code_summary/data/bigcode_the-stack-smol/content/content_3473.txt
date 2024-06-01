#Curso Python #06 - Condições Aninhadas

#Primeiro Exemplo

#nome = str(input('Qual é seu Nome: '))
#if nome == 'Jefferson':
#    print('Que Nome Bonito')
#else:
#    print('Seu nome é bem normal.')
#print('Tenha um bom dia, {}'.format(nome))

#Segundo Exemplo

nome = str(input('Qual é seu Nome: '))
if nome == 'Jefferson':
    print('Que Nome Bonito')
elif nome == 'Pedro' or nome == 'Marcos' or nome == 'Paulo':
    print('Seu nome é bem popular no Brasil.')
elif nome in 'Jennifer Vitoria Mariana Deborah':
    print('Belo nome você tem em !')
else:
    print('Seu nome é bem normal.')
print('Tenha um bom dia, {}'.format(nome))