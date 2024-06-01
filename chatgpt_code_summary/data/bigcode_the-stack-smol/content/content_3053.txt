# Funcoes servem para quando tivermos coisas repetitivas poder simplificar o programa


def lin():   # para definir um afuncao ela tem que ter parenteses no finalk
    print('=-'*30)
lin()
print('Bem Vindo')
lin()
nome = str(input('Qual seu nome? '))
lin()
print(f'Tenha um otimo dia {nome}!')
lin()


def mensagem(msg):
    print('-'*30)
    print(msg)     # A mensagem que vai aparecer aqui o usuario vai digitar quando chamar a funcao
    print('-'*30)


mensagem('SISTEMA DE ALUNOS')


