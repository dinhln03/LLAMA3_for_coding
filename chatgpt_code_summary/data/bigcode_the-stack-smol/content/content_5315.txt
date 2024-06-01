import random
import time
pc=random.randint(0,10)
tentativas=0
chute=0
print('INICIANDO')
time.sleep(2)
while chute != pc:
    chute=int(input('Digite um número entre 0 a 10: '))
    print('PROCESSANDO')
    time.sleep(1)
    if chute < pc:
        print('Mais, tente novamente.')
    elif chute > pc:
        print('Mneos, tente novamente.')
    tentativas+=1
print('Você acertou com {} tentativas!'.format(tentativas))