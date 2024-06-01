cor = {'traço': '\033[35m', 'ex': '\033[4;31m', 'título': '\033[1;34m', 'str': '\033[1;33m', 'reset': '\033[m'}
print('{}-=-{}'.format(cor['traço'], cor['reset'])*18, '{} Exercício 026  {}'.format(cor['ex'], cor['reset']),
      '{}-=-{}'.format(cor['traço'], cor['reset'])*18)
print('{}Faça um programa que leia uma frase pelo teclado e mostre quantas vezes aparece a letra "A", em que posição '
      'ela aparece a \nprimeira vez e em que posição ela aparece a última vez.{}'.format(cor['título'], cor['reset']))
print('{}-=-{}'.format(cor['traço'], cor['reset'])*42)
frase = str(input('Digite uma frase: ')).strip().upper()
print('A letra "A" aparece {}{}{} vezes na frase.'.format(cor['str'], frase.count('A'), cor['reset']))
print('A primeira vez que a letra "A" apareceu foi na posição {}{}{}.'
      .format(cor['str'], frase.find('A') + 1, cor['reset']))
print('A última vez que a letra "A" apareceu foi na posição {}{}{}.'
      .format(cor['str'], frase.rfind('A') + 1, cor['reset']))
