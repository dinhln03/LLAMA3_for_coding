n1 = float(input('Digite sua primera nota: '))
n2 = float(input('Digite sua segunda nota: '))
media = (n1 + n2) / 2
if media <= 5:
    print('Sua média é {} e você está REPROVADO!'.format(media))
elif media >= 7:
    print('Sua média é {} e você está APROVADO!'.format(media))
elif media >5 or media <6.9:
    print('Sua média é {} e você está de RECUPERAÇÃO '.format(media))
