P1 = float(input('Informe o primeiro preço: '))
P2 = float(input('Informe o primeiro preço: '))
P3 = float(input('Informe oprimeiro preço: '))

if (P1<P2) and (P1<P3):
    print('O preço menor é {}'.format(P1))
elif (P2<P1) and (P2<P3):
    print('O menor preço é {}'.format(P2))
else:
    print('O menor preço é {}'.format(P3))
