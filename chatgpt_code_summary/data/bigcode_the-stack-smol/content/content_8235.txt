valor100= (mediadia-100)*taxa+(diaria*dias)
mediadia=km//dias
diaria=90
taxa=12
carro=str('você deseja alugar um carro? ').upper()
if carro=='SIM':
    dias=int(input('quantos dias você deseja ficar com o carro'))
    km=int(input('quantos km tem sua viagem?'))
    if mediadia<=100 :
        print('o valor a ser pago é {}'.format(dias*diaria)
    