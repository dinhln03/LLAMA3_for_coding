v1=float(input("Digite o valor de um dos lados "))
v2=float(input("Digite o valor de outrpo lados "))
v3=float(input("Digite o valor do ultimo lado lados "))
if v1+v2 > v3 and v1+v3 > v2 and v2+v3 > v1:
    if v1 == v2 == v3:
        print(f'Com esses valores é possivel formar um triangulo equilátero')
    elif v1 != v2 or v3:
        print(f'Com essas medidas pode ser formado um triangulo escaleno')
    elif v1 == v2 and v2 != v3 or v1 == v3 and v3 != v2 or v3 == v2 and v2 != v1:
        print(f'Com essas medidas pode ser formado um triangulo isóceles')
else:
     print(f'Com essas medidas não pode ser formado um triangulo')
