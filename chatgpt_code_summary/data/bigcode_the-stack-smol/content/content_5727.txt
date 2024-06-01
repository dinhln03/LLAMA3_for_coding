def area(l,c):
    a = l*c
    return f'A area de um terreno {l}x{c} e de {a}m²'
print("Controle de Terrenos")
print()
largura = float(input("Largura (m): "))
altura = float(input("altura (m): "))
print(area(largura,altura))