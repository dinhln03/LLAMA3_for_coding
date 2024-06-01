def metade(x=0):
    res = x / 2
    return res


def dobro(x=0):
    res = 2 * x
    return res


def aumentar(x=0, y=0):
    res = x * (1 + y / 100)
    return res


def reduzir(x=0, y=0):
    res = x * (1 - y / 100)
    return res


def moeda(x=0, m='R$'):
    res = f'{m}{x:.2f}'.replace('.', ',')
    return res
