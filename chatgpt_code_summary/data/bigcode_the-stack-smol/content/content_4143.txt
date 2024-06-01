def notas(*n, sit=False):
    """
    Função para analisar notas e situação de varios alunos.
    :param n: Uma ou mais notas dos alunos (aceita varias)
    :param sit: Valor opcional, indicando se deve ou não adicionar a situação.
    :return: Dicionario com varias informações sobre a situação da turma.
    """
    dic = dict()
    dic["total"] = len(n)
    dic["maior"] = max(n)
    dic["menor"] = min(n)
    dic["media"] = sum(n) / len(n)

    if sit:
        if media < 5:
            dic["situação"] = "Critica"
        elif media < 7:
            dic["situação"] = "Rasoavel"
        else:
            dic["situação"] = "Boa"

    return dic


resp = notas(5, 4, 3, sit=True)
print(resp)
