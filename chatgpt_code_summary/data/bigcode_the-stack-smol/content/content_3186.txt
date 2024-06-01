def testaArq(arq):
    """
    -> Verifica se existe o arquivo arq
    :arq: Nome do arquivo a ser testado.
    :return: retorna True se o arquivo for encontrado,
    caso contrário False
    """
    try:
        a = open(arq)
    except FileNotFoundError:  # O arquivo não foi encontrado
        print('Arquivo não encontrado!')
        return False
    else:
        return True


def criaArq(arq=''):
    """
    -> Cria um arquivo de texto, caso ele não exista.
    :param arq: Nome do arquivo.
    :return:
    """
    try:
        a = open(arq, 'xt')
    except FileExistsError:
        print(f'ERRO: o arquivo \"{arq}\" já existe!')
    else:
        print(f'O arquivo \"{arq}\" foi criado com sucesso!')
    finally:
        a.close()
    return


def leArq(arq=''):
    """
        -> Abre e mostra os itens de um arquivo texto.
        :param arq: Nome do arquivo.
        :return:
        """
    return


def editaArq(arq):
    """
    -> Abre um arquivo de texto e adiciona novo item no 
    final do arquivo.
    :param arq: Nome do arquivo.
    :return:
    """
    return
