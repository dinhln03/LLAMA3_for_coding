def leiaint(msg):
    while True:
        try:
            n = int(input(msg))
        except(ValueError, TypeError):
            print("\033[31mERRO: Por favor, digite um numero inteiro valido.\033[m")
        except KeyboardInterrupt:
            print("\n\033[31mO usuario preferiu não digitar esse numero.")
            return 0
        else:
            return n


def leiafloat(msg):
    while True:
        try:
            n = float(input(msg))
        except(ValueError, TypeError):
            print("\033[31mERRO: Por favor, digite um numero inteiro valido.\033[m")
        except KeyboardInterrupt:
            print("\n\033[31mO usuario preferiu não digitar esse numero.")
            return 0
        else:
            return n


n1 = leiaint("Digite um numero inteiro: ")
n2 = leiafloat("Digite um numero real: ")
print(f"Você acabou de digitar o numero inteiro {n1} e o numero real {n2}!")