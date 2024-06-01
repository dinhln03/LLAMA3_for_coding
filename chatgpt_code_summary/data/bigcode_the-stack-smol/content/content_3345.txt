class Pessoa():
    
    def criar_lista(self, n, id_ade, sexo, saude):
        lista = []
        qunt = int(input('quantas pessoas são: '))
        for c in range(qunt):
            n = input('digite seu nome: ')
            lista.append(n)
            idade = int(input('digite sua idade: '))
            while idade < 0:
                print('idade invalida, digite novamente:')
                idade = int(input('digite sua idade: '))
            sexo = input('digite seu sexo: F ou M ').upper()
            while sexo != 'F' and sexo != 'M':
                print('sexo invalido, digite novamente: ')
                sexo = input('digite seu sexo: F ou M ').upper()
            saude = input('diga como está sua saúde: boa ou ruim ').upper()
            while saude != 'BOA' and saude != 'RUIM':
                print('opção de saude invalida. Digite novamente')
                saude = input('diga como está sua saúde: boa ou ruim ').upper()
            if idade < 18:
                print('voce nao está apta a cumprir o serviço militar obrigatório!')
                continue
            elif sexo == 'F':
                print('voce nao está apta a cumprir o serviço militar obrigatório!')
                continue
            elif saude == 'RUIM':
                print('voce nao está apta a cumprir o serviço militar obrigatório! Cuide da sua saude primeiro.')
                continue
            else:
                print('parabens, voce ESTÁ apta a cumprir o serviço militar obrigatório!')
        
        
    

pessoa = Pessoa()
pessoa.criar_lista(None, int, None, None)


