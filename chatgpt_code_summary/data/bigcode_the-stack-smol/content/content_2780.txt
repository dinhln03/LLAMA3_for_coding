class Pessoa:
    def __init__(self,nome,idade,cpf,salario):
        self.nome = nome
        self.idade = idade
        self.cpf = cpf
        self.salario = salario
        
    def Aumento(self):
        return self.salario *0.05
    

class Gerente(Pessoa):
      def __init__(self,nome,idade,cpf,salario,senha):
          super().__init__(nome,idade,cpf,salario)
          self.senha = senha
      def Aumento(self):
          return self.salario * 0.01 + 1000
p = Gerente('Fabio',25,41075570816,21000,456578)
print(p.nome)
print(p.idade)
print(p.cpf)
print(p.senha)
print(p.salario)
print(p.Aumento())

print('='*30)


class Animal:
    def __init__(self,nome,raca,cor,peso,comportamento = True):
        self.nome = nome
        self.raca = raca
        self.cor = cor
        self.peso = peso
        self.comportamento = comportamento

    def Comportamento(self):
        if(self.comportamento == False):
            return self.peso + 500
            print('Ta Gordo por sem ruim')
    
class Pitbull(Animal):
    pass
    #def Comportamento(self):
        #return False

dog = Pitbull('Luci','Pitbull','Preta',53,False)

print(dog.nome)
print(dog.raca)
print(dog.cor)
print(dog.peso)
print(dog.Comportamento())




        
