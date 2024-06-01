"""
Polimorfismo e o principio onde permite que classes derivadas
de uma mesma superclasse tenha metodos iguais (com a mesma
assinatura) mas comportamentos diferentes
Mesma assinatura - Mesma quantidade e tipo de parametros
"""
from abc import ABC, abstractmethod


class A(ABC):
    @abstractmethod
    def fala(self, msg):
        pass


class B(A):
    def fala(self, msg):
        print(f'B está falando {msg}')


class C(A):
    def fala(self, msg):
        print(f'C está falando {msg}')


b = B()
c = C()

b.fala('maça')
c.fala('banana')
