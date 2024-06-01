from interface import *    

class M2:
    interface = Interface()
    def __init__(self, tamanhoDaLista, tempoDeAtraso, charPixel = '  '):
        self.guardarNumero = 0
        
        self.interface.set_tamanhoDaLista(tamanhoDaLista)
        self.interface.set_tempoDeAtraso(tempoDeAtraso)
        self.interface.set_charPixel(charPixel)

    
    def Maneira2(self):
        self.guardarNumero
        
        for c in range(len(self.interface.lista)):
            for i in range(len(self.interface.lista)):

                if i+1 == len(self.interface.lista):
                    continue
                else:
                    if self.interface.lista[i] > self.interface.lista[i+1]:
                        guardarNumero = self.interface.lista[i]
                        self.interface.lista[i] = self.interface.lista[i+1]
                        self.interface.lista[i+1] = guardarNumero
                        self.interface.converterPMostrar(i+1)

            for i in reversed(range(len(self.interface.lista))):

                if i+1 == len(self.interface.lista):
                    continue
                else:
                    if self.interface.lista[i] > self.interface.lista[i+1]:
                        guardarNumero = self.interface.lista[i]
                        self.interface.lista[i] = self.interface.lista[i+1]
                        self.interface.lista[i+1] = guardarNumero
                        self.interface.converterPMostrar(i)                        
 
            
            
            