horasextra = int(input("¿Cuantas horas extra has trabajado? "))
horas = horasextra + 35 #elminimo
extra = 0
sueldo = 0
class trabajo:
    def __init__(self, horasextra, horas, extra, sueldo): #defino el constructor
        self.horasextra = horasextra
        self.horas = horas
        self.extra = extra
        self.sueldo = sueldo
    
    def horas_totales(self):
        if 36 < self.horas < 43:
            self.extra = float(self.horas*17) * 1.25 
            self.sueldo = (35 * 17) + self.extra  
            print("Ha trabajado: ",horasextra,"horas extra y su sueldo es: ",self.sueldo, "€ ya que ha trabajado en total: ",self.horas,"horas")
            
        if self.horas >= 44:
            self.extra = float(self.horas*17) * 1.50
            self.sueldo = (35*17) + self.extra
            print("Ha trabajado: ",horasextra,"horas extra y su sueldo es: ",self.sueldo,"€ ya que ha trabajado en total: ",self.horas,"horas")
            
resultado = trabajo(horasextra, horas, extra, sueldo)
print(resultado.horas_totales())