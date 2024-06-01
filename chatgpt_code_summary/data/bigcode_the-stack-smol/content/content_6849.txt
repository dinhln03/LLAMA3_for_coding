class Developer:
    def __init__(self,name):
        self.name = name

    def coding(self):
        print(self.name+' is developer!')

class PythonDevloper(Developer):
    def coding(self):
        print(self.name + ' is Python developer!')

class JavaDevloper(Developer):
    def coding(self):
        print(self.name + ' is Java developer!')

class CPPDevloper(Developer):
    def coding(self):
        print(self.name + ' is C++ developer!')

dev1 = PythonDevloper('Chris')
dev2 = JavaDevloper('Jason')
dev3 = CPPDevloper('Bryan')

dev1.coding()
dev2.coding()
dev3.coding()
