from synbioweaver.core import *
from synbioweaver.aspects.designRulesAspect import *
from synbioweaver.aspects.printStackAspect import *
from synbioweaver.aspects.pigeonOutputAspect import *

declareNewMolecule('A')
declareNewMolecule('B')
declareNewMolecule('C')
declareNewMolecule('In')

declareNewPart('t1',Terminator)
declareNewPart('t2',Terminator)
declareNewPart('t3',Terminator)
declareNewPart('r1',RBS )
declareNewPart('r2',RBS )
declareNewPart('r3',RBS )
declareNewPart('cA',CodingRegion,moleculesAfter=[A])
declareNewPart('cB',CodingRegion,moleculesAfter=[B])
declareNewPart('cC',CodingRegion,moleculesAfter=[C])
declareNewPart('Pin', PositivePromoter, [In])
declareNewPart('Pb', NegativePromoter, [A] )
declareNewPart('Pc', HybridPromoter, [A,B], regulatorInfoMap={A:False,B:False} )

class simpleCircuit(Circuit):
    def mainCircuit(self):
        self.createMolecule(In)
        self.createMolecule(B)

        self.addPart(Pin)
        self.addPart(r1)
        self.addPart(cA)
        self.addPart(t1)
        self.addPart(Pb)
        self.addPart(r2)
        self.addPart(cB)
        self.addPart(t2)
        self.addPart(Pc)
        self.addPart(r3)
        self.addPart(cC)
        self.addPart(t3)
        
#compiledDesign = Weaver(constGFP, DesignRules, PrintStack, PigeonOutput).output()
compiledDesign = Weaver(simpleCircuit, PigeonOutput).output()

compiledDesign.printPigeonOutput()
