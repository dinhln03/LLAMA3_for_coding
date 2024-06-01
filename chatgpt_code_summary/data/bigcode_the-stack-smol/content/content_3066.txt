#!---------------------------------------------------------------------!
#! Written by Madu Manathunga on 07/01/2021                            !
#!                                                                     !
#! Copyright (C) 2020-2021 Merz lab                                    !
#! Copyright (C) 2020-2021 Götz lab                                    !
#!                                                                     !
#! This source file is a part of QUICK-GenInt code generator and       !
#! is subjected to the terms of the Mozilla Public License, v. 2.0.    !
#! If a copy of the MPL was not distributed with this file, you can    !
#! obtain one at http://mozilla.org/MPL/2.0/.                          !
#!_____________________________________________________________________!

#!---------------------------------------------------------------------!
#! This source file contains classes necessary for generating one      !
#! electron integrals. Note that we use vertical recurrence relations  !
#! algorithm developed by Obara and Saika. See J. Chem. Phys. 1986, 84,!
#! 3963−3974 paper for theoretical details.                            !
#!                                                                     !
#!---------------------------------------------------------------------!

import src.common.params as params
import src.common.file_handler as file_handler
from src.oei.iclass.OEint import OEint

# [p|f] class, subclass of OEint
class PFint(OEint):
    def gen_int(self):
        # write code paths for integrals. Note that we use C++ classes here.
        for m in range(0,self.max_m+1):
            if m == 0:
                self.fhc.write("\n/* PF true integral, m=%d */ \n" % (m)) 
                self.fhd.write("\n/* PF true integral, m=%d */ \n" % (m)) 
            else:
                self.fhc.write("\n/* PF auxilary integral, m=%d */ \n" % (m))
                self.fhd.write("\n/* PF auxilary integral, m=%d */ \n" % (m))              

            self.fhc.write("class PFint_%d{ \n" % (m))
            self.fhc.write("public: \n")

            # write class variables; convention being used is s=0, p=1-3, d=4-9, f=10-19, g=20-34
            self.fhc.write("#ifdef REG_PF \n")
            for i in range(0,10):
                for j in range(0,3):
                    self.fhc.write("  QUICKDouble x_%d_%d; // %s, %s \n" % (j+1, i+10, self.p_lbl[j], self.f_lbl[i]))
            self.fhc.write("#endif \n") 

            # write class functions
            self.fhc.write("  %s PFint_%d(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,\n\
                QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,\n\
                QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp); \n" % (self.func_qualifier, m))          
            self.fhc.write("}; \n")

            # write partial classes, these are useful to reduce the registry pressure 
            for i in range(0,10):
                self.fhc.write("\n/* PF integral partial class, - Part %d, m=%d */ \n" % (i+1, m)) 
                self.fhc.write("class PFint_%d_%d{ \n" % (m, i+1))
                self.fhc.write("public: \n")
                #self.fhc.write("#ifdef REG_PF \n")
                for j in range(0,3):
                    self.fhc.write("  QUICKDouble x_%d_%d; // %s, %s \n" % (j+1, i+10, self.p_lbl[j], self.f_lbl[i]))
                #self.fhc.write("#endif \n") 

                # write partial class functions
                self.fhc.write("  %s PFint_%d_%d(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,\n\
                QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,\n\
                QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp); \n" % (self.func_qualifier, m, i+1))          
                self.fhc.write("}; \n")

            # write function definitions
            self.fhd.write("%s PFint_%d::PFint_%d(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,\n\
                QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,\n\
                QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ \n\n" % (self.func_qualifier, m, m))

            self.fhd.write("  SDint_%d sd_%d(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|d] for m=%d \n" % (m, m, m))
            self.fhd.write("  SDint_%d sd_%d(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|d] for m=%d \n" % (m+1, m+1, m+1))
            self.fhd.write("  SFint_%d sf_%d(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|f] for m=%d \n" % (m, m, m))
            self.fhd.write("  SFint_%d sf_%d(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|f] for m=%d \n\n" % (m+1, m+1, m+1))

            # save all computed values into class variables that will reside in register/lmem space
            self.fhd.write("#ifdef REG_PF \n")
            for i in range(0,10):
                for j in range(0,3):
                    tmp_mcal=[params.Mcal[i+10][0], params.Mcal[i+10][1], params.Mcal[i+10][2]]
                    
                    for k in range(0,3):
                        #self.fhd.write("a(i,j) %d %d %d %d %d\n" % (tmp_mcal[0], tmp_mcal[1], tmp_mcal[2], params.Mcal[j+1][k], tmp_mcal[k]))
                        if params.Mcal[j+1][k] != 0:

                            self.fhd.write("#ifdef REG_SF \n")
                            self.fhd.write("  x_%d_%d = %s * sf_%d.x_%d_%d - %s * sf_%d.x_%d_%d; \n" % (j+1, i+10, self.PA[k], m, 0, i+10,\
                            self.PC[k], m+1, 0, i+10))
                            self.fhd.write("#else \n")
                            self.fhd.write("  x_%d_%d = %s * LOCSTOREFULL(store, %d, %d, STOREDIM, STOREDIM, %d)- %s * LOCSTOREFULL(store, %d, %d, STOREDIM, STOREDIM, %d); \n" % (j+1, i+10, self.PA[k], 0, i+10, m,\
                            self.PC[k], 0, i+10, m+1))
                            self.fhd.write("#endif \n")

                            if tmp_mcal[k] != 0:                                
                                tmp_mcal[k] -= 1
                                tmp_i=params.trans[tmp_mcal[0]][tmp_mcal[1]][tmp_mcal[2]]
                                self.fhd.write("  x_%d_%d += TwoZetaInv * %f * (sd_%d.x_%d_%d - sd_%d.x_%d_%d); \n" % (j+1, i+10, params.Mcal[i+10][k], m, 0, tmp_i-1, m+1, 0, tmp_i-1))
                             

                            break
                
            self.fhd.write("#else \n")

            # save all computed values into store array in global memory
            self.fhd.write("  QUICKDouble val; \n")

            for i in range(0,10):
                for j in range(0,3):
                    tmp_mcal=[params.Mcal[i+10][0], params.Mcal[i+10][1], params.Mcal[i+10][2]]

                    for k in range(0,3):
                        if params.Mcal[j+1][k] != 0:

                            self.fhd.write("#ifdef REG_SF \n")
                            self.fhd.write("  val = %s * sf_%d.x_%d_%d - %s * sf_%d.x_%d_%d; \n" % (self.PA[k], m, 0, i+10,\
                            self.PC[k], m+1, 0, i+10))
                            self.fhd.write("#else \n")
                            self.fhd.write("  val = %s * LOCSTOREFULL(store, %d, %d, STOREDIM, STOREDIM, %d) - %s * LOCSTOREFULL(store, %d, %d, STOREDIM, STOREDIM, %d); \n" % (self.PA[k], 0, i+10, m,\
                            self.PC[k], 0, i+10, m+1))
                            self.fhd.write("#endif \n")

                            if tmp_mcal[k] != 0:
                                tmp_mcal[k] -= 1
                                tmp_i=params.trans[tmp_mcal[0]][tmp_mcal[1]][tmp_mcal[2]]
                                self.fhd.write("  val += TwoZetaInv * %f * (sd_%d.x_%d_%d - sd_%d.x_%d_%d); \n" % (params.Mcal[i+10][k], m, 0, tmp_i-1, m+1, 0, tmp_i-1))

                            self.fhd.write("  LOCSTOREFULL(store, %d, %d, STOREDIM, STOREDIM, %d) = val; \n" % (j+1, i+10, m))
                            break
            self.fhd.write("#endif \n") 
            self.fhd.write("\n } \n")

            # write definition for partial classes, note that we will not write code for global memory based implementation here
            for i in range(0,10):
                self.fhd.write("\n/* PF integral partial class - Part %d, m=%d */ \n" % (i+1, m))
                self.fhd.write("%s PFint_%d_%d::PFint_%d_%d(QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,\n\
                QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz, QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz,\n\
                QUICKDouble TwoZetaInv, QUICKDouble* store, QUICKDouble* YVerticalTemp){ \n\n" % (self.func_qualifier, m, i+1, m, i+1))

                self.fhd.write("  SDint_%d sd_%d(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|d] for m=%d \n" % (m, m, m))
                self.fhd.write("  SDint_%d sd_%d(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|d] for m=%d \n" % (m+1, m+1, m+1))
                self.fhd.write("  SFint_%d sf_%d(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|f] for m=%d \n" % (m, m, m))
                self.fhd.write("  SFint_%d sf_%d(PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); // construct [s|f] for m=%d \n\n" % (m+1, m+1, m+1))

                # save all computed values into class variables that will reside in register/lmem space
                for j in range(0,3):
                    tmp_mcal=[params.Mcal[i+10][0], params.Mcal[i+10][1], params.Mcal[i+10][2]]
                    
                    for k in range(0,3):
                        #self.fhd.write("a(i,j) %d %d %d %d %d\n" % (tmp_mcal[0], tmp_mcal[1], tmp_mcal[2], params.Mcal[j+1][k], tmp_mcal[k]))
                        if params.Mcal[j+1][k] != 0:

                            self.fhd.write("#ifdef REG_SF \n")
                            self.fhd.write("  x_%d_%d = %s * sf_%d.x_%d_%d - %s * sf_%d.x_%d_%d; \n" % (j+1, i+10, self.PA[k], m, 0, i+10,\
                            self.PC[k], m+1, 0, i+10))
                            self.fhd.write("#else \n")
                            self.fhd.write("  x_%d_%d = %s * LOCSTOREFULL(store, %d, %d, STOREDIM, STOREDIM, %d)- %s * LOCSTOREFULL(store, %d, %d, STOREDIM, STOREDIM, %d); \n" % (j+1, i+10, self.PA[k], 0, i+10, m,\
                            self.PC[k], 0, i+10, m+1))
                            self.fhd.write("#endif \n")

                            if tmp_mcal[k] != 0:                                
                                tmp_mcal[k] -= 1
                                tmp_i=params.trans[tmp_mcal[0]][tmp_mcal[1]][tmp_mcal[2]]
                                self.fhd.write("  x_%d_%d += TwoZetaInv * %f * (sd_%d.x_%d_%d - sd_%d.x_%d_%d); \n" % (j+1, i+10, params.Mcal[i+10][k], m, 0, tmp_i-1, m+1, 0, tmp_i-1))
                            break    
                self.fhd.write("\n } \n")

    # generate code to save computed [p|f] integral
    def save_int(self):
        self.fha.write("\n  // PF integral, m=%d  \n" % (0))
        self.fha.write("  if(I == 1 && J == 3){ \n")
        self.fha.write("    PFint_0 pf(PAx, PAy, PAz, PBx, PBy, PBz, PCx, PCy, PCz, TwoZetaInv, store, YVerticalTemp); \n")

        self.fha.write("#ifdef REG_PF \n")
        for i in range(0,10):
            for j in range(0,3):
                self.fha.write("    LOCSTORE(store, %d, %d, STOREDIM, STOREDIM) = pf.x_%d_%d;\n" % (j+1, i+10, j+1, i+10))
        self.fha.write("#endif \n")

        # include print statements if debug option is on    
        if OEint.debug == 1:
            self.fha.write("\n#ifdef DEBUG_OEI \n")
            for i in range(0,10):
                for j in range(0,3):
                    self.fha.write("    printf(\"II %%d JJ %%d %s store[%d,%d] = %%f \\n\", II, JJ, LOCSTORE(store, %d, %d, STOREDIM, STOREDIM)); \n" % ( "PF", j+1, i+10, j+1, i+10))
            self.fha.write("#endif \n\n")

        self.fha.write("  } \n")

