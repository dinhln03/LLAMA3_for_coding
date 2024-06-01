#!/usr/bin/python

from __future__ import division
import sys
import math
import cmath
import numpy as np
from numpy import genfromtxt
import csv
from decimal import Decimal
import os
import random
from lyrics import *

# BEATLES: Bundle of Essential and Assistive Tools Library for Electronic Structure
#          A tribute to the Beatles
#
#          Updated June 14, 2020  by Hassan Harb
#
#          /     |    \
#         /      |     \
#        /O   O  | O   O\
#       //|\ /|\  /|\ /|\\              
#      /=/ \=/ \= / \=/ \=\
#     / ==  ==  ==  ==  == \
#    / ==   ==  ==  ==   == \ 
#     (The original Beatles)
# (ASCII retrieved from https://www.asciiart.eu/music/musicians/beatles )
#
#########################################################################
#
# NBasGrab: reads in a name of .fchk file
# output:  -Number of basis functions
#          -Charge
#          -Multiplicity
#          -Number of Atoms
#          -Cartesian Coordinates
#          -Atomic Symbols
#          -SCF Energy 
#          -Total Energy (needs to be added)

# Section 1: Reading from gaussian formatted checkpoint file

def NBasGrab(filename):
  NBasis = 0  
  NElem = 0
  SCFEnergy = 0.0
  Charge = 0
  Multiplicity = 0
  NAtoms = 0
  temp = 1
  with open(filename, 'r') as origin:
     for line in origin:
        if "Number of basis functions" in line:
           words = line.split()
           for i in words:
              for letter in i:
                 if(letter.isdigit()):
                    NBasis = NBasis*10 + int(letter)
        if "Charge   " in line:
           words = line.split()
           for i in words:
              for letter in i:
                 if(letter=="-"):
                   temp = -1
                 if(letter.isdigit()):
                    Charge = Charge*10 + int(letter)
              Charge = Charge*temp
        if "Multiplicity" in line:
           words = line.split()
           for i in words:
              for letter in i:
                 if(letter.isdigit()):
                    Multiplicity = Multiplicity*10 + int(letter)
        if "Number of atoms" in line:
           words = line.split()
           for i in words:
              for letter in i:
                 if(letter.isdigit()):
                    NAtoms = NAtoms*10 + int(letter)

        if "SCF Energy" in line:
            words = line.split()
#            print "SCF Energy = ", words[3], " Hartree"
            SCFEnergy = float(words[3])
#            print "SCF Energy (float) = ", SCFEnergy

#        if "Total Energy" in line:
#            words = line.split()
#            TotalEnergy = float(words[3])
#            print "Total Energy = ", TotalEnergy, " Hartree"

  NElem = NBasis*NBasis
#  print "Number of Basis Functions (subroutine) = ", NBasis, "\n"
#  print "Charge (subroutine) = ", Charge, "\n"
  return NBasis, NElem, Charge, Multiplicity, NAtoms, SCFEnergy

# GeomGet: reads in the file name, number of atoms
# Output: -One dimensional vector (NAtoms * 3) that includes the cartesian coordinates of each atom
#
 
def GeomGet(filename,NAtoms):
   p = 0
   r = 0
   n = 1
   NElements = NAtoms * 3 
   RawCart = np.zeros(NElements)
   if (NElements%5 == 0):
      n = 0
   RawCartLines = int(NElements/5) + n
#   print "Raw Cart lines = ", RawCartLines
#   print "Number of Atoms =", NAtoms
#   print "Number of coordinates =", NElements
   with open(filename,'r') as origin:
      for i, line in enumerate(origin):
         if "Current cartesian coordinates" in line:
            i = i + 1
            pointer = i
#            print "Cartesian Coordinates starts at line :", pointer
            endpointer = pointer + RawCartLines - 1
#            print "Cartesian Coordinates ends at line :", endpointer
            for m in range(0,endpointer - pointer +1):
               nextline = origin.next()
               nextline = nextline.split()
               for p in range(p,len(nextline)):
                  RawCart[r] = nextline[p]
                  r = r + 1
               p = 0
#   print "Raw Cart (subroutine) = ", RawCart
   RawCart = RawCart/1.88973
#   print "Raw Cart (converted to Angstroms) = ", RawCart
   return  RawCart

# GetAtoms:  Reads in file name, number of atoms
# output:   -One dimensional vector (NAtoms) that contains the atomic numbers of the atoms 
#

def GetAtoms(filename1,NAtoms):
   p = 0
   r = 0
   n = 1
   AtomicNum = np.zeros(NAtoms)
   if (NAtoms%6 ==0):
     n = 0
   AtomLines = int(NAtoms/6) + n

   with open(filename1,'r') as origin:
      for i, line in enumerate(origin):
         if "Atomic numbers" in line:
            i = i + 1
            pointer = i
            endpointer = pointer + AtomLines -1
            for m in range(0, endpointer - pointer + 1):
                nextline = origin.next()
                nextline = nextline.split()
                for p in range(p,len(nextline)):
                   AtomicNum[r] = nextline[p]
                   r = r + 1
                p = 0
   return AtomicNum

# MatGrab: Reads in filename, NBasis, user-defined switch
# Output: -Alpha MO Coefficients (Done)
#         -Beta MO Coefficients (Done)
#         -Alpha Density Matrix (Done)
#         -Beta Density Matrix (Done)
#         -Alpha MO Energies (Done)
#         -Beta MO Energies (Done)
#
# Switch: 1 = Alpha MO Coefficients
#        -1 = Beta MO Coefficients
#         2 = Alpha and Beta Density Matrices
#         3 = Alpha MO Energies
#        -3 = Beta MO Energies
#
def MatGrab(filename,NBasis,switch):
   if (switch == 1):
      filename1 = filename
      MOElements = NBasis * NBasis
      MOlines = int(MOElements/5) + 1
      if (NBasis%5 == 0):
         MOlines = MOlines - 1
      p = 0
      r = 0
      AOE = 0
      MOrawa = np.zeros(NBasis*NBasis)
      with open(filename1,'r') as origin:
          for i, line  in enumerate(origin):
              if "Alpha Orbital Energies" in line:
                    AOE = i
              if  "Alpha MO coefficients" in line:
                    i=i+1
                    AMO=i
#                    print "Alpha MO coefficients starts at line :", i
                    j=i+MOlines-1
#                    print "Alpha MO coefficients ends at line :", j
                    for m in range(0,j-i+1):
                       nextline = origin.next()
                       nextline = nextline.split()
                       for p in range(p,len(nextline)):
                          MOrawa[r] = nextline[p]
                          r = r+1
                       p = 0
#      print "MO Raw = ", MOrawa
      return MOrawa

   if (switch == -1):
      filename1 = filename
      MOElements = NBasis * NBasis
      MOlines = int(MOElements/5) + 1
      if (NBasis%5 == 0):
         MOlines = MOlines - 1
      p = 0
      r = 0
      BOE = 0
      BMO = 0
      MOrawb = np.zeros(NBasis*NBasis)
      with open(filename1,'r') as origin:
         for i, line  in enumerate(origin):
              if "Beta Orbital Energies" in line:
                    BOE = i
              if  "Beta MO coefficients" in line:
                    i=i+1
                    BMO=i
                    j=i+MOlines-1
                    for m in range(0,j-i+1):
                       nextline = origin.next()
                       nextline = nextline.split()
                       for p in range(p,len(nextline)):
                          MOrawb[r] = nextline[p]
                          r = r+1
                       p = 0

#         print "MO Raw = ", MOrawb
         return MOrawb

   if (switch == 2):
      filename1 = filename
      PElements = int(NBasis*(NBasis+1)/2)
      Plines = int(PElements/5) + 1
      TotalPraw = np.zeros(PElements)
      SpinPraw = np.zeros(PElements)

      with open(filename1,'r') as origin:
       for i, line in enumerate(origin):
          if  "Total SCF Density" in line:
              i=i+1
              r = 0
              p = 0
#              print "Total SCF Density starts at line :", i
              j=i+Plines-1
#              print "Total SCF Density ends at line :", j
              for m in range(0,j-i+1):
                 nextline = origin.next()
                 nextline = nextline.split()
                 for p in range(0,len(nextline)):
                   if (r != PElements):
                       TotalPraw[r] = nextline[p]
                       r = r+1
                 p = 0
# HH + : Bug ... :(
      with open(filename1,'r') as origin:
       for i, line in enumerate(origin):
          if  "Spin SCF Density" in line:
#              print "Found Spin density!"
              i=i+1
              r = 0
              p = 0
#              print "Spin SCF Density starts at line: ", i
              j=i+Plines-1
#              print "Spin SCF Density ends at line: ", j
              for m in range(0,j-i+1):
                 nextline = origin.next()
                 nextline = nextline.split()
                 for p in range(p,len(nextline)):
                     if (r != PElements):
                       SpinPraw[r] = nextline[p]
                       r = r+1
                 p = 0
# HH - : End of bug (hopefully!)

      PalphaRaw = (np.add(TotalPraw,SpinPraw)) * 0.5
      PbetaRaw = (np.subtract(TotalPraw,SpinPraw)) * 0.5
      Palpha = symmetrize(PalphaRaw)
      Pbeta  = symmetrize(PbetaRaw)       
      return Palpha, Pbeta

   if (switch == 3):
      filename1 = filename
      AlphaMO = np.zeros(NBasis)
      AlphaMOlines = int(NBasis/5) + 1
      if (NBasis % 5 == 0):
         AlphaMOlines = AlphaMOlines - 1
      with open(filename1,'r') as origin:
         for i, line in enumerate(origin):
             if "Alpha Orbital Energies" in line:
                i = i + 1
                r = 0
                p = 0
#                print "Alpha MO Energies starts at line: ", i
                j = i + AlphaMOlines - 1
#                print "Alpha MO Energies ends at line: ", j
                for m in range(0,j-i+1):
                    nextline = origin.next()
                    nextline = nextline.split()
                    for p in range(p,len(nextline)):
                        AlphaMO[r] = nextline[p]
                        r = r + 1
                    p = 0
#      print "Alpha MO energies = ", AlphaMO
      return AlphaMO

   if (switch == -3):
      filename1 = filename
      BetaMO = np.zeros(NBasis)
      BetaMOlines = int(NBasis/5) + 1
      if (NBasis % 5 == 0):
         BetaMOlines = BetaMOlines - 1
      with open(filename1,'r') as origin:
         for i, line in enumerate(origin):
             if "Beta Orbital Energies" in line:
                i = i + 1
                r = 0
                p = 0
#                print "Beta MO Energies starts at line: ", i
                j = i + BetaMOlines - 1
#                print "Beta MO Energies ends at line: ", j
                for m in range(0,j-i+1):
                    nextline = origin.next()
                    nextline = nextline.split()
                    for p in range(p,len(nextline)):
                        BetaMO[r] = nextline[p]
                        r = r + 1
                    p = 0
#      print "Beta MO energies = ", BetaMO
      return BetaMO

# sci_notation:  reads in a number
# output:        prints the number in the desired scientific notation. note that this function has a different output than the one found in nio.py
#
def sci_notation(n):
    a = '%.8f' % n
    return '%.8f' % Decimal(n.real)

# fchk_notation: reads in a number
# output:        prints the number in the desired notation for fchk files
#
def fchk_notation(n):
    a = '%.8E' % n
    return '%.8E' % Decimal(n.real)

# AtomicSymbol:  Reads in atomic number of the element
# Output:       -Atomic Symbol
# 

def AtomicSymbol(AtomicNumber):
    p = AtomicNumber - 1
    PTlist = ['H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P','S','Cl','Ar','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr','Rb','Sr','Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ah','Cd','In','Sn','Sb','Te','I','Xe','Cs','Ba','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Hf','Ta','W','Re','Os','Ir','Pt','Au','Hb','Tl','Pb','Bi','Po','At','Rn','Fr','Ra','Ac','Th','Pa','U','Np','Pu','Am','Cm','Bk','Cf','Es','Fm','Md','No','Lr','Rf','Db','Sg','Bh','Hs','Mt','Ds','Rg','Cn','Uut','Fl','Uup','Lv','Uus','Uuo']
#    print "There are currently ", len(PTlist), " atoms defined"
    return PTlist[p]

# Symmetrize:  Reads in a packed symmetric column matrix into NBasis x NBasis square matrix 
# Output:     -Matrix(NBasis,NBasis)
#

def symmetrize(a):
  Nbas = int((np.sqrt(8*len(a)+1)-1)/2)
  b = np.zeros((Nbas,Nbas))
  n = 0
  for i in range(0,Nbas):
    for j in range(0,i+1):
      b[i,j]=a[n]
      b[j,i]=a[n]
      n=n+1
  return b

# Column2Square:  Reads in a packed column matrix, number of basis functions.
# Output:        -Matrix(NBasis,NBasis)

def column2square(A,NBasis):
  C = np.zeros((NBasis,NBasis))
  t=0
  for i in range(0,NBasis):
     for j in range(0,NBasis):
       C[j,i]=float(A[t])
       t=t+1
  return C

# GetOverlap:  Reads in packed column matrix, number of basis functions.
# Output:     -Overlap Matrix (NBasis,NBasis)

def GetOverlap(A,NBasis):
   C = column2square(A,NBasis)
   CInv = np.linalg.inv(C)
   S = np.dot(np.transpose(CInv),CInv)
   return S  

# PrintSI: Reads in filename, user-defined switch
# Output: -SCF Energy, Charge, Multiplicity, Geometry
#
# Switch:  1 = print to new file (filename1-SI.txt)
#         -1 = print to screen
#

def PrintSI(filename1,switch):
    NBasis, NElementsGrab, Charge, Multiplicity, NAtoms, SCFEnergy  = NBasGrab(filename1)
    AtomicNum = GetAtoms(filename1,NAtoms)
    RawCart = GeomGet(filename1,NAtoms)
    Cart = np.resize(RawCart,(NAtoms,3))
    filename2 = os.path.splitext(filename1)[0] + "-SI.txt"
    filename1 = os.path.splitext(filename1)[0] 
    if (switch == 1):
       with open(filename2,'w') as f2:
         f2.write("SI info for ")
         f2.write(filename1)
         f2.write("\n\n")
         f2.write("SCF Energy = ")
         f2.write(str(SCFEnergy))
         f2.write(" Hartree")
         f2.write("\n\n")
         f2.write(str(Charge))
         f2.write(" ")
         f2.write(str(Multiplicity))
         f2.write("\n")
         for i in range(0,NAtoms):
             h = i + 1
             z = AtomicNum[i]
             Atom  = AtomicSymbol(int(z))
             f2.write(Atom)
             f2.write(" ")
             for j in range(0,3):
                 if (Cart[i,j] >= 0):
                    f2.write(" ")
                 f2.write(str(sci_notation(Cart[i,j])))
                 f2.write(" ")
             f2.write("\n")
         f2.write(" ")
         f2.write("\n\n")
    return filename2
    if (switch == -1):
      print "SCF Energy = ", SCFEnergy, " Hartree\n"
      print "Charge = ", Charge, "\n"
      print "Multiplicity = ", Multiplicity, "\n"
      print "Cartesian Geometry:\n"
      for i in range(0,NAtoms):
          h = i + 1
          z = AtomicNum[i]
          Atom = AtomicSymbol(int(z))
          print Atom, sci_notation(Cart[i,0]), sci_notation(Cart[i,1]), sci_notation(Cart[i,2])
      print "\n"    
      
# CalcNO: Reads in filename, NBasis
# Output: Natural Orbitals eigenvalues and eigenvectors (both alpha and beta)
# 

def CalcNO(filename,NBasis):
       Palpha, Pbeta = MatGrab(filename,NBasis,2)   
       C = MatGrab(filename,NBasis,1)
       S = GetOverlap(C,NBasis)
       Svals, Svecs = np.linalg.eig(S)
       Sval_minhalf = (np.diag(Svals**(0.5)))
       Shalf = np.dot(Svecs,np.dot(Sval_minhalf,np.transpose(Svecs)))
       NOvalsA, NOvecsA = np.linalg.eig(np.dot(Shalf,np.dot(Shalf,Palpha)))
       NOvalsB, NOvecsB = np.linalg.eig(np.dot(Shalf,np.dot(Shalf,Pbeta))) 
       NOvalsA = NOvalsA.real
       NOvalsB = NOvalsB.real 
       NOvecsA = NOvecsA.real
       NOvecsB = NOvecsB.real
       NOvecsA = np.dot(np.linalg.inv(Shalf),NOvecsA)
       NOvecsB = np.dot(np.linalg.inv(Shalf),NOvecsB)
       return NOvecsA, NOvecsB, NOvalsA, NOvalsB

# NElec: Reads in filename
# Output: Total number of electrons, Alpha Electrons, Beta Electrons
#

def NElec(filename):
  NElec = 0
  NAlpha = 0
  NBeta = 0
  with open(filename, 'r') as origin:
     for line in origin:
        if "Number of electrons" in line:
           words = line.split()
           for i in words:
              for letter in i:
                 if(letter.isdigit()):
                    NElec = NElec*10 + int(letter)
        if "Number of alpha electrons" in line:
           words = line.split()
           for i in words:
              for letter in i:
                 if(letter.isdigit()):
                    NAlpha = NAlpha*10 + int(letter)
        if "Number of beta electrons" in line:
           words = line.split()
           for i in words:
              for letter in i:
                 if(letter.isdigit()):
                    NBeta = NBeta*10 + int(letter)
  return NElec, NAlpha, NBeta

# OrbTransform: Reads in Alpha Density Matrix, Beta Density Matrix, Overlap Matrix, n
# Output: New Density Matrices: P' = S**(1-n).P.S**(n)
#

def OrbTransform(Pa,Pb,S,n):
       Svals, Svecs = np.linalg.eig(S)
       Sval1 = np.diag(Svals**(n))
       Sval2 = np.diag(Svals**(1-n)) 
       Sdag1 = np.dot(Svecs,np.dot(Sval1,np.transpose(Svecs)))
       Sdag2 = np.dot(Svecs,np.dot(Sval2,np.transpose(Svecs)))
       PdagAlpha = np.dot(Sdag1,np.dot(Pa,Sdag2))
       PdagBeta = np.dot(Sdag1,np.dot(Pb,Sdag2))
#       print "OrbTransform Subroutine test:\n"
#       print "PdagAlpha = ", PdagAlpha, "\n"
#       print "PdagBeta = ", PdagBeta, "\n"
       OvalsA, OvecsA = np.linalg.eig(PdagAlpha)
       OvalsB, OvecsB = np.linalg.eig(PdagBeta) 
#       print "OVals A = ", OvalsA, "\n"
#       print "OVecs A = ", OvecsA, "\n"
#       print "OVals B = ", OvalsB, "\n"
#       print "OVecs B = ", OvecsB, "\n"
       return PdagAlpha, PdagBeta, OvecsA, OvecsB, OvalsA, OvalsB

# CartoZmat: Transforms Cartesian coordinates to z-matrix form
# Input: NAtoms, RawCart, AtomicNum
# Output: z-matrix printed on the screen
#

# Note that there are three other functions here, Dist, Angle, and Torsion. 
# They are used to calculate the appropriate parameters for the z-matrix
# switch =  1 : print z-matrix to screen
# switch = -1 : print z-matrix to new textfile   

def DistAB(e1,e2):
    R = 0.0
    for i in range(len(e1)):
      R = R + (e1[i]-e2[i])**(2)
    R = R**(0.5) 
    return R

def AngleABC(e1,e2,e3):
    eab_x = (e2[0] - e1[0]) / DistAB(e1,e2)
    eab_y = (e2[1] - e1[1]) / DistAB(e1,e2)
    eab_z = (e2[2] - e1[2]) / DistAB(e1,e2)

    ebc_x = - (e3[0] - e2[0]) / DistAB(e2,e3)
    ebc_y = - (e3[1] - e2[1]) / DistAB(e2,e3)
    ebc_z = - (e3[2] - e2[2]) / DistAB(e2,e3)

    eab = [eab_x, eab_y, eab_z]
    ebc = [ebc_x, ebc_y, ebc_z]
   
    cos_angle = np.dot(eab,ebc)
    angle = np.arccos(cos_angle) / 3.1415926535 * 180
    return eab, ebc, angle    

def TorsionABCD(e1,e2,e3,e4):

    eab_x = (e2[0] - e1[0]) / DistAB(e1,e2)
    eab_y = (e2[1] - e1[1]) / DistAB(e1,e2)
    eab_z = (e2[2] - e1[2]) / DistAB(e1,e2)    

    ebc_x =  (e3[0] - e2[0]) / DistAB(e2,e3)
    ebc_y =  (e3[1] - e2[1]) / DistAB(e2,e3)
    ebc_z =  (e3[2] - e2[2]) / DistAB(e2,e3)

    ecd_x = (e4[0] - e3[0]) / DistAB(e3,e4)
    ecd_y = (e4[1] - e3[1]) / DistAB(e3,e4)
    ecd_z = (e4[2] - e3[2]) / DistAB(e3,e4)

    eab = [eab_x, eab_y, eab_z]
    ebc = [ebc_x, ebc_y, ebc_z]
    ecd = [ecd_x, ecd_y, ecd_z]

    n1 = np.cross(eab,ebc) / (np.linalg.norm(np.cross(eab,ebc))) 
    n2 = np.cross(ebc,ecd) / (np.linalg.norm(np.cross(ebc,ecd)))

    u1 = n2
    u3 = ebc/np.linalg.norm(ebc)
    u2 = np.cross(u3,u1)

    cos_angle = np.dot(n1,n2)
    sin_angle = np.dot(n1,u2)
    
    angle = -math.atan2(sin_angle,cos_angle) / 3.1415926535 * 180
    return angle

def CartoZmat(RawCart,NAtoms,AtomicNum,filename2,switch):
  if (switch == 1):
    Cart = np.resize(RawCart,(NAtoms,3))
#    print "Cartesian = ", Cart 
#    print "Atoms list = ", AtomicNum
    for i in range(len(AtomicNum)):
        Symbol = AtomicSymbol(int(AtomicNum[i]))
        if (i > 2):
           e4 = [Cart[i,0],Cart[i,1],Cart[i,2]]
           e3 = [Cart[2,0],Cart[2,1],Cart[2,2]]
           e2 = [Cart[1,0],Cart[1,1],Cart[1,2]]
           e1 = [Cart[0,0],Cart[0,1],Cart[0,2]]
           R = DistAB(e4,e1)
           eab, ebc, A = AngleABC(e2,e1,e4)
           D = TorsionABCD(e4,e1,e2,e3)
           print Symbol, 1 ,  R , 2,  A , 3,  D  
        elif (i > 1):
           e4 = [Cart[i,0],Cart[i,1],Cart[i,2]]
           e2 = [Cart[1,0],Cart[1,1],Cart[1,2]]
           e1 = [Cart[0,0],Cart[0,1],Cart[0,2]]
           R = DistAB(e4,e1)
           eab, ebc, A = AngleABC(e2,e1,e4)
           print Symbol, 1 , R , 2, A
        elif (i > 0):
           e4 = [Cart[i,0],Cart[i,1],Cart[i,2]]
           e1 = [Cart[0,0],Cart[0,1],Cart[0,2]]
           R = DistAB(e4,e1)
           print Symbol, 1, R 
        elif (i == 0):
           print Symbol
  elif (switch == -1):
    Cart = np.resize(RawCart,(NAtoms,3))
    #open new file
    filename =  os.path.splitext(filename2)[0] + "-zmat.txt"
    with open(filename,'w') as f2:
      NBasis, NElem, Charge, Multiplicity, NAtoms, SCFEnergy =  NBasGrab(filename2)
      f2.write("Z-Matrix file for ")
      f2.write(filename2)
      f2.write("\n\n")
      f2.write(str(Charge))
      f2.write(" ")
      f2.write(str(Multiplicity))
      f2.write("\n")
      for i in range(len(AtomicNum)):
         Symbol = AtomicSymbol(int(AtomicNum[i]))
         if (i > 2):
           e4 = [Cart[i,0],Cart[i,1],Cart[i,2]]
           e3 = [Cart[2,0],Cart[2,1],Cart[2,2]]
           e2 = [Cart[1,0],Cart[1,1],Cart[1,2]]
           e1 = [Cart[0,0],Cart[0,1],Cart[0,2]]
           R = DistAB(e4,e1)
           eab, ebc, A = AngleABC(e2,e1,e4)
           D = TorsionABCD(e4,e1,e2,e3)
           f2.write(Symbol)
           f2.write(" 1 ") 
           f2.write(str(R)) 
           f2.write(" 2 ") 
           f2.write( str(A))  
           f2.write(" 3 ")  
           f2.write(str(D))
           f2.write("\n")           
         elif (i > 1):
           e4 = [Cart[i,0],Cart[i,1],Cart[i,2]]
           e2 = [Cart[1,0],Cart[1,1],Cart[1,2]]
           e1 = [Cart[0,0],Cart[0,1],Cart[0,2]]
           R = DistAB(e4,e1)
           eab, ebc, A = AngleABC(e2,e1,e4)
           f2.write(str(Symbol)) 
           f2.write(" 1 ")
           f2.write (str(R)) 
           f2.write(" 2 ") 
           f2.write(str(A)) 
           f2.write("\n")
         elif (i > 0):
           e4 = [Cart[i,0],Cart[i,1],Cart[i,2]]
           e1 = [Cart[0,0],Cart[0,1],Cart[0,2]]
           R = DistAB(e4,e1)
           f2.write(Symbol)  
           f2.write(" 1 ")
           f2.write(str(R))
           f2.write("\n")
         elif (i == 0):
           f2.write(Symbol)
           f2.write("\n")           
#    print "test test"

# Section 2: Reading from gaussian matrix files

# MatGrab2: Reads in matrices from gaussian matrix file
# 
# Switch:  1 : Alpha Core Hamiltonian
#         -1 : Beta Core Hamiltonian
#          2 : Alpha Fock Matrix
#         -2 : Beta Fock Matrix
#          3 : Dipole matrix elements (x,y,z) [IN PROGRESS]

def MatGrab2(filename,NBasis,switch):
    print "Reading from Matrix file\n"
    if (switch == 1):
        print "Reading Alpha Core Hamiltonian Matrix:\n"
        NElements = int(NBasis*(NBasis + 1)/2)
        print "Looking for ", NElements, " elements of the core hamilonian\n"
        CoreHRawa = np.zeros(NElements)
        p = 0
        n = 0 
        r = 0
        with open(filename,'r') as origin:
             for i, line in enumerate(origin):
                if "CORE HAMILTONIAN ALPHA" in line :
                   while (p < (NElements)):
                     NLines = NBasis - 5*r
                     if (NLines < 0):
                        print "Done Reading Core Hamolitonian"
                     j = i+3
                     i = i + 4
                     end = j + NLines - 1
                     nextline = origin.next()
                     for m in range(i,i+NLines):
                         nextline = origin.next()
                         words = nextline.split()
                         for j in range(1,len(words)):
                             CoreHRawa[p] = float(words[j].replace('D','E'))
                             p = p + 1
                     r = r + 1
                     i = m - 2
        return CoreHRawa
    if (switch == -1):
        print "Reading Beta Core Hamiltonian Matrix:\n"
        NElements = int(NBasis*(NBasis + 1)/2)
        print "Looking for ", NElements, " elements of the core hamilonian\n"
        CoreHRawb = np.zeros(NElements)
        p = 0
        n = 0
        r = 0
        with open(filename,'r') as origin:
             for i, line in enumerate(origin):
                if "CORE HAMILTONIAN BETA" in line :
                   while (p < (NElements)):
                     NLines = NBasis - 5*r
                     if (NLines < 0):
                        print "Done Reading Core Hamolitonian"
                     j = i+3
                     i = i + 4
                     end = j + NLines - 1
                     nextline = origin.next()
                     for m in range(i,i+NLines):
                         nextline = origin.next()
                         words = nextline.split()
                         for j in range(1,len(words)):
                             CoreHRawb[p] = float(words[j].replace('D','E'))
                             p = p + 1
                     r = r + 1
                     i = m - 2
        return CoreHRawb

    if (switch == 2):
        print "Reading Alpha Fock Matrix:\n"
        NElements = int(NBasis*(NBasis + 1)/2)
        print "Looking for ", NElements, " elements of the fock matrix\n"
        FockRawA = np.zeros(NElements)
        p = 0
        n = 0
        r = 0
        with open(filename,'r') as origin:
             for i, line in enumerate(origin):
                if "ALPHA FOCK MATRIX" in line :
                   while (p < (NElements)):
                     NLines = NBasis - 5*r
                     if (NLines < 0):
                        print "Done Reading fock matrix"
                     j = i+3
                     i = i + 4
                     end = j + NLines - 1
                     nextline = origin.next()
                     for m in range(i,i+NLines):
                         nextline = origin.next()
                         words = nextline.split()
                         for j in range(1,len(words)):
                             FockRawA[p] = float(words[j].replace('D','E'))
                             p = p + 1
                     r = r + 1
                     i = m - 2
        return FockRawA

    if (switch == -2):
        print "Reading Beta Fock Matrix:\n"
        NElements = int(NBasis*(NBasis + 1)/2)
        print "Looking for ", NElements, " elements of the fock matrix\n"
        FockRawB = np.zeros(NElements)
        p = 0
        n = 0
        r = 0
        with open(filename,'r') as origin:
             for i, line in enumerate(origin):
                if "BETA FOCK MATRIX" in line :
                   while (p < (NElements)):
                     NLines = NBasis - 5*r
                     if (NLines < 0):
                        print "Done Reading fock matrix"
                     j = i+3
                     i = i + 4
                     end = j + NLines - 1
                     nextline = origin.next()
                     for m in range(i,i+NLines):
                         nextline = origin.next()
                         words = nextline.split()
                         for j in range(1,len(words)):
                             FockRawB[p] = float(words[j].replace('D','E'))
                             p = p + 1
                     r = r + 1
                     i = m - 2
        return FockRawB

    if (switch == 3):
#       print "Reading Dipole integrals, matrix x\n"
       NElements = int(NBasis*(NBasis +1)/2)
#       print "Looking for ", NElements, " elements of the Dipole integrals matrix x\n"
       DipX_Raw = np.zeros(NElements)
       p = 0
       n = 0
       r = 0
       with open(filename,'r') as origin:
            for i, line in enumerate(origin):
                if " DIPOLE INTEGRALS, matrix     1" in line:
                   while (p < NElements):
                     NLines = NBasis - 5*r
                     if (NLines < 0):
#                        print "Done reading Dipole X matrix\n"
                     j = i+3
                     i = i + 4
                     end = j + NLines -1
                     nextline = origin.next()
                     words = nextline.split()
                     for m in range(i,i+NLines):
                         nextline = origin.next()
                         words = nextline.split()
                         for j in range(1,len(words)):
                             DipX_Raw[p] = float(words[j].replace('D','E'))
                             p = p + 1
                     r = r + 1
                     i = m - 2
#       print "Dip X raw = ", DipX_Raw

#       print "Reading Dipole integrals, matrix y\n"
       NElements = int(NBasis*(NBasis +1)/2)
       print "Looking for ", NElements, " elements of the Dipole integrals matrix y\n"
       DipY_Raw = np.zeros(NElements)
       p = 0
       n = 0
       r = 0
       with open(filename,'r') as origin:
            for i, line in enumerate(origin):
                if " DIPOLE INTEGRALS, matrix     2" in line:
                   while (p < NElements):
                     NLines = NBasis - 5*r
                     if (NLines < 0):
#                        print "Done reading Dipole Y matrix\n"
                     j = i+3
                     i = i + 4
                     end = j + NLines -1
                     nextline = origin.next()
                     words = nextline.split()
                     for m in range(i,i+NLines):
                         nextline = origin.next()
                         words = nextline.split()
                         for j in range(1,len(words)):
                             DipY_Raw[p] = float(words[j].replace('D','E'))
                             p = p + 1
                     r = r + 1
                     i = m - 2
#       print "Dip Y raw = ", DipY_Raw                  

 #      print "Looking for ", NElements, " elements of the Dipole integrals matrix z\n"
       DipZ_Raw = np.zeros(NElements)
       p = 0
       n = 0
       r = 0
       with open(filename,'r') as origin:
            for i, line in enumerate(origin):
                if " DIPOLE INTEGRALS, matrix     3" in line:
                   while (p < NElements):
                     NLines = NBasis - 5*r
                     if (NLines < 0):
                        print "Done reading Dipole Z matrix\n"
                     j = i+3
                     i = i + 4
                     end = j + NLines -1
                     nextline = origin.next()
                     words = nextline.split()
                     for m in range(i,i+NLines):
                         nextline = origin.next()
                         words = nextline.split()
                         for j in range(1,len(words)):
                             DipZ_Raw[p] = float(words[j].replace('D','E'))
                             p = p + 1
                     r = r + 1
                     i = m - 2
 #      print "Dip Z raw = ", DipZ_Raw
       return symmetrizeMat(DipX_Raw), symmetrizeMat(DipY_Raw), symmetrizeMat(DipZ_Raw)





# SymmetrizeMat: Reads in packed matrix (recovered from Matrix file) and prints out NBasis x NBasis matrix
# Input: Packed lower triangular A
# Output: N x N Matrix

def symmetrizeMat(a):
   NBasis = int((np.sqrt(8*len(a)+1)-1)/2)
   NewMat = np.zeros((NBasis,NBasis))
   NElements = len(a)
   t = 0
   l = 0
   start = 0
   loop = NBasis
   nBlock = int(NBasis/5)
   nRem = NBasis%5
#   print "nBlock = ", nBlock
#   print "nRem = ", nRem
   i = start
   j = start
   if (nBlock == 0):
      nBlock =1

   while (l < nBlock):
#       print "retrieving block ", l
       for i in range (start,loop):
              for j in range(start,start+5):
                  if (j<=i):
#                    print "i,j = ",i,j
                    NewMat[i,j] = a[t]
                    NewMat[j,i] = a[t]
#                    print "A[t]= ", a[t]
                    t = t + 1
       start = start + 5
       l = l + 1
#   print "t = ", t
#   print "values of i and j after nBlock loop is over: ", i, j
   j = j + 1
   start = j
#   print "NBasis - nRem = ", NBasis -nRem
   i = NBasis - nRem
   while (i < NBasis):
       j = start
       while (j <= i):
#         print "i,j = ",i,j
         NewMat[i,j] = a[t]
         NewMat[j,i] = a[t]
#         print "A[t]= ", a[t]
         t = t + 1
         j = j + 1
       i = i + 1
#   print "final value of t = ", t
   return NewMat

# ERIRead: reads in regular 2e integrals from formatted matrix file
# Note that to get these integrals, use SCF=Conventional and int=NoRaff (saves integrals to disk and prints out regular 2e integrals)
# Input: matrix filename
# Output: 2D Matrix, two columns: Column 1 = compound index, Column 2 = integral value
# 
# Two small functions are defined here: swap(a,b) and Fourindex(a,b,c,d)


def swap(a,b):
   return b,a

def Fourindex(a,b,c,d):
   a = int(a)
   b = int(b)
   c = int(c)
   d = int(d)
   if (a < b):
       a, b = swap(a,b)
   if (c < d):
       c, d = swap(c,d)
   e = int(a*(a+1)/2 + b)
   f = int(c*(c+1)/2 + d)
   if (e<f):
      e,f = swap(e,f)
   g = e*(e +1)/2 + f
   return int(g)

def ERIRead(filename,NBasis):
    NElements = 0
    p = 0
    print "Reading ERIs from Gaussian Matrix File"
    print "Subroutine can only read regular 2e integrals (NO RAFINETTI)"
    with open(filename,'r') as origin:
        for i, line in enumerate(origin):
            if "Label REGULAR 2E INTEGRALS" in line:
                print "Found 2e integrals!"
                words = line.split()
                print "Total number of elements = ", words[9]
                NElements = int(words[9])
                print "NElements = ", NElements
                eri_raw = np.zeros((NElements,5))
            while (p < NElements):
                nextline = origin.next()
                words = nextline.split() 
                eri_raw[p,0] = words[1]
                eri_raw[p,1] = words[3]
                eri_raw[p,2] = words[5]
                eri_raw[p,3] = words[7]
                eri_raw[p,4] = float(words[9].replace('D','E'))
#                print "(",int(eri_raw[p,0]),int(eri_raw[p,1]),"|",int(eri_raw[p,2]),int(eri_raw[p,3]),") = ", eri_raw[p,4]
                p = p + 1
#    print "ERI RAW = ", eri_raw
    NTotal = Fourindex(NBasis,NBasis,NBasis,NBasis) + 1
    eri_array = np.zeros(NTotal)
    eri_compact = np.zeros((NElements,2))
    print "Total length of sparse 1D vector =", NTotal
    print "Now forming compound indices"
    for i in range(0,NElements):
        eri_compact[i,0] = Fourindex(eri_raw[i,0], eri_raw[i,1], eri_raw[i,2], eri_raw[i,3])
        eri_compact[i,1] = eri_raw[i,4]
        eri_array[int(eri_compact[i,0])] = eri_compact[i,1]
    #    print "mu nu lambda sigma = ", int(eri_compact[i,0]), ", int = ", eri_compact[i,1], "One D array Value =", eri_array[eri_compact[i,0]]

    return eri_array

# OVParse breaks down the MO coefficient matrix (NBasis x NBasis) into an occupied (NBasis x NOcc) and a virtual (NBasis x (Nbasis-NOcc)) matrices
# Input: A: MO Coefficient (NBasis x NBasis)
#        NBasis
#        NOcc = number of electrons
#
# Output: A_Occ: rectangular NBasis x NOcc matrix: Columns of occupied MOs
#         A_Virt: rectangular NBasis x (NBasis - NOcc) matrix: Columns of virtual MOs

##  Note TO SELF: Needs to be tested more, was only tested on H2 and V jobs.

def OVParse(A,NBasis,NOcc):

    A_Occ = np.zeros((NBasis,NOcc))
    A_Virt = np.zeros((NBasis,NBasis-NOcc))

    for i in range(0,NOcc):
        A_Occ[:,i] = A[:,i]

    for j in range(0,NBasis-NOcc):
        A_Virt[:,j] = A[:,j+NOcc]

    return A_Occ, A_Virt

# Biorthog: Calculates the overlap between two sets of MO Coefficients, prints out the final value of the overlap
# Input: A, B: MO Coefficients, can either be full or parsed (using OVParse subroutine)
#        S: AO overlap matrix
#
# Output: the final value of the overlap
# 
# Option: switch: 1 : print all relevant matrices
#                -1 : Dont print any matrices
#

def Biorthog(A,B,S,switch):                 # eqn numbers based on personal notes
   D = np.dot(np.transpose(B),np.dot(S,A))  # eq. 1
   u, d, v  = np.linalg.svd(D,full_matrices=True)              # eq. 2
    
   DtD = np.dot(np.transpose(D),D)
   l, V = np.linalg.eig(DtD)
   U = np.dot(D,V)
   if (switch==1):
      print "D = ", D
      print "DtD = ", DtD
      print "lambdas = ", l
      print "Eig Vecs of DtD = ", V
      print "Determinants = ", np.linalg.det(u), np.linalg.det(v)
      print "u = ", u
      print "v = ", v
   overlap =  np.linalg.det(u)*np.prod(d)*np.linalg.det(v)
   return d, u, v, D

# PickColumn: Subroutine that selects a specific column from a two dimensional matrix (NBasis,NBasis), outputs an array (NBasis,1)
# Input: A: Two dimensional matrix
#        NBasis: Number of basis functions for A
#        i: the position of the column to be selected
#
# Output: One dimensional array (NBasis,1) that is the i-th column of matrix A
#

def PickColumn(A,NBasis,i):
    A_Column = np.zeros((NBasis,1))
    for  j in range(0,NBasis):
        A_Column[j,0] = A[j,i]

    return A_Column

# WriteMOs: Subroutine that replaces the MO coefficients and orbital energies in a fchk file
# Input:    Input filename, output filename, Orbital coefficient alpha, orbital coefficient beta, Orbtial energies alpha, orbital energies beta, number of basis functions 
# 
# Output:   None. New file will be generated (filename3) that has the new Orbital coefficients and energies
#

def WriteMOs(filename1,filename3,V1,V2,e1,e2,NBasis):

  MOlines = int(len(V1)/5) + 1
  p = 0
  r = 0
  AOE = 0

  with open(filename1,'r') as origin:
      for i, line  in enumerate(origin):
          if "Alpha Orbital Energies" in line:
                AOE = i
          if  "Alpha MO coefficients" in line:
                i=i+1
                AMO=i
                j=i+MOlines-1
                for m in range(0,j-i+1):
                   nextline = origin.next()
                   nextline = nextline.split()
                   for p in range(p,len(nextline)):
                     r = r+1
                   p = 0
          if "Beta Orbital Energies" in line:
                BOE = i
          if "Beta MO coefficients" in line:
                r = 0
                i=i+1
                BMO = i
                j=i+MOlines-1
                for m in range(0,j-i+1):
                   nextline = origin.next()
                   nextline = nextline.split()
                   for p in range(p,len(nextline)):
                     r = r+1
                   p = 0

  pointer=0
  counter=1


  with open(filename1,'r') as origin:
    data = origin.readlines()
    if "Alpha Orbital Energies" in line:
      AOE = i  
      BOE = AOE + int(NBasis/5) + 1
    with open(filename3,'w') as f2:
  
        print "Writing results to new output file: ", filename3, " ... "
  
        while (pointer < AOE+1):
           f2.write(data[pointer])
           pointer = pointer+1
        for j in range(0,NBasis):
            f2.write(" ")
            if (e1[j] >= 0):
               f2.write(" ")
            f2.write(str(fchk_notation(e1[j].real)))
            if (counter%5 == 0):
                f2.write("\n")
                counter=0
            counter=counter+1
        counter =1      
        BOE = AOE + (int(NBasis/5)+2)
        if (NBasis%5 != 0):
            f2.write("\n")
        if (NBasis%5 == 0):
            BOE = BOE - 1 
        f2.write(data[BOE])
        for j in range(0,NBasis):
            f2.write(" ")
            if (e2[j] >= 0):
               f2.write(" ")
            f2.write(str(fchk_notation(e2[j].real)))
            if (counter%5 ==0):
                f2.write("\n")
                counter=0
            counter = counter+1
        counter =1
        AMO = BOE + (int(NBasis/5)+2)
        if (NBasis%5 != 0):
            f2.write("\n")
        if (NBasis%5 == 0):
            AMO = AMO - 1
        f2.write(data[AMO])
        for i in range(0,NBasis):
            for j in range(0,NBasis):
                 f2.write(" ")
                 if (V1[j,i] >= 0):
                    f2.write(" ")
                 f2.write(str(fchk_notation(V1[j,i].real)))
                 if (counter%5 ==0):
                     f2.write("\n")
                     counter=0
                 counter = counter + 1
        counter = 1
        BMO = AMO + (int(NBasis*NBasis/5))+2
        if (NBasis%5 != 0):
            f2.write("\n")
        if (NBasis%5 == 0):
            BMO = BMO - 1
        f2.write(data[BMO])
        for i in range(0,NBasis):
               for j in range(0,NBasis):
                    f2.write(" ")
                    if (V2[j,i] >= 0):
                       f2.write(" ")
                    f2.write(str(fchk_notation(V2[j,i].real)))
                    if (counter%5 ==0):
                        f2.write("\n")
                        counter=0
                    counter = counter + 1
        counter = 1
        if (NBasis%5 != 0):
           f2.write("\n")
        pointer = BMO + (int(NBasis*NBasis/5))+2
        while (pointer < len(data)):
           f2.write(data[pointer])
           pointer = pointer+1
  print "Done."    

# OVMerge: Does the opposite of OVParse, merges back the Occ and Virt components of the MO Coefficient matrix
# Input  : A (Occ Matrix), B(Vir Matrix), Number of occupied orbitals, NBasis
# 
# Output : V = Full MO Coefficient Matrix
#
# (this subroutine has the exact opposite functionality of OVParse)
#

def OVMerge(A,B,NOcc,NBasis):
    V = np.zeros((NBasis,NBasis))
    for i in range(0,NOcc):
        V[:,i] = A[:,i]

    for j in range(NOcc,NBasis):
        V[:,j] = B[:,j-NOcc]

    return V

# DistanceMatrix: Calculates distances between all atoms in a molecule
# Input         : fchk file name
# 
# Output        : Returns Distance Matrix and Atomic Symbol array.
#
# Unfinished part: generate and return a distance matrix (NAtoms x NAtoms) 
#

def DistanceMatrix(filename):
    NBasis, NElem, Charge, Multiplicity, NAtoms, SCFEnergy = NBasGrab(filename)
    Atomic_Numbers = GetAtoms(filename,NAtoms)
    Atomic_Symbol = [""]*NAtoms
    for i in range(0,NAtoms):
        Atomic_Symbol[i] = AtomicSymbol(int(Atomic_Numbers[i]))
    RawCart = GeomGet(filename,NAtoms)
    Cart = np.resize(RawCart,(NAtoms,3))
    Distance_Matrix = np.zeros((NAtoms,NAtoms))
    for i in range(0,NAtoms):
        for j in range(i+1,NAtoms):
           e2 = [Cart[j,0],Cart[j,1],Cart[j,2]]
           e1 = [Cart[i,0],Cart[i,1],Cart[i,2]]
           Distance_Matrix[i,j] = np.around(DistAB(e1,e2),decimals=2)
           Distance_Matrix[j,i] = np.around(DistAB(e1,e2),decimals=2)
    return Distance_Matrix, Atomic_Symbol

# PrintLyrics: A Function made just for fun, prints out a random quote from the Beatles songs
# Input:       None, but reads in the lyrics.py library file (partially complete)
#
# Output:      None, prints lyrics.
#
    
def PrintLyrics():
    n = random.randint(1,32)
    LyricsLibrary(n)

# GetAtomicWeights: Grabs the "real atomic weights" from the fchk file
# Input:            filename, Number of Atoms
# 
# Output:           One dimensional array, AtomicWeight, of dimensions NAtoms.
#

def GetAtomicWeights(filename1,NAtoms):
   p = 0
   r = 0
   n = 1
   AtomicWeight = np.zeros(NAtoms)
   if (NAtoms%5 ==0):
     n = 0
   AtomLines = int(NAtoms/5) + n

   with open(filename1,'r') as origin:
      for i, line in enumerate(origin):
         if "Real atomic weights" in line:
            i = i + 1
            pointer = i
            endpointer = pointer + AtomLines -1
            for m in range(0, endpointer - pointer + 1):
                nextline = origin.next()
                nextline = nextline.split()
                for p in range(p,len(nextline)):
                   AtomicWeight[r] = nextline[p]
                   r = r + 1
                p = 0
   AtomicWeight = np.around(AtomicWeight,decimals=3)
   return AtomicWeight


# WriteMOsQChem: Subroutine that replaces the MO coefficients and orbital energies in a fchk file (QChem Version)
# Input:         Input filename, output filename, Orbital coefficient alpha, orbital coefficient beta, Orbtial energies alpha, orbital energies beta, number of basis functions 
# 
# Output:        None. New file will be generated (filename3) that has the new Orbital coefficients and energies
#

def WriteMOsQChem(filename1,filename3,V1,V2,e1,e2,NBasis):

  MOlines = int(len(V1)/5) + 1
  p = 0
  r = 0
  with open(filename1,'r') as origin:
      for i, line  in enumerate(origin):
          if "Alpha Orbital Energies" in line:
                AOE = i+1
                AOE_header = line
          if  "Alpha MO coefficients" in line:
                AMO = i+1
                AMO_header = line
          if "Beta Orbital Energies" in line:
                BOE = i+1
                BOE_header = line
          if "Beta MO coefficients" in line:
                BMO = i+1
                BMO_header = line

  pointer=0
  counter=1

  Start_point = min(AMO,BMO,AOE,BOE)

  with open(filename1,'r') as origin:
    data = origin.readlines()
    with open(filename3,'w') as f2:
  
        print "Writing results to new output file: ", filename3, " ... "
  
        while (pointer < Start_point-1):
           f2.write(data[pointer])
           pointer = pointer+1
        print "pointer at line = ", pointer
        f2.write(AOE_header)
        for j in range(0,NBasis):
            f2.write(" ")
            if (e1[j] >= 0):
               f2.write(" ")
            f2.write(str(fchk_notation(e1[j].real)))
            if (counter%5 == 0):
                f2.write("\n")
                counter=0
            counter=counter+1
        counter =1      
        BOE = AOE + (int(NBasis/5)+2)
        if (NBasis%5 != 0):
            f2.write("\n")
        if (NBasis%5 == 0):
            BOE = BOE - 1 
        f2.write(BOE_header)
#        f2.write("Beta Orbital Energies\n")
        for j in range(0,NBasis):
            f2.write(" ")
            if (e2[j] >= 0):
               f2.write(" ")
            f2.write(str(fchk_notation(e2[j].real)))
            if (counter%5 ==0):
                f2.write("\n")
                counter=0
            counter = counter+1
        counter =1
        AMO = BOE + (int(NBasis/5)+2)
        if (NBasis%5 != 0):
            f2.write("\n")
        if (NBasis%5 == 0):
            AMO = AMO - 1
#        f2.write("Alpha MO coefficients\n")
        f2.write(AMO_header)
        for i in range(0,NBasis):
            for j in range(0,NBasis):
                 f2.write(" ")
                 if (V1[j,i] >= 0):
                    f2.write(" ")
                 f2.write(str(fchk_notation(V1[j,i].real)))
                 if (counter%5 ==0):
                     f2.write("\n")
                     counter=0
                 counter = counter + 1
        counter = 1
        BMO = AMO + (int(NBasis*NBasis/5))+2
        if (NBasis%5 != 0):
            f2.write("\n")
        if (NBasis%5 == 0):
            BMO = BMO - 1
#        f2.write("Beta MO Coefficients\n")
        f2.write(BMO_header)
#        f2.write(data[BMO])
        for i in range(0,NBasis):
               for j in range(0,NBasis):
                    f2.write(" ")
                    if (V2[j,i] >= 0):
                       f2.write(" ")
                    f2.write(str(fchk_notation(V2[j,i].real)))
                    if (counter%5 ==0):
                        f2.write("\n")
                        counter=0
                    counter = counter + 1
        counter = 1
        if (NBasis%5 != 0):
           f2.write("\n")
        pointer = BMO + (int(NBasis*NBasis/5))+2
#        while (pointer < len(data)):
#           f2.write(data[pointer])
#           pointer = pointer+1
  print "Done."    

# ContractMat: Subroutine that reads in two square matrices (NBasis x NBasis) and returns their contraction (scalar)
# Input:       Matrices A and B (dimensions: NBasis x NBasis), NBasis
#
# Output:      scalar m = Sum_(mu,nu) A_(mu,nu) * B_(mu,nu)
#

def ContractMat(A,B,NBasis):
    value = 0.0
    for i in range(0,NBasis):
      for j in range(0,NBasis):
        value = value + A[i,j]*B[i,j]
    return value

# Work in progress: Basis set reader:

def ReadBasisSet(filename):
    NBasis, NElem, Charge, Multiplicity, NAtoms, SCFEnergy = NBasGrab(filename)
    print "Number of Basis functions =", NBasis
    print "Number of atoms =", NAtoms
    Atomic_Numbers = GetAtoms(filename,NAtoms)
    print "Atomic Numbers =", Atomic_Numbers
    Atomic_Symbol = [""]*NAtoms
    for i in range(0,NAtoms):
        Atomic_Symbol[i] = AtomicSymbol(int(Atomic_Numbers[i]))
    print "Atomic Symbols =", Atomic_Symbol


