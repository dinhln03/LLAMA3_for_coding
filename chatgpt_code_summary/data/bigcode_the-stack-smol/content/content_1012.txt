#!/usr/bin/env python
# File: plot_icd_vs_colorgrad.py
# Created on: Tue 08 May 2012 11:03:26 AM CDT
# Last Change: Sun 21 Oct 2012 02:43:33 PM CDT
# Purpose of script: <+INSERT+>
# Author: Steven Boada

import pylab as pyl
from mk_galaxy_struc import mk_galaxy_struc

galaxies = mk_galaxy_struc()

f1 = pyl.figure(1,figsize=(8,8))
f1s1 = f1.add_subplot(221)
f1s2 = f1.add_subplot(222)
f1s3 = f1.add_subplot(223)
f1s4 = f1.add_subplot(224)

for galaxy in galaxies:
    if galaxy.ston_I >= 30. and galaxy.Color_grad != None and galaxy.sersic !=\
None:
        if galaxy.sersic < 1.:
            col1 =f1s1.scatter(galaxy.ICD_IH, galaxy.Color_grad, s=50, c='k',
            edgecolor='w')
        if 1. < galaxy.sersic < 2.:
            col2 =f1s2.scatter(galaxy.ICD_IH, galaxy.Color_grad, s=50,c='k',
            edgecolor='w')
        if 2. < galaxy.sersic < 3.:
            col3 =f1s3.scatter(galaxy.ICD_IH, galaxy.Color_grad, s=50, c='k',
            edgecolor='w')
        if 3. <  galaxy.sersic:
            col4 =f1s4.scatter(galaxy.ICD_IH, galaxy.Color_grad, s=50, c='k',
            edgecolor='w')


        #pyl.scatter(galaxy.ICD_IH,galaxy.Color_grad,s=50,edgecolor='w')


#f1s1.vlines(0.04,-3.,1,lw=2,zorder=0)
#f1s1.hlines(0.0,-0.1,0.25,lw=2,zorder=0)

#pyl.text(0.24, 0.7, "Blue Core, Red Edge", size=15, ha="right", va="top",
#        bbox = dict(boxstyle="round", ec=(1., 0.5, 0.5),
#        fc=(1., 0.8, 0.8)))

#pyl.text(0.24, -2.5, "Red Core, Blue Edge", size=15, ha="right", va="top",
#        bbox = dict(boxstyle="round", ec=(1., 0.5, 0.5),
#        fc=(1., 0.8, 0.8)))

# Finish Plot
f1s1.set_xlim(-0.05,0.25)
f1s1.set_ylim(-3.,1)
f1s2.set_xlim(-0.05,0.25)
f1s2.set_ylim(-3.,1)
f1s3.set_xlim(-0.05,0.25)
f1s3.set_ylim(-3.,1)
f1s4.set_xlim(-0.05,0.25)
f1s4.set_ylim(-3.,1)
#pyl.subplots_adjust(left=0.15,bottom=0.15)

f1s1.set_xlabel(r'$\xi[I,H]$')
f1s1.set_ylabel('Color Gradient')

pyl.savefig('icd_vs_color_grad_vs_sersic_IH.eps',bbox='tight')

pyl.show()
