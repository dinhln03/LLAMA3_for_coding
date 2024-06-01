import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, MaxNLocator
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.colors import BoundaryNorm
import matplotlib.image as mpimg

Uinf=1
R=15
PI=np.pi
alpha = 1
w = alpha/R

gamma= -w * 2*PI* R*R

angle = np.linspace(0, 360, 360)

cp = 1 - (4*(np.sin(angle*(PI/180) )**2) + (2*gamma*np.sin(angle *(PI/180)))/(PI*R*Uinf)  + (gamma/(2*PI*R*Uinf))**2  )




fig, ax = plt.subplots()

ax.plot(angle, cp, '--k')
#ax.plot(angle, Z[edge_x,edge_y], 'ok', markersize=5)


#ax.set_ylim(limits[0], limits[1]) 

#Grid
ax.xaxis.set_minor_locator(AutoMinorLocator(4))
ax.yaxis.set_minor_locator(AutoMinorLocator(4))
ax.grid(which='major', color='#CCCCCC', linestyle='-', alpha=1)
ax.grid(which='minor', color='#CCCCCC', linestyle='--', alpha=0.5)

fig.savefig(f'./cp_{alpha}.png')
plt.close()