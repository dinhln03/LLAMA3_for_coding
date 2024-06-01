import MPS_class as MPS
import MPO_class as MPO
from ncon import ncon
import numpy as np
from scipy.linalg import expm
#%%
def TEBD_evo(MPS_,Lx,Ly,J=1,epsilon=0.1,etrunc=0,chiMAX=256,chiMAXswap=256,info=True):
    L = Lx*Ly
    config = np.arange(0,L).reshape(Lx,Ly)
    
    theta = (np.pi+2*epsilon)
    flip_op = np.eye(2)*np.cos(theta/2) - 1j*np.sin(theta/2)*np.array([[0,1],[1,0]])    
    
    sigma_z = np.array([[1,0],[0,-1]])
    Uprop = expm(-1j*np.kron(sigma_z,-J*sigma_z)).reshape(2,2,2,2)
    nn_list_forward = [[] for x in range(L)]
    for x in range(L):
        i,j = np.where(config == x)
        if j != Ly-1: nn_list_forward[x].append( config[i,j+1])
        if i != Lx-1: nn_list_forward[x].append( config[i+1,j])
        nn_list_forward[x] = np.array(nn_list_forward[x]).ravel()
    nn_list_backward = [[] for x in range(L)]    
    for x in reversed(range(L)):
        i,j = np.where(config == x)
        if j != 0: nn_list_backward[x].append( config[i,j-1])
        if i != 0: nn_list_backward[x].append( config[i-1,j])
        nn_list_backward[x] = np.array(nn_list_backward[x]).ravel()
    
    for x in range(L):
        for nn in nn_list_forward[x]:
            # If they are nearest neighbours
            if nn == x+1:
               shpM1,shpM2 = MPS_.M[x].shape, MPS_.M[nn].shape
               Theta = ncon([MPS_.M[x],MPS_.M[nn],Uprop],[[-1,2,1],[1,3,-4],[2,3,-2,-3]]) 
               Theta = Theta.reshape(shpM1[0]*shpM1[1], shpM2[1]*shpM2[2])
               U,S,V = MPS.svdtruncate(Theta, etrunc, chiMAX,info=info)
               MPS_.M[x]  = U.reshape(shpM1[0],shpM1[1],S.size)
               MPS_.M[nn] = (np.diag(S)@V).reshape(S.size,shpM2[1],shpM2[2])
            else:
                for index in range(x,nn-1):
                    MPS_.swap(index,chiMAX=chiMAXswap,info=info)
                shpM1,shpM2 = MPS_.M[nn-1].shape, MPS_.M[nn].shape
                Theta = ncon([MPS_.M[nn-1],MPS_.M[nn],Uprop],[[-1,2,1],[1,3,-4],[2,3,-2,-3]]) 
                Theta = Theta.reshape(shpM1[0]*shpM1[1], shpM2[1]*shpM2[2])
                U,S,V = MPS.svdtruncate(Theta, etrunc, chiMAX,info=info)
                MPS_.M[nn-1]  = (U@np.diag(S)).reshape(shpM1[0],shpM1[1],S.size)
                MPS_.M[nn] = V.reshape(S.size,shpM2[1],shpM2[2]) 
                for index in reversed(range(x,nn-1)):
                    MPS_.swap(index,chiMAX=chiMAXswap,info=info)
        MPS_.M[x] = ncon([MPS_.M[x],flip_op],[[-1,1,-3],[1,-2]])
    
    for x in reversed(range(L)):
        for nn in nn_list_backward[x]:
            # If they are nearest neighbours
            if nn == x-1:
               shpM1,shpM2 = MPS_.M[nn].shape, MPS_.M[x].shape
               Theta = ncon([MPS_.M[nn],MPS_.M[x],Uprop],[[-1,2,1],[1,3,-4],[2,3,-2,-3]]) 
               Theta = Theta.reshape(shpM1[0]*shpM1[1], shpM2[1]*shpM2[2])
               U,S,V = MPS.svdtruncate(Theta, etrunc, chiMAX,info=info)
               MPS_.M[nn]  = (U@np.diag(S)).reshape(shpM1[0],shpM1[1],S.size)
               MPS_.M[x] = (V).reshape(S.size,shpM2[1],shpM2[2])
            else:
               for index in range(x-1,nn,-1):
                   MPS_.swap(index,chiMAX=chiMAXswap,center='i',info=info)
               shpM1,shpM2 = MPS_.M[nn].shape, MPS_.M[nn+1].shape
               Theta = ncon([MPS_.M[nn],MPS_.M[nn+1],Uprop],[[-1,2,1],[1,3,-4],[2,3,-2,-3]]) 
               Theta = Theta.reshape(shpM1[0]*shpM1[1], shpM2[1]*shpM2[2])
               U,S,V = MPS.svdtruncate(Theta, etrunc, chiMAX,info=info)
               MPS_.M[nn]  = U.reshape(shpM1[0],shpM1[1],S.size)
               MPS_.M[nn+1] = (np.diag(S)@V).reshape(S.size,shpM2[1],shpM2[2]) 
               for index in reversed(range(x-1,nn,-1)):
                   MPS_.swap(index,chiMAX=chiMAXswap,center='i',info=info)           
        MPS_.M[x] = ncon([MPS_.M[x],flip_op],[[-1,1,-3],[1,-2]])

Lx = 5
Ly = Lx

L  = Lx*Ly
psi_state = MPS.getAllUp(L)

mag = []
err = 0.
info = True
mag.append(MPO.return_LocalMz(psi_state).real.reshape(Lx,Ly))
for k in range(20):
    print('k',k,np.max(mag[k]-mag[k].T))
    for x in psi_state.M:
        print(x.shape)
    TEBD_evo(psi_state, Lx, Ly,J = -1j,chiMAX=256,chiMAXswap=256,etrunc=1e-12,info=info)
    mag.append(MPO.return_LocalMz(psi_state).real.reshape(Lx,Ly))
mag = np.array(mag)
#%%

from scipy.sparse.linalg import expm_multiply
import scipy.sparse as sps
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import seaborn as sns


def nested_tensor(lst):
    if len(lst) == 2:
        return sps.kron(lst[0],lst[1],format='csc')
    else:
        return sps.kron(lst[0], nested_tensor(lst[1:]),format='csc')

def spin_op_construct(sigma, j, L):
    before = [sps.eye(2) for _ in range(j)]
    mid    = [sigma]
    after  = [sps.eye(2) for _ in range(j+1,L)]
    return nested_tensor(before+mid+after)
def int_spin_op_construct(sigma1,sigma2,i1,i2,L):
    if i2 < i1:
        i1,i2 = i2,i1
    before1 = [sps.eye(2) for _ in range(i1)]
    mid1    = [sigma1]
    after1  = [sps.eye(2) for _ in range(i1+1,i2)]
    mid2    = [sigma2]
    after2  = [sps.eye(2) for _ in range(i2+1,L)]
    return nested_tensor(before1+mid1+after1+mid2+after2)

def H1(L, epsilon):
    sigma_x = sps.csc_matrix(np.array([[0,1],[1,0]]))
    op1 = 0
    for i in range(L):
        op1 += spin_op_construct(-sigma_x*(np.pi/2+epsilon),i,L)
    return op1
    
def H2(Lx, Ly, J=1):
    L = Lx*Ly
    sigma_z = sps.csc_matrix(np.array([[1,0],[0,-1]]))
    op2 = 0
    config = np.arange(L).reshape(Lx,Ly)
    for i in range(Lx):
        for j in range(Ly):
            nn = []
            if i != Lx-1:
                nn.append(config[i+1,j])
            if j != Ly-1:
                nn.append(config[i,j+1])
            op = 0
            for x in nn:
                op += int_spin_op_construct(-J*sigma_z,sigma_z,config[i,j],x,L)
            op2 += op
    return op2

def H2_pbc(Lx, Ly, J=1):
    L = Lx*Ly
    sigma_z = sps.csc_matrix(np.array([[1,0],[0,-1]]))
    op2 = 0
    config = np.arange(L).reshape(Lx,Ly)
    for i in range(Lx):
        for j in range(Ly):
            nn = []
            nn.append(config[(i+1)%(Lx-1),j])
            nn.append(config[i,(j+1)%(Ly-1)])
            op = 0
            for x in nn:
                op += int_spin_op_construct(-J*sigma_z,sigma_z,config[i,j],x,L)
            op2 += op
    return op2


def H2_pbc1D(L, J=1):
    sigma_z = sps.csc_matrix(np.array([[1,0],[0,-1]]))
    op2 = 0
    for i in range(L):
        op2 += int_spin_op_construct(-J*sigma_z,sigma_z,i,(i+1)%L,L)
    return op2

def H2_pbc1D_var(L, J=1):
    sigma_z = sps.csc_matrix(np.array([[1,0],[0,-1]]))
    op2 = 0
    for i in range(1,L-1):
        op2 += int_spin_op_construct(-J*sigma_z,sigma_z,i,(i+1),L)
    op2 += spin_op_construct(-J*0.5*np.eye(2), L-1, L)    
    op2 += spin_op_construct(-J*0.5*np.eye(2), 0, L)    
    return op2

Lx = 4;
Ly = Lx;
L  = Lx*Ly;
D  = 2**L
en = []

mz_config = np.zeros(D)
for i,state in enumerate(np.vectorize(np.binary_repr)(np.arange(2**L),L)):
    mz_config[i] = (L-2*state.count('1'))/L

Hdouble_1d  = H2_pbc1D(L)
Hdouble_1dv = H2_pbc1D_var(L) 

epsilon = 0.1

Hsingle = H1(L,epsilon)
psi0    = np.zeros(D)
psi0[0] = 1
psi1dv  = [psi0]
psi1d   = [psi0]
for n in range(200):
    print(n,' ',end='')
    psi1dv.append(expm_multiply(-1j*Hsingle,expm_multiply(-1j*Hdouble_1dv,psi1dv[-1])))
    psi1d.append(expm_multiply(-1j*Hsingle,expm_multiply(-1j*Hdouble_1d,psi1d[-1])))
#%%
psi1dv = np.array(psi1dv)
psi1d = np.array(psi1d)

mag_ED = np.abs(psi1dv)**2@mz_config.reshape(D,1)
mag_ED = mag_ED.reshape(mag_ED.size)

mag_ED1d = np.abs(psi1d)**2@mz_config.reshape(D,1)
mag_ED1d = mag_ED1d.reshape(mag_ED1d.size)


plt.plot(np.abs(mag_ED))
plt.plot(np.abs(mag_ED1d))
#%%

L1d = np.zeros(psi1d.shape[0])
L2d = np.zeros(psi1d.shape[0])
for x in range(psi1d.shape[0]):
    if x%2 == 0: k = 0
    else: k=-1
    L1d[x] = np.abs(psi1d[x,k])**2
    L2d[x] = np.abs(psi[x,k])**2

#%%
K = (sps.eye(2**L)-1j*Hsingle)
U = (sps.eye(2**L)-1j*Hdouble)
psi_1 = []
psi_1.append(psi0)
for n in range(100):
    print(n,' ',end='')
    psi_1.append(K.dot(U.dot(psi_1[-1])))
    psi_1[-1] /= np.linalg.norm(psi_1[-1])
psi_1 = np.array(psi_1)
mag_ED1 = np.abs(psi_1)**2@mz_config.reshape(D,1)

plt.plot(mag_ED)
plt.plot(mag_ED1)