#!/usr/bin/env python

import sys
from os import path
import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
from qcnico.remove_dangling_carbons import remove_dangling_carbons
from qcnico.coords_io import read_xsf
from monte_carlo import dipole_coupling

@njit(parallel=True)
def dipole_coupling_default(M, pos, sites2MOs):
    N = M.shape[0] #nb of atoms
    n = M.shape[1] #nb of MOs
    d = pos.shape[1] # nb of spatial dimensions (2 or 3) 
    m = sites2MOs.shape[0] #nb of sites
    J_MOs = np.empty((n,n),dtype='float')
    J = np.empty((m,m),dtype='float')
    #e = 1.602e-19
    e = 1.00
    #Mcol_prod = (M.T)[:,:,None] * M # Mcol_prod[j] = {jth column of M} * M (column-wise) != M[:,j] * M (<--- row-wise multiplication)
    
    # Doing triple for loop to avoid storing a huge array of shape (n,n,N,2)
    print("Entering triple for-loop...")
    for i in prange(n):
        for j in prange(n):
            tmp = np.empty(d,dtype='float')
            for k in range(N):
                tmp += M[k,i] * M[k,j] * pos[k,:] 
            J_MOs[i,j] = np.linalg.norm(tmp) * e
    print("Done!")
    
    for i in prange(m):
        for j in prange(m):
            # print(f"(i,j) = ({i}, {j}) ---> ({sites2MOs[i]}, {sites2MOs[j]})")
            J[i,j] = J_MOs[sites2MOs[i], sites2MOs[j]]

    return J

@njit
def dipole_coupling_serial(M, pos, sites2MOs):
    N = M.shape[0] #nb of atoms
    n = M.shape[1] #nb of MOs
    d = pos.shape[1] # nb of spatial dimensions (2 or 3) 
    m = sites2MOs.shape[0] #nb of sites
    J_MOs = np.empty((n,n),dtype='float')
    J = np.empty((m,m),dtype='float')
    #e = 1.602e-19
    e = 1.00
    #Mcol_prod = (M.T)[:,:,None] * M # Mcol_prod[j] = {jth column of M} * M (column-wise) != M[:,j] * M (<--- row-wise multiplication)
    
    # Doing triple for loop to avoid storing a huge array of shape (n,n,N,2)
    print("Entering triple for-loop...")
    for i in range(n):
        for j in range(n):
            tmp = np.empty(d,dtype='float')
            for k in range(N):
                tmp += M[k,i] * M[k,j] * pos[k,:] 
            J_MOs[i,j] = np.linalg.norm(tmp) * e
    print("Done!")
    
    for i in range(m):
        for j in range(m):
            # print(f"(i,j) = ({i}, {j}) ---> ({sites2MOs[i]}, {sites2MOs[j]})")
            J[i,j] = J_MOs[sites2MOs[i], sites2MOs[j]]
    return J

@njit(parallel=True)
def dipole_coupling_full_parallel(M, pos, sites2MOs):
    N = M.shape[0] #nb of atoms
    n = M.shape[1] #nb of MOs
    d = pos.shape[1] # nb of spatial dimensions (2 or 3) 
    m = sites2MOs.shape[0] #nb of sites
    J_MOs = np.empty((n,n),dtype='float')
    J = np.empty((m,m),dtype='float')
    #e = 1.602e-19
    e = 1.00
    #Mcol_prod = (M.T)[:,:,None] * M # Mcol_prod[j] = {jth column of M} * M (column-wise) != M[:,j] * M (<--- row-wise multiplication)
    
    # Doing triple for loop to avoid storing a huge array of shape (n,n,N,2)
    print("Entering triple for-loop...")
    for i in prange(n):
        for j in prange(n):
            tmp = np.empty(d,dtype='float')
            for k in prange(N):
                tmp += M[k,i] * M[k,j] * pos[k,:] 
            J_MOs[i,j] = np.linalg.norm(tmp) * e
    print("Done!")
    
    for i in prange(m):
        for j in prange(m):
            # print(f"(i,j) = ({i}, {j}) ---> ({sites2MOs[i]}, {sites2MOs[j]})")
            J[i,j] = J_MOs[sites2MOs[i], sites2MOs[j]]
    return J



nsample = 150

MOdir = path.expanduser(f"~/Desktop/simulation_outputs/percolation/40x40/MOs_ARPACK/")
M = np.load(path.join(MOdir,f'MOs_ARPACK_bigMAC-{nsample}.npy'))
print(M)

percolate_datadir = f'/Users/nico/Desktop/simulation_outputs/percolation/40x40/percolate_output/sample-{nsample}/'


site_inds = np.load(percolate_datadir + 'ii.npy')
print(site_inds)


strucdir = '/Users/nico/Desktop/simulation_outputs/percolation/40x40/structures/'

pos = remove_dangling_carbons(read_xsf(strucdir + f'bigMAC-{nsample}_relaxed.xsf')[0], 1.8 )
print(pos)

outdir = '/Users/nico/Desktop/simulation_outputs/percolation/40x40/monte_carlo/dipole_couplings/testing/local'

Js0 = dipole_coupling(M,pos,site_inds)
np.save(f"{outdir}/defaultJs-{nsample}.npy", Js0)
plt.imshow(Js0)
plt.suptitle("from `monte_carlo`")
plt.colorbar()
plt.show()


Js1 = dipole_coupling_full_parallel(M,pos,site_inds)
np.save(f"{outdir}/fulparJs-{nsample}.npy", Js1)
plt.imshow(Js1)
plt.suptitle("full parallel")
plt.colorbar()
plt.show()

# Js2 = dipole_coupling_serial(M,pos,site_inds)
# np.save(f"{outdir}/seriJs-{nsample}.npy", Js2)
# plt.imshow(Js2)
# plt.suptitle("full serial")
# plt.colorbar()
# plt.show()

print   ("Default == FullParallel: ", np.all(Js0==Js1))
# print("Default == Serial: ", np.all(Js0==Js2))
# print("FullParallel == Serial: ", np.all(Js1==Js2))
