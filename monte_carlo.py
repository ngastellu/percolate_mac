#!/usr/bin/env python

import numpy as np
from percolate import generate_site_list, LR_sites_from_MOgams, LR_sites_from_scratch
from numba import njit, guvectorize, prange, float64, int64
from numba.experimental import jitclass


# Tells Numba what types go into the MACHopSites constructor
spec = [
    ('pos', float64[:,:]),
        ('M', float64[:,:]),
        ('e_MOs', float64[:]),
        ('sites', float64[:,:]),
        ('e_sites', float64[:]),
        ('inds', int64[:]),
        ('L',int64[:]),
        ('R',int64[:]),
        ('dE', float64[:,:]),
        ('dR', float64[:,:,:])
        ]

@jitclass(spec)
class MACHopSites:

    def __init__(self, pos, M, energies, sites_data, MOgams):
        # Read in data
        self.M = M
        self.pos = pos
        self.e_MOs = energies

        # Perform MOs --> hopping sites transformation
        # sites_data used to be an optional argument, but I made it obligatory for Numba compatibility
        # the 3 arrays it contains are the output of the `generate_site_list` function defined in the 
        # `percolate` module.

        self.sites, self.e_sites, self.inds = sites_data         
        gamL, gamR = MOgams 
        self.L, self.R = LR_sites_from_MOgams(gamL, gamR, self.inds)
        
        
        # Get energy and position distances
        self.dE = self.e_sites[:,None] - self.e_sites[None,:]
        # self.dR = (self.sites[:,None,:] - self.sites[None,:,:])     
        self.dR = diffpos(self.sites) #dR[i,j,:] = r_i - r_j = pos[i] - pos[j] (vector)

    # def MCpercolate(self, T, vdos, E=np.array([1.0,0]), e_reorg=0.1, return_traj=False):
    #     Js = vdos_couplings(self.M,self.pos,self.inds)
    #     #print(Js)
    #     rates = kMarcus_gu(self.e_sites, self.sites, e_reorg, Js, T, E)
    #     if return_traj:
    #        t, traj = t_percolate(self.sites, self.L, self.R, rates, return_traj) 
    #        return t, traj
    #     else:
    #         #print("yii")
    #         t, traj = t_percolate(self.sites, self.L, self.R, rates, return_traj)
    #         return t

    def MCpercolate_MA(self, T, E, a0, return_traj=False, interMO_hops_only=False):
        rates = kMillerAbrahams(self.e_sites, self.dR, T, a0, E)
        if interMO_hops_only:
            rates = zero_intraMO_rates(rates,self.inds)
        t, traj = t_percolate(self.sites, self.L, self.R, rates, return_traj) 
        return t, traj


    def MCpercolate_dipoles(self, Jdipoles, T, E, e_reorg, return_traj, interMO_hops_only):
        rates = kMarcus_njit(self.e_sites, self.sites, e_reorg, Jdipoles, T, E)
        if interMO_hops_only:
            rates = zero_intraMO_rates(rates,self.inds)
        t, traj = t_percolate(self.sites, self.L, self.R, rates, return_traj) 
        return t, traj


@njit(parallel=True)
def diffpos(pos):
    N, d = pos.shape
    dR = np.empty((N,N,d),dtype='float')
    for i in prange(N):
        for j in prange(N):
            for k in prange(d):
                dR[i,j,k] = pos[i,k] - pos[j,k]
    return dR


@njit
def hop_step(i,rates,sites):
    zero_inds = np.unique((rates[i]==0).nonzero()[0])
    # print(f'ZERO RATES [i = {i}]: ', zero_inds)
    N = sites.shape[0]
    x = - np.log(1 - np.random.rand(N))
    # print('any(x<0) = ', np.any(x<0))

    #remove ith entry from hopping times to avoid remaining on the same site forever
    if i > 0 and i < N-1: #this condition prevents trying to take argmin of empty array (eg. hop_times[:0])
        # print("Case 1\n")
        hop_times1 = x[:i]/rates[i,:i]
        # print("0Hop times 1 = ", hop_times1[zero_inds[zero_inds < i]])
        # print('hop_times1.shape = ', hop_times1.shape)
        hop_times2 = x[i+1:]/rates[i,i+1:]
        # print("0Hop times 2 = ", hop_times2[zero_inds[zero_inds > i]-i-1])
        # print('hop_times2.shape = ', hop_times2.shape)
        j1 = np.argmin(hop_times1)
        j2 = np.argmin(hop_times2)
        if hop_times1[j1] < hop_times2[j2]:
            return j1, hop_times1[j1]
        else:
            return i+1+j2, hop_times2[j2]
    elif i == 0:
        # print("Case 2\n")
        hop_times = x[1:] / rates[0,1:]
        return np.argmin(hop_times) + 1, np.min(hop_times)
    else: # i = N-1
        # print("Case 3\n")
        hop_times = x[:N] / rates[N-1,:N]
        return np.argmin(hop_times), np.min(hop_times)

# #@njit
# def kMarcus(ediffs,rdiffs,Js,T,E=np.array([1.0,0]),e_reorg=0.1):
#     kB = 8.617e-5
#     hbar = 6.582e-1 # in eV * fs
#     A = 4 * e_reorg * kB * T
#     e = 1.00
#     # e = 1.602e-19 #C
#     return 2 * np.pi * (Js**2) * np.exp(-((e_reorg - ediffs + e * np.dot(rdiffs,E))**2)/A) / (hbar * np.sqrt(np.pi * A))


@guvectorize([(float64[:,:], int64[:], float64[:,:])], '(N,n), (m) -> (m,m)', nopython=True, target_backend='parallel')
def get_site_overlap(M, sites2MOs, S_sites):     
        nsites = sites2MOs.shape[0] #sites2MOs[i] = MO index corresponding to site i
        S = np.abs(M).T @ np.abs(M)
        #translate overlap matrix from MO indices to site indices 
        for i in prange(nsites):
            for j in prange(nsites):
                S_sites[i,j] = S[sites2MOs[i],sites2MOs[j]]
                #print(S_sites[i,j])


# @guvectorize([(float64[:,:], float64[:,:], int64[:], float64[:,:])], '(N,n), (N,p), (m) -> (m,m)', nopython=True)#, target_backend='parallel')
@njit(parallel=True)
def dipole_coupling(M, pos, sites2MOs):
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

def vdos_couplings(S_sites, dE_sites, vdos, T, A=1.0):
        if vdos.ndim > 1:
            D = np.interp(dE_sites, vdos[0], vdos[1])
        else:
            D = vdos[0] #vdos is ndarray with a single element

        return (A**2) * (S_sites**2) * (bose_einstein(dE_sites, T) + 1) * D / dE_sites

@guvectorize([(float64[:], float64[:,:], float64, float64[:,:], float64, float64[:], float64[:,:])], '(n),(n,p),(),(n,n),(),(p) -> (n,n)', nopython=True,target_backend='parallel')
def kMarcus_gu(energies, pos, e_reorg, Js, T, E, out):
    kB = 8.617e-5 # eV * K
    hbar = 6.582e-1 # eV * fs
    A = 4 * e_reorg * kB * T
    #e = 1.602e-19
    e = 1.00
    N = energies.shape[0]
    for i in range(N):
        for j in range(N):
            out[i,j] = 2 * np.pi * Js[i,j]* Js[i,j] * np.exp(-(e_reorg + energies[j] - energies[i] - e * np.dot(E,(pos[j] - pos[i])))**2/A) / (hbar * np.sqrt(np.pi * A))

@njit(parallel=True)
def kMarcus_njit(energies, pos, e_reorg, Js, T, E):
    print("Evaluating Marcus rates...")# Printing sites (i,j) between whom k_ij = 0:")
    kB = 8.617e-5 # eV * K
    hbar = 6.582e-1 # eV * fs
    A = 4 * e_reorg * kB * T
    #e = 1.602e-19
    e = 1.00
    cnt = 0
    N = energies.shape[0]
    print('N = ', N)
    out = np.empty((N,N),dtype='float')
    for i in prange(N):
        for j in prange(N):
            out[i,j] = 2 * np.pi * Js[i,j]* Js[i,j] * np.exp(-(e_reorg + energies[j] - energies[i] - e * np.dot(E,(pos[j] - pos[i])))**2/A) / (hbar * np.sqrt(np.pi * A))
            if out[i,j] == 0: 
                # print(A)
                #print(np.exp(-(e_reorg + energies[j] - energies[i] - e * np.dot(E,(pos[j] - pos[i])))**2/A))  
                #print(-(e_reorg + energies[j] - energies[i] - e * np.dot(E,(pos[j] - pos[i])))**2/A)
                cnt += 1
    print("Done!")
    print(f"Fraction of zero elements = {cnt}/{out.size}")
    return out

@njit(parallel=True)
def kMillerAbrahams(energies, dR, T, a0, E): 
    """!!!! SITE energies and positions must be used here (as opposed to MO energies and atomic positions) !!!!"""
    kB = 8.617e-5 # eV * K
    e = 1.0
    N = energies.shape[0]
    out = np.empty((N,N),dtype='float')
    for i in prange(N):
        for j in prange(N):
            out[i,j] = np.exp( -(np.abs(energies[i] - energies[j]) + np.abs(energies[i] + energies[j]) - e*np.dot(E, dR[i,j,:]))/(2*kB*T) - (2*np.linalg.norm(dR[i,j,:]))/a0 )
    return out


@njit(parallel=True)
def zero_intraMO_rates(K,sites2MOs):
    nsites = sites2MOs.shape[0]
    assert K.shape[0] == nsites, "Number of sites does not match size of rates matrix!"

    for i in prange(nsites):
        MOind = sites2MOs[i]
        matchingMOs = (sites2MOs == MOind).nonzero()[0] # indices of sites who came from the same MO
        K[i,matchingMOs] = 0
    
    return K



#@vectorize([float64(float64,float64)], nopython=True)
def bose_einstein(e,T):
    kB = 8.617e-5
    return 1.0/(np.exp(e/(kB*T)) - 1)



@njit
def t_percolate(sites, L, R, rates, return_traj=False):
    site = np.random.choice(L)
    print(f"Starting t_percolate at site {site}")
    t = 0
    nstep = 0
    if return_traj:
        nbuffer = 10000000
        traj = np.ones(nbuffer,'int') * -1
    
    
    while site not in R:
        # if nstep % 10000 == 0:
        #     print(f"t = {nstep}; site = {site}")
        if return_traj:
            if nstep < nbuffer:
                traj[nstep] = site
            else:
                print("Buffer exceeded! Reallocating traj array...")
                nbuffer += 10000000
                tmp = traj.copy()
                traj = np.ones(nbuffer,'int') * -1
                traj[:nstep] = tmp
                traj[nstep] = site
                tmp = 0
                print("Done.")
                #del tmp
        site, hop_time = hop_step(site, rates, sites)
        # print("hop time = ", hop_time)
        t+=hop_time
        nstep += 1
    if return_traj:
        if nstep < nbuffer:
            traj[nstep] = site
        else:
            print("At boundary return case of t_percolate! [monte_carlo.py:190]")
            tmp = traj.copy()
            traj = np.ones(nbuffer+1,'int') * -1
            traj[:nstep] = tmp
            traj[nstep] = site
            tmp = 0
            #del tmp
        print("Exiting t_percolate (return_traj = True)")
        return t, traj[traj >= 0]
    else:
        print("Exiting t_percolate (return_traj = False)")
        #print("yo")
        return t, np.zeros(1,'int') #return np.zeros(1) to get the return types to match (otherwise Numba freaks out)