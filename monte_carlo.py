#!/usr/bin/env python

import numpy as np
from percolate import generate_site_list, LR_sites_from_MOgams, LR_sites_from_scratch
from numba import njit, guvectorize, float64, int64
#from numba.experimental import jitclass

#@jitclass
class MACHopSites:

    def __init__(self, pos, M, energies, sites_data=None, MOgams=None):
        # Read in data
        self.M = M
        self.pos = pos
        self.e_MOs = energies

        # Perform MOs --> hopping sites transformation
        if sites_data is None:
            self.sites, self.e_sites, self.inds = generate_site_list(self.pos, self.M, self.e_MOs)
        else:
            self.sites, self.e_sites, self.inds = sites_data
        
        # Define sites that are strongly coupled to the leads
        if MOgams is None:
            self.L, self.R = LR_sites_from_scratch(pos,self.inds,return_ndarray=True)
        else:
            gamL, gamR = MOgams
            self.L, self.R = LR_sites_from_MOgams(gamL, gamR, self.inds,return_ndarray=True)
        
        
        # Get energy and position distances
        self.dE = diff_array(self.e_sites)
        print(np.max(np.abs(self.dE)[np.abs(self.dE) > 0]))
        self.dR = (self.sites[:,None,:] - self.sites[None,:,:])
    

    def MCpercolate(self, T, vdos, E=np.array([1.0,0]), e_reorg=0.1, return_traj=False):
        Js = vdos_couplings(self.M,self.pos,self.inds)
        #print(Js)
        rates = kMarcus_gu(self.e_sites, self.sites, e_reorg, Js, T, E)
        if return_traj:
           t, traj = t_percolate(self.sites, self.L, self.R, rates, return_traj) 
           return t, traj
        else:
            #print("yii")
            t, traj = t_percolate(self.sites, self.L, self.R, rates, return_traj)
            return t

    def MCpercolate_dipoles(self, Jdipoles, T, E=np.array([1.0,0]), e_reorg=0.1, return_traj=False):
        rates = kMarcus_gu(self.e_sites, self.sites, e_reorg, Jdipoles, T, E)
        if return_traj:
           t, traj = t_percolate(self.sites, self.L, self.R, rates, return_traj) 
           return t, traj
        else:
            t, traj = t_percolate(self.sites, self.L, self.R, rates, return_traj)
            return t

def diff_array(arr):
    return arr[:,None] - arr[None,:]

@njit
def hop_step(i,rates,sites):
    N = sites.shape[0]
    x = - np.log(1 - np.random.rand(N))

    #remove ith entry from hopping times to avoid remaining on the same site forever
    if i > 0 and i < N-1: #this condition prevents trying to take argmin of empty array (eg. hop_times[:0])
        #print("Case 1\n")
        hop_times1 = x[:i]/rates[i,:i]
        hop_times2 = x[i+1:]/rates[i,i+1:]
        j1 = np.argmin(hop_times1)
        j2 = np.argmin(hop_times2)
        if hop_times1[j1] < hop_times2[j2]:
            return j1, hop_times1[j1]
        else:
            return i+1+j2, hop_times2[j2]
    elif i == 0:
        #print("Case 2\n")
        hop_times = x[1:] / rates[0,1:]
        return np.argmin(hop_times) + 1, np.min(hop_times)
    else: # i = N-1
        #print("Case 3\n")
        hop_times = x[:N] / rates[N,:N]
        return np.argmin(hop_times), np.min(hop_times)

#@njit
def kMarcus(ediffs,rdiffs,Js,T,E=np.array([1.0,0]),e_reorg=0.1):
    kB = 8.617e-5
    hbar = 6.582e-1 # in eV * fs
    A = 4 * e_reorg * kB * T
    e = 1.00
    # e = 1.602e-19 #C
    return 2 * np.pi * (Js**2) * np.exp(-((e_reorg - ediffs + e * np.dot(rdiffs,E))**2)/A) / (hbar * np.sqrt(np.pi * A))


@guvectorize([(float64[:,:], int64[:], float64[:,:])], '(N,n), (m) -> (m,m)', nopython=True)
def get_site_overlap(M, sites2MOs, S_sites):     
        nsites = sites2MOs.shape[0] #sites2MOs[i] = MO index corresponding to site i
        S = np.abs(M).T @ np.abs(M)
        #translate overlap matrix from MO indices to site indices 
        for i in range(nsites):
            for j in range(nsites):
                S_sites[i,j] = S[sites2MOs[i],sites2MOs[j]]
                #print(S_sites[i,j])

@guvectorize([(float64[:,:], float64[:,:], int64[:], float64[:,:])], '(N,n), (N,p), (m) -> (m,m)', nopython=True)
def dipole_coupling(M, pos, sites2MOs, J):
    N = M.shape[0] #nb of atoms
    n = M.shape[1] #nb of MOs
    d = pos.shape[1] # nb of spatial dimensions (2 or 3) 
    m = sites2MOs.shape[0]
    J_MOs = np.zeros((n,n),dtype='float')
    # e = 1.602e-19
    e = 1.00
    #Mcol_prod = (M.T)[:,:,None] * M # Mcol_prod[j] = {jth column of M} * M (column-wise) != M[:,j] * M (<--- row-wise multiplication)
    
    # Doing triple for loop to avoid storing a huge array of shape (n,n,N,2)
    print("Entering triple for-loop...")
    for i in range(n):
        for j in range(n):
            tmp = np.zeros(d)
            for k in range(N):
                tmp += M[k,i] * M[k,j] * pos[k,:] 
            J_MOs[i,j] = np.linalg.norm(tmp) * e
    print("Done!")
    
    for i in range(m):
        for j in range(m):
            J[i,j] = J_MOs[sites2MOs[i], sites2MOs[j]]

def vdos_couplings(S_sites, dE_sites, vdos, T, A=1.0):
        if vdos.ndim > 1:
            D = np.interp(dE_sites, vdos[0], vdos[1])
        else:
            D = vdos[0] #vdos is ndarray with a single element

        return (A**2) * (S_sites**2) * (bose_einstein(dE_sites, T) + 1) * D / dE_sites

@guvectorize([(float64[:], float64[:,:], float64, float64[:,:], float64, float64[:], float64[:,:])], '(n),(n,p),(),(n,n),(),(p) -> (n,n)', nopython=True)
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

#@vectorize([float64(float64,float64)], nopython=True)
def bose_einstein(e,T):
    kB = 8.617e-5
    return 1.0/(np.exp(e/(kB*T)) - 1)



@njit
def t_percolate(sites, L, R, rates, return_traj=False):
    site = np.random.choice(L)
    t = 0
    nstep = 0
    if return_traj:
        nbuffer = 10000
        traj = np.ones(nbuffer,'int') * -1
    
    
    while site not in R:
        print(f"t = {nstep}; site = {site}")
        if return_traj:
            if nstep < nbuffer:
                traj[nstep] = site
            else:
                nbuffer += 10000
                tmp = traj.copy()
                traj = np.ones(nbuffer,'int') * -1
                traj[:nstep] = tmp
                traj[nstep] = site
        site, hop_time = hop_step(site, rates, sites)
        t+=hop_time
        nstep += 1
    if return_traj:
        if nstep < nbuffer:
            traj[nstep] = site
        else:
            tmp = traj.copy()
            traj = np.ones(nbuffer+1,'int') * -1
            traj[:nstep] = tmp
            traj[nstep] = site
        return t, traj[traj >= 0]
    else:
        #print("yo")
        return t, np.zeros(1,'int') #return np.zeros(1) to get the return types to match (otherwise Numba freaks out)