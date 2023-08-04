#!/usr/bin/env python

import numpy as np
from numpy.random import default_rng
from percolate import generate_site_list, LR_sites
from qcnico.vdos import vdos
#from numba import njit, vectorize, guvectorize


class MACHopSites:

    def __init__(self, pos, M, energies, sites_data=None, edge_sites=None):
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
        if edge_sites is None:
            self.L, self.R = LR_sites(pos,self.inds)
        else:
            self.L, self.R = edge_sites
        
        # Get energy and position distances
        self.dE = diff_array(self.e_sites)
        self.dR = self.pos[:,None,:] - self.pos[None,:,:]
    
    def MCpercolate(self, T, E=1.0, e_reorg=0.1,):
        vdos = get_vdos(T)
        Js = MO_couplings(self.M, self.dE, self.dR, vdos)
        rates = kMarcus(self.dE, self.dR, E, e_reorg)
        rng = default_rng()
        return t_percolate(self.sites, self.L, self.R,rng)

def diff_array(arr):
    return arr[:,None] - arr[None,:]

#@njit
def hop_step(i,rates,sites):
    N = sites.shape[0]
    hop_times = -np.log(1 - np.random.rand(N))/rates[i,:]
    return np.argmin(hop_times)

#@njit
def kMarcus(ediffs,rdiffs,Js,T,E=1.0,e_reorg=0.1):
    kB = 8.617e-5
    hbar = 6.582e-16
    A = 4 * e_reorg * kB * T
    e = 1.602e-19
    return 2 * np.pi * Js * np.exp(-((e_reorg - ediffs + e * np.dot(rdiffs,E))**2)/A) / (hbar * np.sqrt(np.pi * A))

#@guvectorize([(float64[:], float64[:], float64, float64[:,:], float64, float64, float64[:,:])], '(n),(n),(),(n,n),(),() -> (n,n)', nopython=True)
def kMarcus_gu(energies,xpos,e_reorg,Js,T,E, out):
    kB = 8.617e-5
    hbar = 6.582e-16
    A = 4 * e_reorg * kB * T
    e = 1.602e-19
    N = energies.shape[0]
    for i in range(N):
        for j in range(N):
            out[i,j] = 2 * np.pi * Js[i,j] * np.exp(-(e_reorg + energies[j] - energies[i] - e * E * (xpos[j] - xpos[i]))**2/A) / (hbar * np.sqrt(np.pi * A))

#@vectorize([float64(float64,float64)], nopython=True)
def bose_einstein(e,T):
    kB = 8.617e-5
    return 1.0/(np.exp(e/(kB*T)) - 1)


def MO_couplings(M, ediffs, vdos, T, A=1.0):
    S = np.abs((M.T) @ M)
    D = np.interp(ediffs, *vdos)
    return (A**2) * (S**2) * (bose_einstein(ediffs) + 1) * D / ediffs

#@njit
def t_percolate(sites, L, R, rates, rng):
    site = rng.choice(L)
    t = 0
    while site not in R:
        site = hop_step(site, rates, sites)
        t+=1
    return t