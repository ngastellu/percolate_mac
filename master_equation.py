#!/usr/bin/env python

import numpy as np
from numba import njit, vectorize
# from monte_carlo import kMillerAbrahams

"""This script contains a set of functions to solve the master equation describing steady-state
charge hopping in disordered systems. The implicit iteration method is shamelessly stolen from 
https://journals.aps.org/prb/abstract/10.1103/PhysRevB.63.085202"""

@vectorize
def fermi_dirac(E,T):
    kB = 8.617e-5
    beta = 1.0/(kB*T)
    return 1.0/(np.exp(beta*E))

def initialise(energies, positions, T, E):
    """Initialise the occupation probabilities using a Fermi-Dirac distribution (accoundting for E-field).
       Sites and positions are those of the hopping sites, not of the MOs!
    """
    e = 1.0 # charge of electron
    return fermi_dirac(energies - e*np.dot(E,positions), T)

@njit
def iterate_implicit(Pold,rates):
    """Carry out an implicit iteration step.
    
    Parameters
    ----------
    Pold: np.ndarray; shape = (N,)
        Site occupation probabilities from previous iteration step.
    rates: np.ndarray; shape = (N,N) 
        ASYMMETRIC hopping rate matrix, where rates[i,j] = hopping rate i --> j.
    
    Outputs
    -------
    Pnew: np.ndarray, shape = (N,)
        New site occupation matrix
    """
    norm = np.sum(rates,axis=1)
    N = rates.shape[0]
    Pnew = np.zeros(N, dtype='float')
    for i in range(N):
        sum_top = Pnew[:i]*rates[:i,i] + Pold[i:]*rates[i:,i]
        sum_bot = (rates[:i,i] - rates[i,:i])*Pnew[:i] + (rates[i:,i] - rates[i,i:])*Pold[i:] 
        Pnew[i] = (sum_top / norm[i]) / (1 - (sum_bot/norm[i]))
    
    return Pnew


@njit
def solve(Pinit, rates, maxiter=1e6,eps=1e-10):
    cntr = 0
    Pold = Pinit
    deltaP = eps + 1
    conv = np.zeros(maxiter,dtype='float')
    while cntr < maxiter and np.any(deltaP > eps):
        Pnew = iterate_implicit(Pold, rates)
        deltaP = np.abs(Pnew - Pold)
        conv[cntr] = deltaP
        cntr += 1
    return Pnew, conv

@njit
def miller_abrahams_asymm(energies, positions, T, E, a=30):
    e = 1.0
    kB = 8.617e-5
    beta = 1.0/(kB*T)
    N = energies.shape[0]
    energies -= e*np.dot(E,positions)
    diff_e = energies[None,:] - energies[:,None]
    diff_e[diff_e < 0] = 0
    dists = np.linalg.norm(pos[None,:] - pos[:,None],axis=2)
    return np.exp(-2*dists/a) * np.exp(-(beta/2)*(diff_e))






    
    