#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from numba import njit, vectorize
from qcnico.plt_utils import setup_tex
# from monte_carlo import kMillerAbrahams

"""This script contains a set of functions to solve the master equation describing steady-state
charge hopping in disordered systems. The implicit iteration method is shamelessly stolen from 
https://journals.aps.org/prb/abstract/10.1103/PhysRevB.63.085202"""

@vectorize
def fermi_dirac(E,T):
    kB = 8.617e-5
    beta = 1.0/(kB*T)
    return 1.0/(np.exp(beta*E))

@njit
def initialise(energies, positions, T, E):
    """Initialise the occupation probabilities using a Fermi-Dirac distribution (accoundting for E-field).
       Sites and positions are those of the hopping sites, not of the MOs!
    """
    e = 1.0 # charge of electron
    return fermi_dirac(energies - e*np.dot(E,positions), T)

@njit
def miller_abrahams_asymm(energies, pos, T, E, a=30):
    """Compute asymmetric Miller-Abrahams rate matrix to describe hopping between sites. Asymmetry
    comes from the fact that if E_i - E_j < 0, no energy dependence is assumed in the i --> j hopping
    rate.
    """
    e = 1.0
    kB = 8.617e-5
    beta = 1.0/(kB*T)
    N = energies.shape[0]
    energies -= e*np.dot(E,pos)
    diff_e = energies[None,:] - energies[:,None]
    # diff_e[diff_e < 0] = 0
    mask = (diff_e < 0).nonzero()
    for i, j in mask:
        print(i,j)
        diff_e[i,j] = 0
    # dists = np.linalg.norm(pos[None,:] - pos[:,None],axis=2)
    dR = pos[None,:] - pos[:,None]
    dists = np.zeros((N,N),dtype='float')
    for i in range(N):
        for j in range(N):
            dists[i,j] = np.linalg.norm(dR[i,j])
    return np.exp(-2*dists/a) * np.exp(-(beta/2)*(diff_e))

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
        sum_top = 0
        sum_bot = 0
        for j in range(i):
            sum_top += Pnew[j]*rates[j,i]
            sum_bot += (rates[j,i] - rates[i,j])*Pnew[j]
        for j in range(i,N):
            sum_top += Pold[j]*rates[j,i]
            sum_bot += (rates[j,i] - rates[i,j])*Pold[j]
        Pnew[i] = sum_top
        Pnew[i] = Pnew[i] / (1 - sum_bot/norm[i])

    
    return Pnew


@njit
def solve(Pinit, rates, maxiter=1e6,eps=1e-10):
    maxiter = int(maxiter) #doin this so Numba stops bein a lil btch
    cntr = 0
    Pold = Pinit
    converged = False
    conv = np.zeros((maxiter,2),dtype='float')
    while cntr < maxiter and (not converged):
        Pnew = iterate_implicit(Pold, rates)
        deltaP = np.abs(Pnew - Pold)
        conv[cntr,0] = np.max(deltaP)
        conv[cntr,1] = np.mean(deltaP)
        converged = np.all(deltaP < eps)
        cntr += 1
    return Pnew, conv

@njit
def inner_vel_loop(rates_ij, occs_i, occs_j, diff):
    "Writing this as a separate function to get Numba to compile `carrier_velocity`."
    return rates_ij * occs_i * (1 - occs_j) * diff

@njit
def carrier_velocity(rates, occs, pos):
    N = rates.shape[0]
    v = np.zeros(pos.shape[1],dtype='float')
    for i in range(N):
        # r = pos[i,:]
        # p = occs[i]
        for j in range(N):
            dR = (pos[j] - pos[i]) # using r_j - r_i bc we assume charge carriers are < 0
            v += inner_vel_loop(rates[i,j],occs[i],occs[j], dR)
        # v += np.sum(rates[i,:]*occs[i]*(1-occs)*(pos - r),axis=0)
    
    return v

@njit
def run(energies, pos, temps, E):
    velocities = np.zeros((temps.shape[0],3))
    for k, T in enumerate(temps):
        print(T)
        K = miller_abrahams_asymm(energies, pos, T, E)
        # np.save("/Users/nico/Desktop/simulation_outputs/percolation/40x40/miller_abrahams_asymm/MA_asymm")
        P0 = initialise(energies,pos,T,E)
        Pfinal = solve(P0,K)
        velocities[k] = carrier_velocity(K,Pfinal,pos)
    return velocities


# -------- MAIN ----------

nsample = 150

percolate_datadir = f'/Users/nico/Desktop/simulation_outputs/percolation/40x40/percolate_output/sample-{nsample}/'
M = np.load(f'/Users/nico/Desktop/simulation_outputs/percolation/40x40/MOs_ARPACK/MOs_ARPACK_bigMAC-{nsample}.npy')
eMOs = np.load(f'/Users/nico/Desktop/simulation_outputs/percolation/40x40/eARPACK/eARPACK_bigMAC-{nsample}.npy')
strucdir = '/Users/nico/Desktop/simulation_outputs/percolation/40x40/structures/'


centers = np.load(percolate_datadir + 'cc.npy')
site_energies = np.load(percolate_datadir + 'ee.npy')
site_inds = np.load(percolate_datadir + 'ii.npy')

E = np.array([1,0,0])/400

temps = np.arange(100,450,50,dtype=np.float64)

velocities = run(site_energies,centers,temps,E)

print(velocities)

setup_tex()

plt.plot(1000/temps,np.log(np.linalg.norm(velocities,axis=1)/np.linalg.norm(E)),'r-')
plt.xlabel('$1000/T$ [mK$^{-1}$]')
plt.ylabel('$\mu$ [\AA$^2$/Vs]')
plt.show()

plt.plot((1.0/temps)**(1/3),np.log(np.linalg.norm(velocities,axis=1)/np.linalg.norm(E)),'r-')
plt.xlabel('$1000/T$ [mK$^{-1}$]')
plt.ylabel('$\mu$ [\AA$^2$/Vs]')
plt.show()

