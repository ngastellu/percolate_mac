#!/usr/bin/env python


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from utils_analperc import ablation_Ea, get_dcrits, saddle_pt_sigma
from qcnico.plt_utils import setup_tex, get_cm


def arrhenius_law(T,Ea,A):
    kB = 8.6173332e-5 # eV/K
    return A * np.exp(-Ea/(kB*T))

def sigma_errorbar(dcrits,temps):
    """IDEA: Estimate uncertainty in sigma(T) for each T by omitting one structure from the data set
     before computing sigma, and taking the difference between sigma obtained from this reduced set
     and sigma the computed from the full data set. We cycle over all samples and keep the highest 
     difference between the two estimates of sigma as our uncertainty.
     As always, dcrits is (Ns x Nt) array where Ns = number of structures, and
     Nt = number of temperatures."""
    
    assert dcrits.shape[1] == temps.shape[0], f"Mismatched shapes for dcrits {dcrits.shape} and temps {temps.shape}: 
    nb of columns of dcrits must match the nb of elements in temps!"

    nsamples = dcrits.shape[0]
    sigmas_full = saddle_pt_sigma(dcrits)
    err = np.zeros(temps.shape[0])

    # Get errors in sigma estimates
    for n in range(nsamples):
        sigmas_partial = saddle_pt_sigma(np.roll(dcrits, n, axis=0)[:-1])
        diffs = np.abs(sigmas_full - sigmas_partial)
        inew = (diffs > err).nonzero()[0]
        err[inew] = diffs[inew]
    
    return sigmas_full, err
    


    
    






temps = np.arange(40,440,10)[14:]

tdot25dir = '/Users/nico/Desktop/simulation_outputs/percolation/Ata_structures/tdot25/percolate_output/zero_field/virt_100x100_gridMOs/'
pCNNdir = '/Users/nico/Desktop/simulation_outputs/percolation/40x40/percolate_output/zero_field/100x100_gridMOs/'
t1dir = '/Users/nico/Desktop/simulation_outputs/percolation/Ata_structures/t1/percolate_output/zero_field/virt_100x100_gridMOs/'


tdot25_lbls = list(set(range(30)) - {0, 5, 8 , 18, 20, 21, 25, 26})

with open(pCNNdir + 'good_runs.txt') as fo:
    pCNN_lbls = [int(l.strip()) for l in fo.readlines()]

with open(t1dir + 'good_runs.txt') as fo:
    t1_lbls = [int(l.strip()) for l in fo.readlines()]


dd_rings = '/Users/nico/Desktop/simulation_outputs/ring_stats_40x40_pCNN_MAC/'

ring_data_tdot25 = np.load(dd_rings + 'avg_ring_counts_tdot25_relaxed.npy')
ring_data_t1 = np.load(dd_rings + 'avg_ring_counts_t1_relaxed.npy')
ring_data_pCNN = np.load(dd_rings + 'avg_ring_counts_normalised.npy')

p6c_tdot25 = ring_data_tdot25[3] / ring_data_tdot25.sum()
p6c_t1 = ring_data_t1[3] / ring_data_t1.sum()
p6c_pCNN = ring_data_pCNN[3]
p6c = np.array([p6c_tdot25, p6c_pCNN,p6c_t1])

clrs = get_cm(p6c, 'inferno')
ddirs = [tdot25dir, pCNNdir, t1dir]
run_lbls = [tdot25_lbls,pCNN_lbls,t1_lbls]
curve_lbls = ['$\\tilde{T} = 0.25$', 'PixelCNN', '$\\tilde{T} = 1$']

sample_sizes = [np.arange(len(lbls),len(lbls)-10,-1) for lbls in run_lbls]
Eas = np.zeros((3,10))

setup_tex()
fig, ax = plt.subplots()

for k, dd, ll, cc, cl, ss in zip(range(3), ddirs, run_lbls, clrs, curve_lbls, sample_sizes):    
    dcrits = get_dcrits(ll, temps, dd)
    print(dcrits.shape)
    nstrucs = len(dcrits)
    Eas[k,:] = ablation_Ea(dcrits,ss,temps=temps)
    Eas[k,:] *= 1000 # convert from eV to meV
    ax.plot(Eas[k],'o',c=cc,label=cl,ms=5.0)
    ax.plot(Eas[k],'-',c=cc,lw=0.8)

ax.set_xlabel('Nb. of ablated samples')
ax.set_ylabel('$\Delta E_a$ [meV]')
plt.legend()
plt.show()