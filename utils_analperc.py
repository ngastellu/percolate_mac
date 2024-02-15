#!/usr/bin/env python

from os import path
import pickle
import numpy as np
from scipy.stats import linregress


kB = 8.617333262e-5 #eV/K

def get_dcrits(run_inds,temps,datadir):
    nsamples = len(run_inds)
    ntemps = len(temps)
    dcrits = np.zeros((nsamples,ntemps))
    for k in range(nsamples):
        for l in range(ntemps):
            sampdir = f"sample-{run_inds[k]}"
            pkl = f"out_percolate-{temps[l]}K.pkl"
            fo = open(path.join(datadir,sampdir,pkl),'rb')
            dat = pickle.load(fo)
            dcrits[k,l] = dat[1]
            fo.close()

    return dcrits


def saddle_pt_sigma(dcrits,nbins=30):
    sigmas = np.zeros(dcrits.shape[1])
    for k,ds in enumerate(dcrits.T):
        hist, bin_edges = np.histogram(ds,bins=nbins,density=True)
        bin_inds = np.sum(ds[:,None] > bin_edges,axis=1) - 1
        f = hist[bin_inds] * np.exp(-ds)
        sigmas[k] = np.max(f)
    return sigmas

def arrhenius_fit(T, sigma, w0=1e11, inv_T_scale=1.0, return_for_plotting=False):
    x = np.log(inv_T_scale / T)
    y = np.log(w0*sigma/(kB*T))
    slope, intercept, r, *_ = linregress(x,y)
    print('r^2 of Arrhenius fit = ', r**2)

    if return_for_plotting:
        return slope, intercept, x, y
    else:
        Ea = -slope * kB * inv_T_scale #activation energy in eV
        return Ea, intercept
    
def mott_fit(T, sigma, d=2, w0=1e11, return_for_plotting=False): 
    x = np.power(1.0/T,(d+1))
    y = np.log(w0*sigma/(kB*T))
    slope, intercept, r, *_ = linregress(x,y)
    print(f'r^2 of {d}D Mott fit = ', r**2)

    if return_for_plotting:
        return slope, intercept, x, y
    else:
        T0 = slope**(d+1)
        return T0, intercept