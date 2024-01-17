#!/usr/bin/env python

import numpy as np
from numpy.random import default_rng
import pickle
from os import path


"""This script verifies if the activation energy for charge hopping across our 40x40nm MAC samples
converges to a well-defined value as we gradually increase the number of samples in our analysis."""

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
            dcrits[k,l] = dat[1] + np.log(1e11) #add log(1e11) to set omega0=1e11 in MA hop rates
            fo.close()

    return dcrits


def get_sigma(dcrits,nbins=30):
    hist, bin_edges = np.histogram(dcrits,bins=nbins,density=True)
    bin_inds = np.sum(dcrits[:,None] > bin_edges,axis=1) - 1
    f = hist[bin_inds] * np.exp(-dcrits)
    return np.max(f)


rng = default_rng()

datadir=path.expanduser("~/Desktop/simulation_outputs/percolation/40x40/percolate_output/100x100_gridMOs/")
fgood_runs = path.join(datadir, 'good_runs.txt')
with open(fgood_runs) as fo:
    lines = fo.readlines()

gr_inds = list(map(int,[line.rstrip().lstrip() for line in lines]))

temps = np.arange(40,440,10)

nstructures = len(gr_inds)
print(nstructures)

dcrits = get_dcrits(gr_inds,temps, datadir)

nsamples = np.arange(10,120,5)
for n in nsamples:
    print(n)
    sample_inds = rng.choice(range(nstructures),size=n,replace=False)
    dcrits_sampled = dcrits[sample_inds,:]
    sigmas = [get_sigma(dc) for dc in dcrits_sampled.T]
    data = np.vstack((temps, sigmas))
    np.save(f'data_100x100_gridMOs/sigma_v_T-{n}samples.npy',data)
