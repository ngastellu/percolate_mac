#!/usr/bin/env python

from os import path
import pickle
import numpy as np
from numpy.random import default_rng
from scipy.stats import linregress


# Global consts usefful to calculate conductivity

kB = 8.617333262e-5 #eV/K
e2C = 1.602177e-19 # elementary charge to Coulomb
w0 = 1e15
# This factor combines the hop frequency with the unit conversion (to yield conductivity in siemens)
# w0 is chosen such that the final result matches the results from the AMC paper.
conv_factor = e2C*w0

def get_dcrits(run_inds,temps,datadir, pkl_prefix='out_percolate'):
    nsamples = len(run_inds)
    ntemps = len(temps)
    dcrits = np.zeros((nsamples,ntemps))
    for k in range(nsamples):
        for l in range(ntemps):
            sampdir = f"sample-{run_inds[k]}"
            try:
                pkl = pkl_prefix + f"-{temps[l]}K.pkl"
                fo = open(path.join(datadir,sampdir,pkl),'rb')
            except FileNotFoundError:
                pkl = pkl_prefix + f"-{temps[l]}.pkl"
                fo = open(path.join(datadir,sampdir,pkl),'rb')
            dat = pickle.load(fo)
            dcrits[k,l] = dat[1]
            fo.close()

    return dcrits

def load_dcrits(run_inds,run_name):
    """Meant to be run on Narval"""
    nsamples = len(run_inds)
    data = np.load(f'sample-{run_inds[0]}/{run_name}_pkls/dcrits.npy')
    temps = data[0,:]
    dcrits = np.zeros((nsamples,temps.shape[0]))
    dcrits[0] = data[1,:]
    success = np.ones(nsamples,dtype='bool')
    for k, nn in enumerate(run_inds[1:]):
        print(f'\n{nn}', end =' ')
        datadir = f"sample-{nn}/{run_name}_pkls/"
        try:
            dcrits[k+1,:] = np.load(path.join(datadir,f'dcrits.npy'))[1,:]
        except:
            print('NPY missing!')
            success[k+1] = False
            continue
    return temps, dcrits[success,:]

            


def saddle_pt_sigma(dcrits,temps,nbins=30):
    """Extracts the sigma(T) curve from a set of critical distances `dcrits`,
    a (Ns x Nt) array (Ns = nb. of samples, Nt = nb. of temperatures).
    The estimate is done using the saddle-point approximation to carry out the integral which
    yields sigma (c.f. Rodin, Fogler, PHYSICAL REVIEW B 84, 125447 (2011))"""

    sigmas = np.zeros(dcrits.shape[1])
    for k,ds in enumerate(dcrits.T):
        hist, bin_edges = np.histogram(ds,bins=nbins,density=True)
        bin_inds = np.sum(ds[:,None] > bin_edges,axis=1) - 1
        f = hist[bin_inds] * np.exp(-ds)
        sigmas[k] = np.max(f) * conv_factor / (kB*temps[k])
    return sigmas

def rough_integral_sigma(dcrits,temps,nbins=30):
    """Extracts the sigma(T) curve from a set of critical distances `dcrits`,
    a (Ns x Nt) array (Ns = nb. of samples, Nt = nb. of temperatures), using a very
    rough integration scheme: bin the dcrits into a histogram of `nbins` bins, and
    let h_i and x_i respectively denote the height and center of the ith bin,
    we then approximate sigma as follows: sigma = \int du e^{-u}*P(u) ~ \sum_i h_i*dx_i*e^{-x_i}"""
    sigmas = np.zeros(dcrits.shape[1])
    for k, ds in enumerate(dcrits.T):
        hist, bin_edges = np.histogram(ds,bins=nbins,density=True)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        dx = bin_edges[1:] - bin_edges[:-1]
        sigmas[k] = np.sum(hist*np.exp(-bin_centers)*dx) * conv_factor/(kB*temps[k])
    return sigmas
    


def arrhenius_fit(T, sigma, w0=1.0, inv_T_scale=1.0, return_for_plotting=False, x_start=0, x_end=None,return_err=False):
    x = inv_T_scale / T # T is sorted in increasing order --> 1/T is in decreasing order
    y = np.log(w0*sigma/(kB*T))
    
    if x_end is None:
        lr_out = linregress(x[x_start:], y[x_start:])
    else:
        lr_out = linregress(x[x_start:x_end],y[x_start:x_end])

    slope = lr_out.slope
    intercept = lr_out.intercept
    stderr = lr_out.stderr
    stderr_intercept = lr_out.intercept_stderr
    r = lr_out.rvalue
    
    print('r^2 of Arrhenius fit = ', r**2)


    if return_for_plotting:
        if return_err:
            return slope, intercept, x, y, stderr, stderr_intercept
        else:
            return slope, intercept, x, y
    else:
        Ea = -slope * kB * inv_T_scale #activation energy in eV
        if return_err:
            return Ea, intercept, stderr
        else:
            return Ea, intercept
    
def mott_fit(T, sigma, d=2, w0=1e11, return_for_plotting=False, return_r2=False, x_start=0, x_end=None): 
    x = np.power(1.0/T,(d+1))
    y = np.log(w0*sigma/(kB*T))

    if x_end is None:
        slope, intercept, r, *_ = linregress(x[x_start:],y[x_start:])
    else:
        slope, intercept, r, *_ = linregress(x[x_start:x_end],y[x_start:x_end])
    print(f'r^2 of {d}D Mott fit = ', r**2)

    if return_for_plotting:
        if return_r2:
            return slope, intercept, x, y, r**2
        else:
            return slope, intercept, x, y
    else:
        T0 = slope**(d+1)
        if return_r2:
            return T0, intercept, r**2
        else: 
            return T0, intercept
    

def find_best_fit(T, sigma, fitfunc, scan_direction='down'):
    """Does the same thing as the `arrhenius_fit` or `mott_fit`, but scans the T range
    to get the best fit. 
    !!! N.B. For `mott_fit` with d!=2, must use functools.partial wrapper. !!!
    Kwarg `scan_direction` is up or down depending on which end of the T range you want to begin.
    """
    if scan_direction == 'down':
        T = T[::-1] #sort T in descending order and traverse T range
        sigma = sigma[::-1]
    k = 0
    r2 = 0.0
    better_fit= True

    while better_fit:
        r2prev = r2
        slope, intercept, x, y, r2 = fitfunc(T[k:],sigma[k:],return_for_plotting=True, return_r2=True)
        better_fit = r2 > r2prev
        k+=1
    print('Best fit found! r^2 = ', r2)
    if scan_direction == 'down':
        print(f'Fitting T <=  {T[k]} K')
    else:
        print(f'Fitting T >  {T[k]} K')
    return slope, intercept, x,y


def ablation_Ea(dcrits,sample_sizes,temps=np.arange(40,440,10)):
    """Performs an ablation study on the Ea estimates we get from our percolation clusters.
    `dcrits` is a (Ns x Nt) array (Ns = nb of MAC structures; Nt = nb of temperature values).
    The ablation study is carried out by only inclusing a random subset of MAC structures into the
    data set used to obtain Ea. The goal is to see how well-converged the """
    rng = default_rng()
    Eas = np.zeros(len(sample_sizes))
    N = len(dcrits)
    for k, n in enumerate(sample_sizes):
        sample_inds = rng.choice(N,size=n,replace=False)
        dc = dcrits[sample_inds,:]
        print('dc: ', dc.shape)
        sigmas = saddle_pt_sigma(dc)
        Eas[k], _ = arrhenius_fit(temps, sigmas, inv_T_scale=1000.0)
    return Eas

def get_conduction_data(datadir, pkl_prefix, T):
    try:
        pkl = pkl_prefix + f"-{T}K.pkl"
        fo = open(path.join(datadir,pkl),'rb')
    except FileNotFoundError:
        pkl = pkl_prefix + f"-{T}.pkl"
        fo = open(path.join(datadir,pkl),'rb')
    dat = pickle.load(fo)
    fo.close()
    return dat

def get_conduction_clusters(datadir, pkl_prefix, T):
    dat = get_conduction_data(datadir, pkl_prefix, T)
    clusters = dat[0]
    # Empty clusters usually arise because the max intersite distance is the percolating
    # threshold, and the percolation code stops right before looking at the last pair.
    # If clusters is empty, check that the number of connected sites = N*(N-1)/2 - 1.
    # If yes, the cluster is the set of all sites.
    # If not, there's something VERY weird going on.
    if len(clusters) == 0:
        print(f'Empty cluster found for T = {T}K!')
        A = dat[2]
        N = A.shape[0]
        nbonded_pairs = A.sum() / 2
        npairs_tot = N*(N-1) / 2
        print(f'# of connected pairs = ', nbonded_pairs)
        print(f'Total # of pairs = ', npairs_tot)
        if nbonded_pairs == npairs_tot - 1: # -1 bc last pair is missing
            print('Retruning all sites as cluster.')
            clusters = [np.arange(N)]
    return clusters

def conduction_mask(sites_mask, cluster):
    cluster_mask = sites_mask[list(cluster)]
    conduction_bools = cluster_mask.sum(axis = 0) # sum corresponds to the union of atom sets associated with each site in cluster
    return conduction_bools

def sigma_errorbar(dcrits,temps,sigma_func=rough_integral_sigma):
    """IDEA: Estimate uncertainty in sigma(T) for each T by omitting one structure from the data set
     before computing sigma, and taking the difference between sigma obtained from this reduced set
     and sigma the computed from the full data set. We cycle over all samples and keep the highest 
     difference between the two estimates of sigma as our uncertainty.
     As always, dcrits is (Ns x Nt) array where Ns = number of structures, and
     Nt = number of temperatures.
     
     `sigma_func` is the function that actually computes the conductance. For now, the only two valid 
     values are `rough_integral_sigma` (default) and `saddle_pt_sigma` (see above). New functions can be added, but be 
     careful with the arguments: as things stand this `sigma_func` must accept two positional args: `dcrits`
     and `temps`."""
    
    nsamples = dcrits.shape[0]
    sigmas_full = sigma_func(dcrits,temps)
    err = np.zeros(sigmas_full.shape[0])

    # Get errors in sigma estimates
    for n in range(nsamples):
        sigmas_partial = sigma_func(np.roll(dcrits, n, axis=0)[:-1], temps)
        diffs = np.abs(sigmas_full - sigmas_partial)
        inew = (diffs > err).nonzero()[0] #identify for which T inds we need to keep err 
        err[inew] = diffs[inew]
    
    return sigmas_full, err