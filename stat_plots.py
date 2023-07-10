#!/usr/bin/env python

from os import path
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from qcnico import plt_utils

def get_cluster_stats(run_inds,temps,datadir):
    nsamples = len(run_inds)
    ntemps = len(temps)
    avg_sizes = np.zeros((nsamples,ntemps),dtype=int)
    min_sizes = np.zeros((nsamples,ntemps),dtype=int)
    max_sizes = np.zeros((nsamples,ntemps),dtype=int)
    nclusters = np.zeros((nsamples,ntemps),dtype=int)
    for k in range(nsamples):
        #print(f"******* {run_inds[k]} *******")
        for l in range(ntemps):
            sampdir = f"sample-{run_inds[k]}"
            pkl = f"out_percolate-{temps[l]}K.pkl"
            fo = open(path.join(datadir,sampdir,pkl),'rb')
            dat = pickle.load(fo)
            clusters = dat[0]
            #print(f"T = {temps[l]}K: ",clusters)
            nclusters[k,l] = len(clusters)
            if len(clusters) == 0:
                sizes = 0
                print(run_inds[k], temps[l])
            else:
                sizes = np.array([len(c) for c in clusters])
            avg_sizes[k,l] = np.mean(sizes)
            min_sizes[k,l] = np.min(sizes)
            max_sizes[k,l] = np.max(sizes)
            fo.close()
            #print('\n')

    return nclusters, avg_sizes, min_sizes, max_sizes

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

datadir=path.expanduser("~/Desktop/simulation_outputs/percolation/40x40/percolate_output")
fgood_runs = path.join(datadir, 'good_runs.txt')
with open(fgood_runs) as fo:
    lines = fo.readlines()

gr_inds = list(map(int,[line.rstrip().lstrip() for line in lines]))


temps = np.arange(40,440,10)
start_ind = 14

dcrits = get_dcrits(gr_inds,temps, datadir)
print(dcrits.shape)

davg_T = np.mean(dcrits,axis=0)
dstd_T = np.std(dcrits,axis=0)
print(davg_T.shape)

nclusters, avgs, maxs, mins = get_cluster_stats(gr_inds,temps,datadir)

sizes = avgs

plt_utils.setup_tex()
rcParams['font.size'] = 20
rcParams['figure.dpi'] = 150.0 #increasre figure resolution
rcParams['figure.subplot.top'] = 0.97
rcParams['figure.subplot.bottom'] = 0.12    
rcParams['figure.subplot.left'] = 0.21   
rcParams['figure.subplot.right'] = 0.79  


fig, ax = plt.subplots()
ax.plot(temps,davg_T,'r-')
# ax.plot(temps,davg_T+dstd_T,'k--',lw=0.8)
# ax.plot(temps,davg_T-dstd_T,'k--',lw=0.8)
ax.set_xlabel('T [K]')
ax.set_ylabel('$\langle u_c\\rangle$')
plt.show()

Tcm = plt_utils.get_cm(temps[start_ind:],'coolwarm',max_val=1.0)

plot_inds = [0,12, 25]
print(plot_inds)

fig, ax = plt.subplots()

for n in plot_inds:
    plt_utils.histogram(dcrits[:,n+start_ind],nbins=25,show=False, normalised=False, plt_objs=(fig,ax),
        plt_kwargs={'alpha': 0.95, 'color': Tcm[n], 'label': f'$T = {temps[n+start_ind]}$K'})
ax.set_xlabel('$\langle u_{c}\\rangle$')
plt.legend()
plt.show()

rcParams['figure.subplot.left'] = 0.35
rcParams['figure.subplot.top'] = 0.95
rcParams['figure.subplot.bottom'] = 0.15

fig , axs = plt.subplots(3,1,sharex=True)

for k,n in enumerate(plot_inds):
    print(n)
    plt_utils.histogram(sizes[:,n],nbins=30,show=False, normalised=False, plt_objs=(fig,axs[k]),
        plt_kwargs={'alpha': 1.0, 'color': Tcm[n], 'label': f'$T = {temps[start_ind+n]}$K'})
    axs[k].legend()
axs[-1].set_xlabel('Cluster size')

plt.show()