#!/usr/bin/env python

from os import path
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.animation as animation
from qcnico import plt_utils
from percolate import pair_inds, dArray_logMA, diff_arrs
from qcnico.coords_io import read_xsf

def get_data(run_ind,temp,datadir):
    #nsamples = len(run_inds)
    #ntemps = len(temps)
    sampdir = f"sample-{run_ind}"
    pkl = f"out_percolate-{temp}K.pkl"
    fo = open(path.join(datadir,sampdir,pkl),'rb')
    dat = pickle.load(fo)
    fo.close()
    return dat


def remove_singles(pairs,c):
    """This function weeds out all centres who are only connected to themselves (This happens 2 MOs that can map
    to a single hopping centre)."""
    npairs = pairs.shape[0]
    keep = np.zeros(npairs,dtype=bool) #initialise to all False
    for k, p in enumerate(pairs):
        r1, r2 = p
        if np.all(r1 == r2) or (r1 not in cc):
            continue
        else:
            keep[k] = True
    return pairs[keep] # keep only pairs of DISTINCT centers



def get_center_pairs():
    """Generator function that gets the hopping centers and the hopping center pairs to be plotted/connected
    at each frame."""
    N = centres.shape[0]
    k=0
    # while True:
    for d in dgen:
        # d = next(dgen)
        print('k = ', k)
        connected_inds = (darr < d).nonzero()[0] #darr is 1D array     
        ii, jj = pair_inds(connected_inds,N)
        relevant_pairs = np.array([(centres[i], centres[j]) for i,j in zip(ii,jj)])
        relevant_pairs = remove_singles(relevant_pairs, c)
        yield relevant_pairs
        k += 1


def update(frame):
    """In this animation the frame is defined by which hopping site pairs are connected for a certain threshold 
    distance d; those sites are then plotted, along with the edges connecting them."""
    rel_pairs = frame
    npairs = frame.shape[0]
    rel_centers = np.unique(rel_pairs.reshape(2*npairs,2),axis=0)
    ye = ax.scatter(pos.T[0], pos.T[1], c=rho, s=1.0, cmap='plasma',zorder=1)

    ax.scatter(*rel_centers.T, marker='*', c='r', s=20.0, zorder=2)

    for r1, r2 in rel_pairs:
        pts = np.vstack((r1,r2)).T
        ax.plot(*pts, 'r-', lw=1.0)
    
    return ye


# Get data
datadir=path.expanduser("~/Desktop/simulation_outputs/percolation/40x40/percolate_output")

posdir = path.join(path.dirname(datadir), 'structures')
Mdir = path.join(path.dirname(datadir), 'MOs_ARPACK')
edir = path.join(path.dirname(datadir), 'eARPACK')


nn = 99
T = 200
kB = 8.617e-5
posfile = path.join(posdir,f'bigMAC-{nn}_relaxed.xsf')
Mfile = path.join(Mdir,f'MOs_ARPACK_bigMAC-{nn}.npy')
efile = path.join(edir, f'eARPACK_bigMAC-{nn}.npy')
ccfile = path.join(datadir,f'sample-{nn}','cc.npy')
iifile = path.join(datadir,f'sample-{nn}','ii.npy')
eefile = path.join(datadir,f'sample-{nn}','ee.npy')


M = np.load(Mfile)
centres = np.load(ccfile)
MOinds = np.load(iifile)
energies = np.load(eefile)
pos, _ = read_xsf(posfile)
dat = get_data(nn,T,datadir)
c = np.sort(list(dat[0][0])) #sort cluster inds
cc = centres[c]
cluster_centres_inds = np.array([np.sum(i > c) for i in c])
cluster_centres = centres[cluster_centres_inds]
dcrit = dat[1]
print(dcrit)

# Precompute necessary qties

edArr, rdArr = diff_arrs(energies, centres, a0=30, eF=0)
darr = rdArr + (edArr/(kB*T))
print(darr)
#darr = dArray_logMA(energies, centres, T, a0=30,eF=0)
print(darr[darr<=dcrit].shape)
dgen = np.sort(np.unique(darr[darr <= dcrit]))
print(dgen.shape)
# dgen = iter(np.hstack((dgen[::10], dgen[-2:]))[2:])
dgen = iter(np.hstack((dgen[::5], dgen[-2:]))[1:])
rho = np.sum(M[:,np.unique(MOinds)]**2,axis=1)




plt_utils.setup_tex()




rcParams['font.size'] = 20
rcParams['figure.figsize'] = [10,5]
rcParams['figure.dpi'] = 200.0
rcParams['figure.constrained_layout.use'] = True
fig, ax = plt.subplots()
ye = ax.scatter(pos.T[0], pos.T[1], c=rho, s=1.0, cmap='plasma',zorder=1)
# cbar = fig.colorbar(ye,ax=ax,orientation='vertical')
ax.set_aspect('equal')
ax.set_xlabel("$x$ [\AA]")
ax.set_ylabel("$y$ [\AA]")

ani = animation.FuncAnimation(fig=fig,func=update,frames=get_center_pairs,repeat=False)
ani.save(filename='cluster.gif',writer='pillow')
plt.show()
