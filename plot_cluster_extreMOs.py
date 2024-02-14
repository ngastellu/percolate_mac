#!/usr/bin/env python

from os import path
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from qcnico import plt_utils
from percolate import plot_cluster_brute_force, diff_arrs
from qcnico.coords_io import read_xsf
from qcnico.qchemMAC import MO_com
from qcnico.remove_dangling_carbons import remove_dangling_carbons

def read_pkl(run_ind,temp,datadir):
    #nsamples = len(run_inds)
    #ntemps = len(temps)
    sampdir = f"sample-{run_ind}"
    pkl = f"out_percolate-{temp}K.pkl"
    fo = open(path.join(datadir,sampdir,pkl),'rb')
    dat = pickle.load(fo)
    fo.close()
    return dat

def check_dists(A, centres, energies, dcrit, T, a0=30):
    print("Checking distances... dcrit = ", dcrit)
    kB = 8.617e-5
    ii, jj = A.nonzero()
    print(A.shape)
    for i, j in zip(ii,jj):
        dist = (np.abs(energies[i]) + np.abs(energies[j]) + np.abs(energies[i]-energies[j]))/(kB*T) + 2*np.linalg.norm(centres[i]-centres[j])/a0
        if dist > dcrit:
            print(f"!!! YIKES: u({i}, {j}) = {dist} !!!")


datadir=path.expanduser("~/Desktop/simulation_outputs/percolation/40x40")
rCC = 1.8

mo_type = 'lo'

posdir = path.join(datadir, 'structures')
Mdir = path.join(datadir, f'MOs_ARPACK/{mo_type}/')
edir = path.join(datadir, f'eARPACK/{mo_type}/')
percdir = path.join(datadir, f'percolate_output/extremal_MOs/{mo_type}50')

plt_utils.setup_tex()
rcParams['font.size'] = 20

for nn in [3,10,11,16]:
    for T in [100,200,300,400]:
        print(f'***** {T} *****')
        posfile = path.join(posdir,f'bigMAC-{nn}_relaxed.xsf')
        Mfile = path.join(Mdir,f'MOs_ARPACK_{mo_type}_bigMAC-{nn}.npy')
        efile = path.join(edir, f'eARPACK_{mo_type}_bigMAC-{nn}.npy')
        M = np.load(Mfile)
        energies = np.load(efile)
        pos = remove_dangling_carbons(read_xsf(posfile)[0],rCC)
        dat = read_pkl(nn,T,percdir)
        clusters = dat[0]
        print('Number of clusters = ', len(clusters))
        c = np.array(list(clusters[0]))
        centres = MO_com(pos,M)
        dcrit = dat[1]
        print(c)
        A = dat[2]
        print('dcrit = ', dcrit)
        check_dists(A,centres,energies,dcrit,T)
        fig, ax = plt.subplots()
        plot_cluster_brute_force(c,pos,M,A,show_densities=True, dotsize=2.0, usetex=True, show=False,rel_center_size=10.0,plt_objs=(fig,ax))
        ax.set_title('Typical spanning cluster')
        plt.show()

