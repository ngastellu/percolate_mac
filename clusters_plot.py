#!/usr/bin/env python

from os import path
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from qcnico import plt_utils
from percolate import plot_cluster_brute_force
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

datadir=path.expanduser("~/Desktop/simulation_outputs/percolation/40x40/percolate_output")

posdir = path.join(path.dirname(datadir), 'structures')
Mdir = path.join(path.dirname(datadir), 'MOs_ARPACK')
edir = path.join(path.dirname(datadir), 'eARPACK')

plt_utils.setup_tex()
rcParams['font.size'] = 20

nn = 99
for T in [200,400]:
    posfile = path.join(posdir,f'bigMAC-{nn}_relaxed.xsf')
    Mfile = path.join(Mdir,f'MOs_ARPACK_bigMAC-{nn}.npy')
    efile = path.join(edir, f'eARPACK_bigMAC-{nn}.npy')
    ccfile = path.join(datadir,f'sample-{nn}','cc.npy')
    iifile = path.join(datadir,f'sample-{nn}','ii.npy')
    M = np.load(Mfile)
    centres = np.load(ccfile)
    MOinds = np.load(iifile)
    energies = np.load(efile)
    dat = get_data(nn,T,datadir)
    c = dat[0][0]
    print(c)
    A = dat[2]
    pos, _ = read_xsf(posfile)
    fig, ax = plt.subplots()
    plot_cluster_brute_force(c,pos,M,A,show_densities=True, dotsize=2.0, usetex=True, show=False, centers=centres, inds=MOinds,rel_center_size=10.0,plt_objs=(fig,ax))
    ax.set_title('Typical spanning cluster')
    plt.show()
