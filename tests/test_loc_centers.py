
#!/usr/bin/env python

from os import path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from qcnico import plt_utils
from qcnico.qcplots import plot_MO
from qcnico.coords_io import read_xsf
from qcnico.qchemMAC import gridifyMO
from qcnico.remove_dangling_carbons import remove_dangling_carbons
from MOs2sites import get_MO_loc_centers

def plot_loc_centers(rho, xedges, yedges, centers, colours='r', show=True, plt_objs=None):

    if plt_objs is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = plt_objs
    
    ax.imshow(rho, origin='lower',extent=[*xedges[[0,-1]], *yedges[[0,-1]]],zorder=0)
    ax.scatter(*centers.T,c=colours,marker='*',s=5.0,zorder=2)
    if show:
        plt.show()
    else:
        return fig, ax

def gen_ARPACKdata(datadir,lbls):
    edir = path.join(datadir,'eARPACK', 'virtual')
    Mdir = path.join(datadir,'MOs_ARPACK', 'virtual')
    for n in lbls:
        ee = np.load(path.join(edir,f'eARPACK_{n}.npy'))
        MM = np.load(path.join(Mdir,f'MOs_ARPACK_{n}.npy'))
        yield ee, MM

def gen_pos(posdir,lbls):
    for n in lbls:
        yield remove_dangling_carbons(read_xsf(path.join(posdir,f"{n}_relaxed.xsf"))[0],rCC_MAC) 


datadir = path.expanduser("~/Desktop/simulation_outputs/percolation/40x40/")
posdir = path.join(datadir, "structures")

lbls = [f'bigMAC-{n}' for n in [2,3,5,6,7,9,10]]

rCC_MAC = 1.8

istruc = 4 
iMO = 0 

arpackgen = gen_ARPACKdata(datadir,lbls[:5])
posgen = gen_pos(posdir,lbls[:5])

for eM, pos in zip(arpackgen, posgen):
    e, M = eM
    # rcParams['font.size'] = 20.0
    # rcParams['figure.dpi'] = 200
    # rcParams['figure.figsize'] = [12,12]
    plt_utils.setup_tex()

    for iMO in range(5):

    # fig, ax = plt.subplots()

    # sizes = np.ones(pos.shape[0]) *  2.5 
    # sizes[psi > 0.001] *= 8.5 
    # sizes = 2.0 + np.exp(100*psi**2)
    # print(sizes.shape)

    # ye = ax.scatter(pos[:,0], pos[:,1], s=sizes, c=psi , cmap='plasma')
    # cbar = fig.colorbar(ye,ax=ax,orientation='vertical')
    # ax.set_xlabel('$x$ [\AA]')
    # ax.set_ylabel('$y$ [\AA]')
    # ax.set_aspect('equal')
    # com = psi @ pos
    # ax.scatter(com[0], com[1], s=2.0*20,marker='*',c='r')
    # plt.show()

        centers, rho, xedges, yedges = get_MO_loc_centers(pos,M,iMO,nbins=20,return_realspace=True,return_gridify=True)
        shifted_centers = get_MO_loc_centers(pos,M,iMO,nbins=20,return_realspace=True,return_gridify=False,shift_centers='random')

        fig, ax = plot_loc_centers(rho, xedges, yedges, centers, show=False)
        fig, ax = plot_loc_centers(rho, xedges, yedges, shifted_centers, show=False,plt_objs=(fig,ax),colours='limegreen')

        for r0,r1 in zip(centers, shifted_centers):
            ax.plot(*(np.vstack((r0,r1)).T),'orange','--',lw=0.8)

        plt.show()