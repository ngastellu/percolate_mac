#!/usr/bin/env python

import numpy as np
from qcnico.coords_io import read_xyz
from qcnico.qcplots import plot_MO
from qcnico.plt_utils import setup_tex
from MOs2sites import get_MO_loc_centers_opt
import matplotlib.pyplot as plt
import os


structype = '40x40'
synthtemp = '500'
istruc = 42
iMO = 0
motype = 'virtual'

fontsize_axes = 30
fontsize = 30
setup_tex(fontsize=fontsize)

if motype == 'virtual':
    Mfile_prefx = 'MOs_ARPACK_bigMAC-'
else:
    Mfile_prefx = f'MOs_ARPACK_{motype}_{structype}-'


posfile = os.path.expanduser(f'~/Desktop/simulation_outputs/MAC_structures/relaxed_no_dangle/sAMC-{synthtemp}/sAMC{synthtemp}-{istruc}.xyz')
pos = read_xyz(posfile)


Mfile = os.path.expanduser(f'~/Desktop/simulation_outputs/percolation/{structype}/MOs_ARPACK/{motype}/{Mfile_prefx}{istruc}.npy')
M = np.load(Mfile)




# ------- Plot MO with COM and  rgyr ------- 
fig, ax = plt.subplots()
fig.subplots_adjust(bottom=0.150,top=0.97)
plot_MO(pos, M, iMO, dotsize=1,scale_up=50.0,show_COM=True,show_rgyr=True,plt_objs=(fig,ax),usetex=False,show_title=False,show=True,c_rel_size=400,c_lw=2.5)



# ------- Plot coarse-grained MO with local maxes ------- 
peak_pos, rho, x_edges, y_edges = get_MO_loc_centers_opt(pos,M,iMO,threshold_ratio=0.30, min_distance=30.0)

rho = rho[1:-1,1:-1]

fig, ax = plt.subplots()
fig.subplots_adjust(bottom=0.150,top=0.97)
ye = ax.imshow(rho,origin='lower', extent=(0,400,0,400),cmap='plasma')

cbar = fig.colorbar(ye,ax=ax,orientation='vertical',label='Coarse-grained density $\\rho$')
for k, pp in enumerate(peak_pos):
    if k == 0:
        ax.scatter(pp[0], pp[1], marker='*', color='r',edgecolor='k',lw=0.7, s=150,label='Local maxima')
    else:
        ax.scatter(pp[0], pp[1], marker='*', color='r',edgecolor='k',lw=0.7, s=150)

ax.legend()


ax.set_xlabel('$x$ [\AA]', fontsize=fontsize)
ax.set_ylabel('$y$ [\AA]', fontsize=fontsize)

ax.tick_params('both',labelsize=fontsize_axes)

plt.show()

