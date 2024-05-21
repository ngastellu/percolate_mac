#!/usr/bin/env python

from os import path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.animation as animation
from qcnico import plt_utils
from qcnico.coords_io import read_xsf




# Get data
datadir=path.expanduser("~/Desktop/simulation_outputs/percolation/40x40/percolate_output")

nn = 6
T = 400
kB = 8.617e-5

posdir = path.join(path.dirname(datadir), 'structures')
Mdir = path.join(path.dirname(datadir), 'MOs_ARPACK')
edir = path.join(path.dirname(datadir), 'eARPACK')
trajdir = '/Users/nico/Desktop/simulation_outputs/percolation/40x40/monte_carlo/marcus/trajectories/'
trajfile = trajdir + f'traj-{T}K_nointra-{{i}}.npy'

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
itraj = np.load(trajfile)
traj = centres[itraj]

plt_utils.setup_tex()

fig, ax1 = plt.subplots()

t = np.arange(traj.shape[0])

yo = ax1.plot(*traj.T, '-', c='k', lw=0.5, alpha=0.5,zorder=1)
ye = ax1.scatter(*traj.T, c=t, cmap='inferno', s=30.0, label='$\langle\\bm{R}(t)\\rangle$',zorder=2)
# ax1.plot([0, 0], [0, maxY], 'k--', lw=1.0)
# ax1.plot([0, maxX], [maxY, maxY], 'k--', lw=1.0)
# ax1.plot([maxX, maxX], [0, maxY], 'k--', lw=1.0)
# ax1.plot([0, maxX], [0, 0], 'k--', lw=1.0)

cbar = fig.colorbar(ye, ax=ax1)

ax1.set_aspect('equal')
ax1.set_xlabel('$x$ [\AA]')
ax1.set_ylabel('$y$ [\AA]')
plt.show()

