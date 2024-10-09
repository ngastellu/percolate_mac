#!/usr/bin/env python

import os
import sys
from time import perf_counter
import numpy as np
from MO_crystallinity import MOs_crystallinity
from var_a_percolate_mac.utils_analperc import get_conduction_clusters
from qcnico.data_utils import save_npy


def condsites_v_MOs_crystallinity(S,M,ii,cluster,cryst_mask,cryst_renormalise=False):

    cluster = list(cluster)
    cluster_sites = S[:,cluster]
    iMOs = ii[cluster]
    cluster_MOs = M[:,iMOs]

    MO_crysts = MOs_crystallinity(cluster_MOs,cryst_mask,cryst_renormalise=cryst_renormalise)
    site_crysts = MOs_crystallinity(cluster_sites,cryst_mask,cryst_renormalise=cryst_renormalise)

    return np.vstack((MO_crysts, site_crysts))





ensemble = os.path.basename(os.getcwd())
n = sys.argv[1]
motype = sys.argv[2]
cryst_renormalise = False

print(f'*** Doing sample {n} ***')


if ensemble == '40x40':
    rmax = 18.03
elif ensemble == 'tempdot6':
    rmax = 121.2
elif ensemble == 'tempdot5':
    rmax = 198.69
else:
    print(f'{ensemble} is an invalid ensemble name.')

run_name = f'rmax_{rmax}_sites_gammas_{motype}'
temps = np.arange(180,440,10)



sites_datadir = f'sample-{n}/sites_data_{motype}/'
perc_datadir = f'sample-{n}/{run_name}_pkls/'

outdir = f'condsites_v_MOs_crystallinities_{run_name}/sample-{n}/'

if motype == 'virtual' or motype == 'occupied':
    M = np.load(os.path.expanduser(f'~/scratch/ArpackMAC/{ensemble}/MOs/{motype}/MOs_ARPACK_bigMAC-{n}.npy'))
else:
    M = np.load(os.path.expanduser(f'~/scratch/ArpackMAC/{ensemble}/MOs/{motype}/MOs_ARPACK_{motype}_{ensemble}-{n}.npy'))


N = M.shape[0]

S = np.load(sites_datadir + 'site_state_matrix.npy')
S /= np.linalg.norm(S,axis=0)
radii = np.load(sites_datadir + 'radii.npy')
ii = np.load(sites_datadir + 'ii.npy')

rfilter = radii < rmax
S = S[:,rfilter]
ii = ii[rfilter]

cryst_mask = np.load(os.path.expanduser(f'~/scratch/structural_characteristics_MAC/labelled_ring_centers/{ensemble}/sample-{n}/crystalline_atoms_mask-{n}.npy'))

for T in temps:
    print(f'Doing T = {T}K...', end=' ')
    start = perf_counter()
    cluster = get_conduction_clusters(perc_datadir,f'out_percolate_{run_name}',T)
    if len(cluster) == 1:
        cluster = cluster[0]
    else:
        print(f'found {len(cluster)} percolating clusters!', end=' ')
        cluster = cluster[0].union(*cluster[1:])
    end = perf_counter()

    print(cluster)
    print(f'Done! [{end-start} seconds]')

    out_npy = f'csc_v_Mc-{T}K.npy'
    condsites_v_MO_crysts = condsites_v_MOs_crystallinity(S,M,ii,cluster,cryst_mask,cryst_renormalise=cryst_renormalise)
    save_npy(condsites_v_MO_crysts, out_npy, outdir)
