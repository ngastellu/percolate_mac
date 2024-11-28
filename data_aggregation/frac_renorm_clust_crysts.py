#/usr/bin/env python

import numpy as np
import os
from qcnico.data_utils import save_npy

def get_Natoms_xyz(xyz_file):
    """Reads the number of atoms from an XYZ file"""
    with open(xyz_file) as fo:
        N = int(fo.readline().strip())
    return N


structypes = ['40x40', 'tempdot6', 'tempdot5']
synth_temps = ['500', 'q400', '300']
rmaxs = [18.03, 121.2, 198.69]
motypes = ['virtual', 'lo', 'hi']
temps = np.arange(180, 440, 10)


for st, t, r in zip(structypes, synth_temps, rmaxs):
    print(f'---------- {st} ----------')
    strucdir = f'/Users/nico/Desktop/simulation_outputs/MAC_structures/relaxed_no_dangle/sAMC-{t}'
    for mt in motypes:
        print(f'*** {mt} ***')
        clust_cryst_indir = f'/Users/nico/Desktop/simulation_outputs/percolation/{st}/electronic_crystallinity/cluster_crystallinities_sites_gammas/cluster_crystallinities_rmax_{r}_sites_gammas_{mt}_renormd_by_ncryst_atoms/'
        nn = [int(d.split('-')[1]) for d in os.listdir(clust_cryst_indir)]
        for n in nn:
            if st == '40x40' and n == 131:
                continue
            print(f'{n}:', end=' ')
            clust_cryst_outdir = f'/Users/nico/Desktop/simulation_outputs/percolation/{st}/electronic_crystallinity/cluster_crystallinities_sites_gammas/cluster_crystallinities_rmax_{r}_sites_gammas_{mt}_frac_renormd/sample-{n}'
            xyz_file = os.path.join(strucdir, f'sAMC{t}-{n}.xyz')
            N = get_Natoms_xyz(xyz_file)
            for T in temps:
                print(T, end = ' ')
                try:
                    cc = np.load(os.path.join(clust_cryst_indir,f'sample-{n}',f'clust_cryst-{T}K.npy'))
                except FileNotFoundError as e:
                    print('<-- missing NPY!   ')
                    continue
                cc_frac_renormd = cc * N
                save_npy(cc_frac_renormd,f'clust_cryst-{T}K.npy', clust_cryst_outdir)
            print('\n')