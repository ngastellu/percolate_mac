#!/usr/bin/env python

import sys
import os
import numpy as np
from deploy_percolate import run_percolate, load_atomic_positions, VRH_sites



n = int(sys.argv[1])
struc_type = sys.argv[2]
mo_type = sys.argv[3]


if struc_type == 'sAMC-500':
  rmax = 18.03
elif struc_type == 'sAMC-q400':
  rmax = 136.47
elif struc_type == 'sAMC-300':
  rmax = 199.33
else:
  print(f'Structure type {struc_type} is not a valid  entry. Exiting now.')
  sys.exit()


arpackdir = os.path.expanduser(f'~/scratch/ArpackMAC/')

dod_temp = struc_type.split('-')[1]

xyzdir = os.path.expanduser(f'~/scratch/clean_bigMAC/final_sAMC_structures/{struc_type}')
xyz_prefix = f'sAMC{dod_temp}-'


pos = load_atomic_positions(n,xyzdir, xyz_prefix)

centres, radii, site_energies, MO_indices, labels, site_matrix = VRH_sites(n, struc_type, mo_type,xyzdir, arpackdir, xyz_prefix)

#temps = np.arange(40,440,10)
temps=np.array([300])
dV = 0.0
kB = 8.617333e-5
Tref = 300 # reference temperature used to determine which energies are thermally accessible to hopping
e_thermal = 4 * kB *Tref


run_name = f'production_{mo_type}'
pkl_dir = f'{run_name}_pkls/'


npydir = f'sites_data_{mo_type}' 
if not os.path.isdir(npydir):
    os.mkdir(npydir)

np.save(os.path.join(npydir,'ee.npy'), site_energies)
np.save(os.path.join(npydir, 'radii.npy'), radii)
np.save(os.path.join(npydir, 'centers.npy'), centres)
np.save(os.path.join(npydir, 'ii.npy'), MO_indices)
#    np.save(npydir + 'rr_v_masses_v_iprs_v_ee.npy',np.vstack((r_sorted,masses,site_iprs,e_sorted)))
np.save(os.path.join(npydir, 'site_state_matrix.npy'),site_matrix)


# S = np.load(npydir + 'site_state_matrix.npy')

if mo_type == 'lo' or mo_type == 'virtual':
   efermi = np.min(site_energies)
elif mo_type == 'hi': 
    efermi = np.max(site_energies)
else: # mo_type is 'virtual_w_HOMO': using mid-band states (we assume `e` is sorted here)
    N = pos.shape[0]
    if N % 2 == 0:
        eHOMO = site_energies[0]
        eLUMO = site_energies[1]
        efermi = 0.5 * (eHOMO + eLUMO)
    else:
       efermi = site_energies[0]

run_percolate(pos, centres, site_energies, radii, site_matrix,temps, dV, eF=efermi,rmax=rmax,dE_max=e_thermal,run_name=run_name,pkl_dir=pkl_dir)
