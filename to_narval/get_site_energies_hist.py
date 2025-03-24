#!/usr/bin/env python

import os
import sys
from time import perf_counter
import numpy as np


def get_energy_range(structype):
    arpack_dir = os.path.expanduser(f'~/scratch/ArpackMAC/')
    edir = os.path.join(arpack_dir, f'{structype}/energies/{motype}/')

    e_min_global = np.inf
    e_max_global = -np.inf

    for npy in os.listdir(edir):
        energies = np.load(os.path.join(edir, npy))
        e_min = np.min(energies)
        e_max = np.max(energies)

        if e_min < e_min_global:
            e_min_global = e_min
        if e_max > e_max_global:
            e_max_global = e_max
    
    return e_min, e_max



structype = os.path.basename(os.getcwd())
motype = sys.argv[1]

if structype == '40x40':
    rmax = 18.03
    lbls = np.arange(1,301)
elif structype == 'tempdot6':
    rmax = 121.2
    lbls = np.arange(132)
elif structype == 'tempdot5':
    rmax = 198.69
    lbls = np.arange(117)
else:
    print('Invalid structype. We outtie.')


print(f'Getting energy range...', end=' ')
start = perf_counter()
e_min, e_max = get_energy_range(structype)
end = perf_counter()
print(f'Done! [{end-start} seconds]')
print('Range = ',[e_min, e_max])
nbins = 201
bins = np.linspace(e_min, e_max, nbins)

hist_global = np.zeros(nbins-1, dtype='int')

start = perf_counter()
for n in lbls:
    print(f'\n{n}', end=' ')
    try:
        energies = np.load(f'sample-{n}/sites_data_{motype}/ee.npy')
        radii = np.load(f'sample-{n}/sites_data_{motype}/radii.npy')
    except FileNotFoundError as e:
        print('NPY(s) missing!')
        continue
    rfilter = radii < rmax
    energies = energies[rfilter]
    hist, _ = np.histogram(energies, bins)
    hist_global += hist 
end = perf_counter()


centers = 0.5 * (bins[:-1] + bins[1:])
np.save(f'energy_histogram_{motype}.npy', np.vstack((centers, hist_global)).T)
