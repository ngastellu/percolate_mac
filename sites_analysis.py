#!/usr/bin/env python


from os import path
import sys
import numpy as np
from utils_analperc import get_conduction_cluster, conduction_mask


def undistorted_mask():
    pass


def mask_overlap(mask1, mask2):
    """Computes the AND (i.e. intersection) between two boolean masks."""

    return (mask1 * mask2).nonzero()[0]


def track_overlaps(datadir, structural_mask, sites_mask, pkl_prefix, temps):
    for T in temps:
        clusters = get_conduction_cluster(datadir, pkl_prefix, T)
        if len(clusters) > 1:
            print(f'Multiple ({len(clusters)}) conduction clusters detected!', flush=True)
            print('Cluster sizes = ', [len(c) for c in clusters],flush=True)



rmax = sys.argv[1]
run_lbl = 'rmax_198.69_psipow1'
pkl_prefix = f'out_percolate_{run_lbl}'
temps = np.arange(40,440,10)



with open(f'to_local_{run_lbl}/good_runs_{run_lbl}.txt') as fo:
    lines = fo.readlines()

nn = [int(l.strip()) for l in lines]

for n in nn:
    datadir = f'sample-{n}'

