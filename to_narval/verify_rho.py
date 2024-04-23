#!/usr/bin/env python 

from glob import glob
import numpy as np


iMOs = sorted([int(f.split('-')[1].split('.')[0]) for f in glob('rho-*.npy')])
rho_bools = np.zeros(len(iMOs),dtype='bool')
xedges_bools = np.zeros(len(iMOs),dtype='bool')
yedges_bools = np.zeros(len(iMOs),dtype='bool')
xedges_rollback_bools = np.zeros(len(iMOs),dtype='bool')
yedges_rollback_bools = np.zeros(len(iMOs),dtype='bool')

xedges = np.load('xedges-1.npy')
yedges = np.load('yedges-1.npy')

for k, n in enumerate(iMOs):
    print(f'*** {n} ***')
    rho_new = np.load(f'rho-{n}.npy')
    rho_old = np.load(f'rho_rollback-{n}.npy')
    rho_bools[k] = np.all(rho_new == rho_old)

    xedges_rollback_bools[k] = np.all(xedges == np.load(f'xedges_rollback-{n}.npy'))
    yedges_rollback_bools[k] = np.all(yedges == np.load(f'yedges_rollback-{n}.npy'))
    xedges_bools[k] = np.all(xedges == np.load(f'xedges-{n}.npy'))
    yedges_bools[k] = np.all(yedges == np.load(f'yedges-{n}.npy'))


print('Rho: ', np.all(rho_bools))
print('xedges_rb: ', np.all(xedges_rollback_bools))
print('yedges_rb: ', np.all(yedges_rollback_bools))
print('xedges: ', np.all(xedges_bools))
print('yedges: ', np.all(yedges_bools))