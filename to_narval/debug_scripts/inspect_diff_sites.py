#!/usr/bin/env python

import numpy as np

interesting_iMOs = np.load('bad_ii.npy') #inds of MOs where sites are different between old and new method

print(interesting_iMOs)

ii_new = np.load('ii_new_method.npy')
ii_old = np.load('ii_old_method.npy')

#Gets inds of sites obtained from 'interesting' MOs
inew = np.any(ii_new[:,None] == interesting_iMOs, axis=1).nonzero()[0]
iold = np.any(ii_old[:,None] == interesting_iMOs, axis=1).nonzero()[0]


new_sites = np.load('pk_inds_new_method.npy')
old_sites = np.load('pk_inds_old_method.npy')

print(new_sites.shape)
print(old_sites.shape)

new_sites = new_sites[inew]
old_sites = old_sites[iold]

dists = np.linalg.norm(new_sites[:,None,:] - old_sites, axis=2)

print('Looping over new_sites...')
for k, ns in enumerate(new_sites):
    dd = dists[k,:]
    i = np.argmin(dd)
    closest_os = old_sites[i,:]
    if dd[i] == 0:
        print(f'{ns}\t({inew[k]})\t<--->\t{closest_os}\t({iold[i]}_')
    else: 
        print(f'*** No match found! {ns}\t({inew[k]})\t<--->\t{closest_os}\t({iold[i]}) ***')

print('\n----------------------------\nLooping over new_sites...')
for k, ols in enumerate(old_sites):
    dd = dists[:,k]
    i = np.argmin(dd)
    closest_ns = new_sites[i,:]
    if dd[i] == 0:
        print(f'{ols}\t({iold[k]})\t<--->\t{closest_ns}\t({inew[i]})')
    else: 
        print(f'*** No match found! {ols}\t({iold[k]})\t<--->\t{closest_ns}\t({inew[i]})***')
