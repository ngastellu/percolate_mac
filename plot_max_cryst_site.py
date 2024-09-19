#!/usr/bin/env python

import numpy as np
from qcnico.coords_io import read_xyz
from qcnico.plt_utils import setup_tex, MAC_ensemble_colours
from qcnico.qcplots import plot_MO
import matplotlib.pyplot as plt
import pickle
from glob import glob


def get_struc_labels(ddir, prefix='sample',connector_char='-'):
    return np.array([int(x.split(connector_char)[-1]) for x in glob(f'{ddir}/{prefix}{connector_char}*')])

def full_cryst_radii(nn, temps, ddir, rmax, dir_prefix='sample-', npy_prefix='clust_cryst-',return_isites=False):
    zero_cryst_radii = np.array([-1]) #initialise output arr for easier concatenation
    if return_isites: # !!!! ONLY use of working with a single structure !!!!
        all_izero_cryst = np.zeros(1,dtype='int')
    for n in nn:
        print(f'--------- {n} ---------')
        for T in temps:
            cluster_crysts = np.load(f'{ddir}/electronic_crystallinity/cluster_crystallinities_rmax_{rmax}/{dir_prefix}{n}/{npy_prefix}{T}K.npy')
            with open(f'{ddir}/percolate_output/zero_field/virt_100x100_gridMOs_rmax_{rmax}/sample-{n}/out_percolate_rmax_{rmax}-{T}K.pkl', 'rb') as fo:
                clusters = pickle.load(fo)[0]
            if len(clusters) != 1:
                print(len(clusters))
                continue # hard to deal with systems with multiple clusters; crystallinity unifies all clusters
            cluster = np.array(list(clusters[0]))
            radii = np.load(f'{ddir}/var_radii_data/to_local_sites_data/sample-{n}/sites_data_0.00105/radii.npy')
            izero_crysts_cluster = (cluster_crysts == 0) # indices of zero crystallinity sites in `cluster_crysts`
            # print(izero_crysts_cluster.sum())
            izero_crysts = cluster[izero_crysts_cluster] # actual indices of zero-crystallinity sites
            zero_cryst_radii = np.hstack((zero_cryst_radii, radii[izero_crysts]))
            if return_isites:
                all_izero_cryst = np.hstack((all_izero_cryst,izero_crysts))
    
    if return_isites:
        return zero_cryst_radii[1:], all_izero_cryst[1:]
    else:
        return zero_cryst_radii[1:] #get rid of spurious 1st element


 
structypes = ['40x40', 'tempdot6', 'tempdot5']
lbls= ['PixelCNN', '$\\tilde{T} = 0.6$', '$\\tilde{T} = 0.5$']
rmaxs = [18.03, 121.2, 198.69]
temps = np.arange(180,440,10)
clrs = MAC_ensemble_colours()

simdir = '/Users/nico/Desktop/simulation_outputs/'
percdir = simdir + 'percolation/'

setup_tex()
# fig, axs = plt.subplots(3,1,sharex=True)
nbins = 101
bins = np.linspace(0,100,nbins)



# np.random.seed(0)
structype = 'tempdot5'
nn = [11]
T = 180
rmax = 198.69
datadir = percdir + '/' + structype

crysts = np.load(f'/Users/nico/Desktop/simulation_outputs/percolation/{structype}/electronic_crystallinity/cluster_crystallinities_rmax_{rmax}/sample-11/clust_cryst-{T}K.npy')

imaxcryst = np.argmax(crysts)
max_cryst = crysts[imaxcryst]
print(f'Site {imaxcryst} has the max crystallinity = {max_cryst}')

S = np.load(f'/Users/nico/Desktop/simulation_outputs/percolation/site_ket_matrices/{structype}_rmax_{rmax}/site_kets_psipow2-{nn[0]}.npy')
Sbinary = np.zeros(S.shape,dtype=int)
Sbinary[S.nonzero()] = 1
pos = read_xyz(f'/Users/nico/Desktop/simulation_outputs/MAC_structures/relaxed_no_dangle/sAMC-300/sAMC300-{nn[0]}.xyz')


for ii in np.argsort(crysts)[-2:]:
    print(f'{ii} ---> {crysts[ii]}')
    plot_MO(pos, Sbinary, ii, dotsize=0.5,scale_up=20.0,cmap='bwr')#,loc_centers=np.array([centres[i]]), loc_radii=[all_radii[i]])