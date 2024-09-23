#!/usr/bin/env python

from os import path
import pickle
import numpy as np
import matplotlib.pyplot as plt
from qcnico.plt_utils import setup_tex, histogram, multiple_histograms, get_cm, MAC_ensemble_colours

# @njit
def network_data(sites_dir, percdir, nsample, T, pkl_prefix):
    '''Obtains the radii of the sites in a given structure's percolating cluster, at a given temperature'''

    try:
        pkl_file = percdir + f'sample-{nsample}/{pkl_prefix}-{T}K.pkl'
        fo = open(pkl_file, 'rb')
    except FileNotFoundError as e:
        #some of the runs have the 'K' after the run missing
        pkl_file = percdir + f'sample-{nsample}/{pkl_prefix}-{T}.pkl'
        fo = open(pkl_file,'rb')

    pkl = pickle.load(fo)
    fo.close()

    clusters = pkl[0]
    print('Nb of clusters = ', len(clusters))
    
    # Only expect one percolating cluster, but if there's more than one, take the uniion
    if len(clusters) > 1:
        icluster = np.array(list(clusters[0].union(*clusters[1:])))
    else:
        icluster = np.array(list(clusters[0])) 
    
    print(icluster.dtype)
    
    radii = np.load(sites_dir + 'radii.npy')
    
    radii = radii[icluster]
    d = pkl[1]
    A = pkl[2]
    print(np.all((A == A.T)))


    
    nb_neighbours = A[icluster].sum(axis=1)
    print(radii)

    return radii, d , nb_neighbours


# @njit
def gather_radii(percdir, nn, T, pkl_prefix,structype):
    nMACs = len(nn)
    ntot = 250*nMACs # expect about 250 sites per cluster for each structure

    all_radii = np.zeros(ntot)
    dcrits = np.zeros(nMACs)
    nb_neighbours = np.zeros(ntot,dtype='int')

    nradii = 0
    for k, n in enumerate(nn):
        if structype == '40x40':
            sites_dir =  f'/Users/nico/Desktop/simulation_outputs/percolation/40x40/var_radii_data/virt_100x100_gridMOs_eps_rho_1.05e-3/sample-{n}/'
        else:
            sites_dir = percdir + f'sample-{n}/var_a_npys_eps_rho_1.05e-3/'
        radii, d, degs = network_data(sites_dir,percdir,n,T,pkl_prefix)
        print(radii)
        n_new = radii.shape[0]
        dcrits[k] = d

        if nradii+n_new < ntot:
            # print('ntot = ', ntot)
            # print('nradii = ', nradii)
            # print('n_new = ', n_new)
            # print('Shape of radii = ', radii.shape)
            # print('Shape of degs = ', degs.shape)
            all_radii[nradii:nradii+n_new] = radii
            nb_neighbours[nradii:nradii+n_new] = degs
        
        else:
            ntot += 50*nMACs
            tmp = np.zeros(ntot)
            tmp[:nradii] = all_radii[:nradii]
            tmp[nradii:nradii+n_new] = radii
            all_radii = tmp
            
            tmp = np.zeros(ntot,dtype='int')
            tmp[:nradii] = nb_neighbours[:nradii]
            tmp[nradii:nradii+n_new] = degs
            nb_neighbours = tmp
        
        nradii += n_new
    
    return all_radii[:nradii], dcrits, nb_neighbours[:nradii]

      
percdir = '/Users/nico/Desktop/simulation_outputs/percolation/'
T = 300

structypes = ['40x40', 'tempdot6', 'tempdot5']
rmaxs = [18.03, 121.2, 198.69]

clrs = MAC_ensemble_colours()

lbls = ['sAMC-500', 'sAMC-q400', 'sAMC-300']

setup_tex()

fig, axs = plt.subplots(3,1,sharex=True)

bins = np.linspace(0,200,400)

fig.suptitle(f'Site radii in percolating clusters at $T = {T}$K (lowest MOs)')

for st, rmax, ax, c, lbl in zip(structypes,rmaxs,axs,clrs,lbls):
    print(f'---------- {st} ----------')

    cr_dir = path.join(percdir, st, 'percolate_output/zero_field/cluster_radii/') 
    run_name = f'rmax_{rmax}_psipow2_sites_gammas_lo'
    cluster_radii = np.load(path.join(cr_dir, f'clust_radii_{run_name}-{T}K.npy'))
    rmax_ind = np.argmax(cluster_radii)
    rmax = cluster_radii[rmax_ind]
    hist, bin_edges = np.histogram(cluster_radii,bins=bins)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    dx = centers[1] - centers[0]
    print(f'****** MAX RADII INFO FOR {st} ******')
    print('\trmax = ', rmax)
    print('\tState index = ', rmax_ind)
    ax.bar(centers, hist,align='center',width=dx,color=c,label=lbl)
    ax.legend()
    ax.set_yscale('log')
    ax.legend()

axs[-1].set_xlabel('Conducting site radii [\AA]')
plt.show()
