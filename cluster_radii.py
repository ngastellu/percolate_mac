#!/usr/bin/env python

from os import path
import pickle
import numpy as np
import matplotlib.pyplot as plt
from qcnico.plt_utils import setup_tex, histogram, multiple_histograms, get_cm

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
    
    radii = np.load(sites_dir + f'sample-{nsample}/radii.npy')
    
    radii = radii[icluster]
    d = pkl[1]
    A = pkl[2]
    print(np.all((A == A.T)))


    
    nb_neighbours = A[icluster].sum(axis=1)
    print(radii)

    return radii, d , nb_neighbours


# @njit
def gather_radii(sites_dir, percdir, nn, T, pkl_prefix):
    nMACs = len(nn)
    ntot = 250*nMACs # expect about 100 sites per cluster for each structure

    all_radii = np.zeros(ntot)
    dcrits = np.zeros(nMACs)
    nb_neighbours = np.zeros(ntot,dtype='int')

    nradii = 0
    for k, n in enumerate(nn):
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

      
simdir = '/Users/nico/Desktop/simulation_outputs/percolation/'
run_type = 'virt_100x100_gridMOs_eps_rho_1.05e-3'
pkl_prefix = 'out_percolate_eps_rho_1.05e-3' 
T = 180

outer_dirs = [simdir + d for d in ['40x40/', 'Ata_structures/tempdot6/', 'Ata_structures/tempdot5/']]

dd_rings = '/Users/nico/Desktop/simulation_outputs/ring_stats_40x40_pCNN_MAC/'

ring_data_tempdot5 = np.load(dd_rings + 'avg_ring_counts_tempdot5_new_model_relaxed.npy')
ring_data_pCNN = np.load(dd_rings + 'avg_ring_counts_normalised.npy')
ring_data_tempdot6 = np.load(dd_rings + 'avg_ring_counts_tempdot6_new_model_relaxed.npy')

p6c_tempdot6 = ring_data_tempdot6[4] / ring_data_tempdot6.sum()
p6c_tempdot5 = ring_data_tempdot5[4] / ring_data_tempdot5.sum()
p6c_pCNN = ring_data_pCNN[4]
# p6c = np.array([p6c_tdot25, p6c_pCNN,p6c_t1,p6c_tempdot6])
p6c = np.array([p6c_pCNN,p6c_tempdot6,p6c_tempdot5])

clrs = get_cm(p6c, 'inferno',min_val=0.25,max_val=0.7)
lbls = ['PixelCNN', '$\\tilde{T} = 0.6$', '$\\tilde{T} = 0.5$']

all_radii_ens = []
dcrit_ens = []
degs_ens = []

setup_tex()

fig, axs = plt.subplots(3,1,sharex=True)

fig.suptitle(f'Site radii in percolating clusters at $T = {T}$K')

for k, outer_dir in enumerate(outer_dirs):
    print(f'---------- {outer_dir.split("/")[-2]} ----------')
    sites_dir = outer_dir + f'var_radii_data/{run_type}/'
    percdir = outer_dir + f'percolate_output/zero_field/{run_type}/'
    good_runs_file = percdir + 'good_runs_eps_rho_1.05e-3.txt'
    pkl_prefix = 'out_percolate_eps_rho_1.05e-3'

    fo = open(good_runs_file)
    lines = fo.readlines()
    fo.close()
    nn = [int(l.strip()) for l in lines]
    all_radii, dcrits, nb_neighbours = gather_radii(sites_dir,percdir,nn,T,pkl_prefix)

    all_radii_ens.append(all_radii)
    dcrit_ens.append(dcrits)
    degs_ens.append(nb_neighbours)
    fig, axs[k] = histogram(all_radii,nbins=100,plt_kwargs={'color':clrs[k],'label':lbls[k]},plt_objs=(fig,axs[k]),show=False,log_counts=False)
    axs[k].legend()

axs[-1].set_xlabel('Site radii [\AA]')
plt.show()


for k in range(3):
    rr = all_radii_ens[k]
    dd = degs_ens[k]

    fig, ax = plt.subplots()

    ax.set_title(f'{lbls[k]} ensemble, $T = {T}K$')
    ax.scatter(rr,dd,c=clrs[k],s=2.0)
    ax.set_xlabel('Site radii [\AA]')
    ax.set_ylabel('\# of neighbours')

    plt.show()
