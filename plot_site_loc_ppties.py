#!/usr/bin/env python 

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from qcnico import plt_utils


def gen_data(lbls, ddir,filename='rr_v_masses_v_iprs_v_ee'):
    for n in lbls:
        # print(n)
        npy = ddir + f'{filename}-{n}.npy'
        yield np.load(npy)



simdir = '/Users/nico/Desktop/simulation_outputs/percolation/'
run_type = 'virt_100x100_gridMOs_rmax_50'
npyname = 'rr_v_masses_v_iprs_v_ee' 

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

clrs = plt_utils.get_cm(p6c, 'inferno',min_val=0.25,max_val=0.7)
lbls = ['PixelCNN', '$\\tilde{T} = 0.6$', '$\\tilde{T} = 0.5$']

plt_utils.setup_tex()

rmax = 60

istrucs = np.zeros((3,3),dtype=int) # structure inds of the high-mass sites

# nsites = [104424, 77544, 56588]
nsites = np.zeros(3,dtype='int')
max_density = 0 #useful to normalise colorbar for histogram (2nd plot)

for k, d in enumerate(outer_dirs):
    ddir  = d + 'var_radii_data/' + 'rr_v_masses_v_iprs_v_ee_eps_rho_0.00105/'
    percdir = d + f'percolate_output/zero_field/{run_type}/'
    good_runs_file = percdir + 'good_runs_rmax_50.txt'
    nbig = 0
    
    min_bigR_density = np.inf

    fo = open(good_runs_file)
    lines = fo.readlines()
    fo.close()
    nn = [int(l.strip()) for l in lines]
    datgen = gen_data(nn, ddir, filename=npyname)
    
    fig, ax = plt.subplots() 

    for dat in datgen:
        # print(dat.shape)
        rr, masses, iprs, ee = dat
        densities = masses / (np.pi * rr * rr)
        ee -= np.min(ee)
        ax.scatter(densities, rr, c=1.0/(np.sqrt(iprs)),s=1.0)
        nsites[k] += rr.shape[0]
        nbig += (rr>rmax).sum()
        
        max_rho = np.max(densities)
        if max_rho > max_density:
            max_density = max_rho
        
        ibigRs = rr > rmax
        if np.any(ibigRs):
            bigR_densities = densities[ibigRs]
            smolrho = np.min(bigR_densities)

            if smolrho < min_bigR_density:
                min_bigR_density = smolrho



    # ax.set_xlabel('$1/\sqrt{IPR}$')
    ax.set_xlabel('Site probability density')
    ax.set_ylabel('Site radius [\AA]')

    plt.show()

    print(f'Nb of sites with r > {rmax} angstroms = {nbig} / {nsites[k]}')
    print(f'Min density of sites with r > {rmax} = ', min_bigR_density)
print('Max density = ', np.max(densities))

# ------------- Histograms of site densities -----------

# fig, axs = plt.subplots(3,1,sharex=True)

# colormap = plt.cm.viridis
# # norm = mpl.colors.Normalize(vmin=0,vmax=max_density)
# norm = mpl.colors.LogNorm(vmin=1e-4,vmax=max_density*150)

# rbins = np.linspace(0,220,50)
# centers = 0.5 * (rbins[1:] + rbins[:-1])
# dx = centers[1] - centers[0]

# for k, d in enumerate(outer_dirs):
#     print(nsites[k])
#     ddir  = d + 'var_radii_data/' + 'rr_v_masses_v_iprs_v_ee_eps_rho_0.00105/'
#     percdir = d + f'percolate_output/zero_field/{run_type}/'
#     good_runs_file = percdir + 'good_runs_rmax_50.txt'

#     fo = open(good_runs_file)
#     lines = fo.readlines()
#     fo.close()
#     nn = [int(l.strip()) for l in lines]
#     datgen = gen_data(nn, ddir, filename=npyname)

#     all_radii = np.zeros(nsites[k])
#     all_densities = np.zeros(nsites[k])
    
#     j = 0
#     for dat in datgen:
#         # print(dat.shape)
#         rr, masses, iprs, ee = dat
#         densities = masses / (np.pi * rr * rr)
#         n_new = rr.shape[0]
#         all_radii[j:j+n_new] = rr
#         all_densities[j:j+n_new] = densities
#         j+= n_new
    
#     ii = np.searchsorted(rbins,all_radii) 
#     print(ii)

#     nbins = rbins.shape[0]
#     rcounts = np.zeros(nbins)
#     avg_densities = np.zeros(nbins)

#     for m in range(nbins):
#         mask = (ii == m)
#         rcounts[m] = mask.sum()
#         if rcounts[m] == 0:
#             avg_densities[m] = 0
#         else:
#             avg_densities[m] = np.mean(all_densities[mask])
    
#     print(1000*avg_densities)
#     print(norm(avg_densities))
#     bar_clrs = colormap(norm(1000*avg_densities))
#     print('nb of radii in last bin = ',rcounts[-1])
#     axs[k].bar(centers, rcounts[:-1], color=bar_clrs ,align='center',width=dx)
#     axs[k].set_title(lbls[k])


# axs[-1].set_xlabel('Site radii [\AA]')
# # plt.colorbar()
# plt.show()   

