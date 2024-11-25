#!/usr/bin/env python

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from qcnico.plt_utils import MAC_ensemble_colours, setup_tex
# from mpl_toolkits import inset_axes, InsetPosition, mark_inset


kB = 8.617e-5
e2C = 1.602177e-19 # elementary charge to Coulomb
w0 = 1e15
T = 300

# This factor combines the hop frequency with the unit conversion (to yield conductivity in siemens)
# w0 is chosen such that the final result matches the results from the AMC paper.
conv_factor = e2C*w0

sigmasdir = '/Users/nico/Desktop/simulation_outputs/percolation/sigmas_v_T/'
# motypes = ['kBTlo_dcut500','virtual','kBThi']
motypes = ['lo','virtual_w_HOMO','hi']

# temps = np.arange(40,440,10)[14:]

r_maxs = ['18.03', '136.47', '199.33']
ensemble_lbls = ['sAMC-500', 'sAMC-q400', 'sAMC-300']
gate_names = [r'$\epsilon_0$', r'$\epsilon_{\mathsf{F}}$', r'$\epsilon_N - \mathsf{4} k_{\mathsf{B}}T$']
structypes=['40x40', 'tempdot6', 'tempdot5']
zorders=[3,2,1]

clrs = MAC_ensemble_colours()

# setup_tex()
rcParams['mathtext.fontset'] = 'cm'
rcParams['font.family'] = 'sans-serif'
# rcParams['font.weight'] = 'light'
# rcParams['font.sans-serif'] = 'sans-serif'


fontsize = 60
rcParams['font.size'] = fontsize
# rcParams['figure.figsize'] = [8,7]

fig, ax = plt.subplots()
fig.subplots_adjust(bottom=0.202,top=0.99)
# tick_font_ppties = {'font':'Computer Modern Sans Serif'}

#------ Inset code ------
# inset_xlims = [1.9,2.1]
# inset_ylims = [6e-11,4.5e-10]

# Position of the bottom left corner in cartesian coords
# x0 = 2.3
# y0 = 4e-17

# Convert to fractional coords
# xmin = 0.8
# ymin = 5e-31
# xmax = 3.2
# ymax = 2e-9

# x0 = (x0 - xmin)/(xmax - xmin)
# y0 = (y0 - ymin)/(ymax - ymin)
# y0 = 0.6

# Define size of inset in fractional coords
# width = 0.18
# height = 0.22

# axins = ax.inset_axes([x0,y0,width,height],xlim=inset_xlims,ylim=inset_ylims,xticklabels=[],xticks=[])

for st, rmax, lbl, c, zz in zip(structypes,r_maxs,ensemble_lbls,clrs,zorders):
    print(f'\n---------- {st} ----------')
    plot_sigmas = np.zeros(3)
    for x, mt in zip(range(1,4),motypes):
        if st == '40x40' and mt != 'virtual_w_HOMO':
            temps, sigmas, sig_errs = np.load(f'/Users/nico/Desktop/simulation_outputs/percolation/sigmas_v_T/sigma_v_T_w_err_{st}_rmax_{rmax}_sites_gammas_kBT{mt}.npy').T
        else:
            temps, sigmas, sig_errs = np.load(f'/Users/nico/Desktop/simulation_outputs/percolation/sigmas_v_T/sigma_v_T_w_err_{st}_rmax_{rmax}_{mt}.npy').T
        kselect = (temps==T).nonzero()[0]
        sigma = sigmas[kselect] * conv_factor / (kB*T)
        plot_sigmas[x-1] = sigma
        sig_err = sig_errs[kselect] * conv_factor / (kB*T)
        print(f'{mt} ---> {sigma} ± {sig_err}')
    ax.plot(range(1,4), plot_sigmas, 'o-', ms=20.0, c=c,label=lbl,lw=1.5)

        #     ax.errorbar(x,sigma,yerr=sig_err,fmt='o',label=lbl,ms=10.0,lw=2.0,c=c,zorder=zz)
        # else:
        #     ax.errorbar(x,sigma,yerr=sig_err,fmt='o',ms=10.0,lw=2.0,c=c,zorder=zz)
        # axins.errorbar(x,sigma,yerr=sig_err,fmt='o',ms=30.0,lw=15.0,c=c)
        

# ax.indicate_inset_zoom(axins,edgecolor='k')

fontsize_axes = 45

# for label in ax.get_xticklabels():
#     label.set_fontproperties(tick_font_ppties)
# for label in ax.get_yticklabels():
#     label.set_fontproperties(tick_font_ppties)

ax.set_yscale('log')
ax.set_ylabel(r'$G$ [S]',fontsize=fontsize)
ax.set_xlabel(r'Chemical potential $\mu$',fontsize=fontsize)
ax.tick_params('both',which='major',length=10,width=1.6,labelsize=fontsize_axes)
ax.set_xticks(range(1,4),gate_names,fontsize=fontsize_axes+5)
ax.set_box_aspect(1)
# plt.legend()
plt.show()


