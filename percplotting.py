#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt 
from qcnico import plt_utils

def dcrit_hists(dcrits,temps,nbins,plot_inds=None,colormap='coolwarm',alpha=0.6,usetex=True,plt_objs=None,show=True):
    Tcm = plt_utils.get_cm(temps,colormap,max_val=1.0)
    
    # If indices aren't specified, plot all the histograms
    # We still pass all of the dcrits to this function (even those we don't plot) to get a better
    # contrasted colormap.
    if plot_inds is None:
        plot_inds = range(len(temps))


    if usetex:
        plt_utils.setup_tex()
    
    if plt_objs is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = plt_objs

    for k in plot_inds:
        plt_utils.histogram(dcrits[:,k],nbins=nbins,show=False, density=True, plt_objs=(fig,ax),
            plt_kwargs={'alpha': alpha, 'color': Tcm[k], 'label': f'$T = {temps[k]}$K'})
    
    ax.set_xlabel('Critical distance $u_{c}$')
    ax.set_ylabel('$P(u)$')
    if show:
        plt.legend()
        plt.show()
    else:
        return fig, ax
