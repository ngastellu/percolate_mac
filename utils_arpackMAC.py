#!/usr/bin/env python

import numpy as np
from os import path
from qcnico.coords_io import read_xsf
from qcnico.remove_dangling_carbons import remove_dangling_carbons
from qcnico import plt_utils
from qcnico.qcffpi_io import read_energies, read_MO_file, read_Hao


def gen_mos(Mdir,lbls,filename_template='MOs_ARPACK_bigMAC'):
    for nn in lbls:
        yield np.load(Mdir+f'{filename_template}-{nn}.npy')

def gen_energies(edir,lbls,filename_template='eARPACK_bigMAC'):
    for nn in lbls:
        yield np.load(edir+f'{filename_template}-{nn}.npy')


def gen_pos(posdir, lbls, rCC):
    for nn in lbls:
        yield remove_dangling_carbons(read_xsf(path.join(posdir,f"bigMAC-{nn}_relaxed.xsf"))[0], rCC)

def gen_energy_spacings(energies_ensemble):
    for energies in energies_ensemble:
        yield np.abs(np.diff(energies))

def plot_energy_spacing_stats(edir,lbls,filename_template='eARPACK_bigMAC',hist_kwargs=None):
    energies_ensemble = gen_energies(edir,lbls,filename_template)
    diffs = np.hstack(gen_energy_spacings(energies_ensemble))
    plt_utils.histogram(diffs,**hist_kwargs)


def find_redundant_MOs(e1,e2,M1,M2,e_eps=1e-4,M_eps=1e-5):
    """Given two sets of eigenpairs (e1, M1) and (e2, M2), return the eigenpairs that are common to both.
    This is done in two steps:

        1. Find the pairs of eigenvalues that within `e_eps` of (i.e. basically equal to) each other.

        2. Of those close eigenvalues, see if their corresponding eigenvectors colinear, or equivalently 
           if the abs value of their inner product is with M_eps of 1.
    
    Returns the indices of equivalent eigenpairs in both sets."""
    
    dE = e1[:,None] - e2
    ii, jj = (np.abs(dE) < e_eps).nonzero()
    if len(ii) > 0:
        M1i = M1[:,ii].T
        M2j = M2[:,jj]
        inner_products = np.abs((M1i @ M2j).diagonal())
        
        isame = (np.abs(inner_products - 1) < M_eps).nonzero()
        if len(isame[0]) > 0:
            return ii[isame], jj[isame]
    return [[],[]]

def remove_redundant_eigenpairs(e,M,e_eps=1e-6,M_eps=1e-6):
    """Given a list of eigenpairs (e,M) with potential redundancies (e.g. bc list of eigenpairs is generated
    by concatenating eigenpairs from separate runs), return the shortened list of eigenpairs (e',M') without
    the redundancies. The redundant eigenpairs are found using the method as `find_redundant_MOs`."""
    dE = e[:,None] - e
    ij = np.vstack(np.abs(dE < e_eps).nonzero())
    ij = ij[:,(ij[0,:] - ij[1,:] > 0)] # remove self-differences and equivalent differences

    if ij.shape[1] > 0:
        print(f'!! Found {ij.shape[1]} energy differences smaller than {e_eps} !!', flush=True)
        Mi = M[:,ij[0]].T
        Mj = M[:,ij[1]]
        inner_products = np.abs((Mi @ Mj).diagonal())
        isame = (np.abs(inner_products - 1) < M_eps).nonzero()
        if len(isame[0]) > 0:
            print(f'*!!! Found {len(isame[0])} pairs of MOs with inner products within {M_eps} of 1 !!!*', flush=True)
            iredundant = set(ij[isame,1]) # indices of redundant eigenpairs
            all_inds = set(range(e.shape[0]))
            ikeep = np.array(list(all_inds - iredundant))
            e = e[ikeep]
            M = M[:,ikeep]
    print('Now sorting energies and MOs...', flush=True)
    isorted = np.argsort(e)
    e = e[isorted]
    M = M[:,isorted]
    return e, M
        


def gen_grouped_eigenpairs(edirs,Mdirs,lbls,efilename_templates, Mfilename_templates,remove_redundant=True):
    energies_gen = [gen_energies(edir, lbls,efnt) for (edir, efnt) in zip(edirs, efilename_templates)]
    mos_gen = [gen_mos(Mdir, lbls, mfnt) for (Mdir, mfnt) in zip(edirs, Mfilename_templates)]
    nlbls = range(len(lbls))

    for n in range(lbls):
        energies = np.hstack([egen.next() for egen in energies_gen])
        MOs = np.hstack([mgen.next() for mgen in mos_gen])
        if remove_redundant:
            energies, MOs = remove_redundant_eigenpairs(energies, MOs)
        else:
            isorted = np.argsort(energies)
            energies = energies[isorted]
            MOs = MOs[:,isorted]

        yield energies, MOs


def gen_Hao_qcffpi(lbls,natoms,Hdir,filename_template='Hao-'):
    for l, N in zip(lbls, natoms):
        yield read_Hao(Hdir + filename_template + f'{l}.dat', N)

def gen_energies_qcffpi(lbls,natoms,edir,filename_template):
    for l, N in zip(lbls, natoms):
        yield read_energies(edir + filename_template + f'{l}.dat', N)

def gen_pos_MOs(lbls,Mdir,filename_template,natoms=None,return_pos=True):
    for k, l in enumerate(lbls):
        if natoms is None:
            if return_pos:
                yield read_MO_file(Mdir + filename_template + f'-{l}.dat')
            else:
                yield read_MO_file(Mdir + filename_template + f'-{l}.dat')[1]
        else:
            if return_pos:
                yield read_MO_file(Mdir + filename_template + f'-{l}.dat',Natoms=natoms[k])
            else:
                yield read_MO_file(Mdir + filename_template + f'-{l}.dat',Natoms=natoms[k])
            
