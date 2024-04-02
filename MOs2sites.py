from itertools import combinations_with_replacement
import numpy as np
from scipy.signal import find_peaks
from numba import njit, jit
from qcnico import qchemMAC as qcm

"""
Author: Nico Gastellu
Date: March 31st, 2024

Set of functions that allow us to obtain hopping sites (useful for percolation theory calcls) from MO obtained from tight-binding calcs.
"""

def LR_sites_from_scratch(pos, M, gamma, site_inds, tolscal=3.0, return_ndarray=False):
    """Determines which hopping sites 'strongly coupled' to the leads 'fronm scratch', i.e. 
    without having computed the MO-lead couplings. The `site_inds` argument is an array if inds
    whose jth entry corresponds to the index of the MO which yielded hopping site j (i.e. 
    `site_inds[j] = n` <==> site j come from MO |psi_n>).""" 
    # Get MO couplings
    agaL, agaR = qcm.AO_gammas(pos, gamma)
    gamL, gamR = qcm.MO_gammas(M,agaL, agaR, return_diag=True)

    L,R = LR_sites_from_MOgams(gamL, gamR, site_inds, tolscal)

    if return_ndarray:
        return L, R
    else:
        return set(L), set(R)

@njit
def LR_sites_from_MOgams(gamL, gamR, site_inds, tolscal=3.0):
    """Determines which hopping sites 'strongly coupled' to the leads having already computed 
    the MO-lead couplings (the two first arguments of the function). The `site_inds` argument is an array if inds
    whose jth entry corresponds to the index of the MO which yielded hopping site j (i.e. 
    `site_inds[j] = n` <==> site j come from MO |psi_n>).""" 
    # Define high-coupling threshold
    gamL_tol = np.mean(gamL) + tolscal*np.std(gamL)
    gamR_tol = np.mean(gamR) + tolscal*np.std(gamR)

    # 'Transform' from MO labeling to site labeling
    sgamL = gamL[site_inds]
    sgamR = gamR[site_inds]

    # Get strongly coupled sites
    L = (sgamL >= gamL_tol).nonzero()[0]
    R = (sgamR >= gamR_tol).nonzero()[0]

    # Using sets is faster to check membership; but Numba complains; 
    # Cast L and R into sets if not working with Numba
    return L,R 

@njit
def LR_MOs(gamL, gamR, tolscal=3.0):
    """Determines which MOs are strongly coupled to the leads."""

    # Define high-coupling threshold
    gamL_tol = np.mean(gamL) + tolscal*np.std(gamL)
    gamR_tol = np.mean(gamR) + tolscal*np.std(gamR)

    LMOs = (gamL > gamL_tol).nonzero()[0]
    RMOs = (gamR > gamR_tol).nonzero()[0]

    return LMOs, RMOs



def bin_centers(peak_inds,xedges,yedges):
    centers = np.zeros((len(peak_inds),2))
    for k, ij in enumerate(peak_inds):
        i,j = ij
        centers[k,:] = bin_center(i,j,xedges,yedges)
    return centers

# @jit
def bin_center(i,j,xedges,yedges):
    return np.array([0.5 * (xedges[i] + xedges[i+1]), 0.5 * (yedges[j] + yedges[j+1])])



def get_MO_loc_centers(pos, M, n, nbins=20, threshold_ratio=0.60,return_realspace=True,padded_rho=True,return_gridify=False,shift_centers='none'):
    rho, xedges, yedges = qcm.gridifyMO(pos, M, n, nbins, padded_rho, return_edges=True)
    if padded_rho:
        nbins = nbins+2 #nbins describes over how many bins the actual MO is discretized; doesn't account for padding
    
    # Loop over gridified MO, identify peaks
    all_peaks = {}
    for i in range(1,nbins-1):
        data = rho[i,:]
        peak_inds, _ = find_peaks(data)
        for j in peak_inds:
            peak_val = data[j]
            if peak_val > 1e-4: all_peaks[(i,j)] = peak_val

    threshold = max(all_peaks.values())*threshold_ratio
    peaks = {key:val for key,val in all_peaks.items() if val >= threshold}

    # Some peaks still occupy several neighbouring pixels; keep only the most prominent pixel
    # so that we have 1 peak <---> 1 pixel.
    pk_inds = set(peaks.keys())
    shift = np.array([[0,1],[1,0],[1,1],[0,-1],[-1,0],[-1,-1],[1,-1],[-1,1]])
    
    while pk_inds:
        ij = pk_inds.pop()
        nns = set(tuple(nm) for nm in ij + shift)
        intersect = nns & pk_inds
        for nm in intersect:
            if peaks[nm] <= peaks[ij]:
                #print(nm, peaks[nm])
                peaks[nm] = 0
            else:
                peaks[ij] = 0

    #need to swap indices of peak position; 1st index actually labels y and 2nd labels x
    peak_inds = np.roll([key for key in peaks.keys() if peaks[key] > 0],shift=1,axis=1)
    print(peak_inds.dtype)

    if shift_centers == 'density': #this will return real-space coords of peaks by default
        shifted_centers = shift_MO_loc_centers_rho(peak_inds,rho,xedges,yedges,pxl_cutoff=1)
        if return_gridify:
            return shifted_centers, rho, xedges, yedges
        else:
            return shifted_centers
    elif shift_centers == 'random': #this will return real-space coords of peaks by default
        shifted_centers = shift_MO_loc_centers_random(peak_inds,xedges,yedges)
        if return_gridify:
            return shifted_centers, rho, xedges, yedges
        else:
            return shifted_centers
    else:
        if return_realspace and return_gridify:
            return bin_centers(peak_inds,xedges,yedges), rho, xedges, yedges
        elif return_realspace and (not return_gridify):
            return bin_centers(peak_inds,xedges,yedges)
        elif return_gridify and (not return_realspace):
            return peak_inds, rho, xedges, yedges
        else:
            return peak_inds

# @njit   
def shift_MO_loc_centers_rho(peak_inds, rho, xedges, yedges, pxl_cutoff=1):
    l = np.min([xedges[1]-xedges[0], yedges[1]-yedges[0]]) #* 0.5
    npeaks = peak_inds.shape[0]
    # neighbours = np.array([[0,1],[1,0],[1,1],[0,-1],[-1,0],[-1,-1],[1,-1],[-1,1]])
    neighbours = combinations_with_replacement(np.arange(-pxl_cutoff, pxl_cutoff+1),2)
    shifted_centers = np.zeros((npeaks,2))
    for n in range(npeaks):
        dr = np.zeros(2)
        i,j = peak_inds[n] 
        R0 = bin_center(i,j,xedges,yedges)
        # R0_normd = l * R0 / np.linalg.norm(R0)
        psiM = rho[j,i]
        print(f'**** (i,j) = ({i,j}) -->  R0 = {R0} ; psiM = {psiM} ****')
        n_neighbs = 0
        for neighb in neighbours:
            neighb = np.array(neighb)
            k = i + neighb[0]
            l = j + neighb[1]
            print('(k,l) = ', (k,l))    
            psim = rho[l,k]
            print(psim)
            # R1 = bin_center(k,l,xedges,yedges)
            # R1 = l * R1 / np.linalg.norm(R1)
            # print('R1 = ', R1)
            print('psim = ', psim)
            dr += np.cbrt(psim/psiM)*neighb*l/np.linalg.norm(neighb)
            n_neighbs += 1
        shifted_centers[n,:] = R0 + dr/(n_neighbs)
        print(f'--- R shifted = {shifted_centers[n,:]} ---\n')
    return shifted_centers
        

def shift_MO_loc_centers_random(peak_inds, xedges, yedges): 
    l = np.min([xedges[1]-xedges[0], yedges[1]-yedges[0]]) * 0.5
    npeaks = peak_inds.shape[0]
    dr = np.random.rand(npeaks, 2) * 2*l - l
    shifted_centers = np.zeros((npeaks,2))
    for n in range(npeaks):
        i,j = peak_inds[n]
        R0 = bin_center(i,j,xedges,yedges)
        shifted_centers[n,:] = R0 + dr[n,:]
    return shifted_centers

def correct_peaks(sites, pos, rho, xedges, yedges, side, shift_centers='none'):
    x = pos[:,0]
    length = np.max(x) - np.min(x)
    midx = length/2

    if side == 'L':
        goodbools = sites[:,0] < midx
    else: # <==> side == 'R'
        goodbools = sites[:,0] > midx
    
    # First, remove bad sites
    sites = sites[goodbools]

    # Check if any sites are left, if not, add peak on the right edge, at the pixel with the highest density
    if not np.any(goodbools):
        print('!!! Generating new peaks !!!')
        if side == 'L':
            edge_ind = 1
        else: # <==> side == 'R' 
            edge_ind = -3
        peak_ind = np.argmax(rho[:,edge_ind]) -1 

        if shift_centers == 'density':
            shift_MO_loc_centers_rho(peak_ind,rho,xedges,yedges,pxl_cutoff=1)
        elif shift_centers == 'random':
            shift_MO_loc_centers_random(peak_ind,xedges,yedges)
        else:
            sites = bin_centers([(edge_ind,peak_ind)],xedges,yedges)
    
    
    return sites

def generate_site_list(pos,M,L,R,energies,nbins=20,threshold_ratio=0.60, shift_centers=False):
    centres = np.zeros(2) #setting centers = [0,0] allows us to use np.vstack when constructing centres array
    ee = []
    inds = []
    for n in range(M.shape[1]):
        cc, rho, xedges, yedges = get_MO_loc_centers(pos,M,n,nbins,threshold_ratio,return_gridify=True,shift_centers=shift_centers)
        if n in L:
            print(n)
            cc = correct_peaks(cc, pos, rho, xedges, yedges,'L',shift=shift_centers)
        
        elif n in R:
            print(n)
            cc = correct_peaks(cc, pos, rho, xedges, yedges,'R',shift=shift_centers)
        

        centres = np.vstack([centres,cc])
        ee.extend([energies[n]]*cc.shape[0])
        inds.extend([n]*cc.shape[0]) #this will help us keep track of which centers belong to which MOs
    return centres[1:,:], np.array(ee), np.array(inds) #get rid of initial [0,0] entry in centres