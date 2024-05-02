from itertools import combinations_with_replacement
import numpy as np
from scipy.signal import find_peaks
from numba import njit, jit, int32, float64, objmode
from qcnico import qchemMAC as qcm
from sklearn.cluster import MiniBatchKMeans

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


@njit
def bin_centers(peak_inds,xedges,yedges):
    centers = np.zeros((len(peak_inds),2))
    for k, ij in enumerate(peak_inds):
        i,j = ij
        centers[k,:] = bin_center(i,j,xedges,yedges)
    return centers

@njit
def bin_center(i,j,xedges,yedges):
    return np.array([0.5 * (xedges[i] + xedges[i+1]), 0.5 * (yedges[j] + yedges[j+1])])

@njit
def gridifyMO_opt(pos,M,n,nbins):
    x = pos.T[0]
    y = pos.T[1]
    psi = np.abs(M[:,n])**2
    #_, xedges, yedges = np.histogram2d(x,y,nbins)
    xedges = np.linspace(np.min(x)-0.1,np.max(x)+0.1,nbins+1)
    yedges = np.linspace(np.min(y)-0.1,np.max(y)+0.1,nbins+1)
    rho = np.zeros((nbins,nbins))
    for c, r in zip(psi,pos):
        x, y, _ = r
        i = np.sum(x > xedges) - 1
        j = np.sum(y > yedges) - 1
        rho[j,i] += c # <----- !!!! caution, 1st index labels y, 2nd labels x
    
    # Pad rho with zeros to detect peaks on the edges (hacky, I know)
    padded_rho = np.zeros((nbins+2,nbins+2))
    padded_rho[1:-1,1:-1] = rho
    rho_out = padded_rho

    # Make sure the bin edges are also updated
    dx = np.diff(xedges)[0] #all dxs should be the same since xedges is created using np.linspace
    dy = np.diff(yedges)[0] #idem for dys
    xedges_padded = np.zeros(xedges.shape[0]+2)
    yedges_padded = np.zeros(yedges.shape[0]+2)
    xedges_padded[0] = xedges[0] - dx
    xedges_padded[-1] = xedges[-1] + dx
    yedges_padded[0] = yedges[0] - dy
    yedges_padded[-1] = yedges[-1] + dy
    xedges_padded[1:-1] = xedges
    yedges_padded[1:-1] = yedges

    xedges = xedges_padded
    yedges = yedges_padded

    return rho_out, xedges, yedges

def get_MO_loc_centers(pos, M, n, nbins=20, threshold_ratio=0.60,return_realspace=True,padded_rho=True,return_gridify=False,shift_centers='none'):
    """This function takes in a MO (defined matrix M and index n) and returns a list of hopping sites which correspond to it.
    This version of the function is the 'original', a version made for Numba and for the purposes of using it on the full spectrum of a large MAC structure can be found below."""
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


@njit
def get_MO_loc_centers_opt(pos, M, n, nbins=20, threshold_ratio=0.60,shift_centers=True):
    """This function takes in a MO (defined matrix M and index n) and returns a list of hopping sites which correspond to it.
    This version of the function is Numba-compatible, made for running on sets of >1000 MOs."""
    rho, xedges, yedges = gridifyMO_opt(pos, M, n, nbins)
    nbins = nbins+2 #nbins describes over how many bins the actual MO is discretized; doesn't account for padding
    
    # Loop over gridified MO, identify peaks
    all_peaks = {}
    for i in range(1,nbins-1):
        data = rho[i,:]
        with objmode(peak_inds='intp[:]'):
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
        nns = [(0,0)] * shift.shape[0]
        for k in range(shift.shape[0]):
            nns[k] = (ij[0] + shift[k,0], ij[1] + shift[k,1])        
        nns = set(nns)
        intersect = nns & pk_inds
        for nm in intersect:
            if peaks[nm] <= peaks[ij]:
                peaks[nm] = 0
            else:
                peaks[ij] = 0

    #need to swap indices of peak position; 1st index actually labels y and 2nd labels x
    peak_inds = np.array([key for key in peaks.keys() if peaks[key] > 0])
    temp = np.zeros(2,dtype='int')
    for k in range(peak_inds.shape[0]):
        temp[0] = peak_inds[k,1]
        temp[1] = peak_inds[k,0]
        peak_inds[k,:] = temp[:]


    if shift_centers: #this will return real-space coords of peaks by default
        shifted_centers = shift_MO_loc_centers_random(peak_inds,xedges,yedges)
        return shifted_centers, rho, xedges, yedges
    else:
        return bin_centers(peak_inds,xedges,yedges), rho, xedges, yedges
    
@njit   
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
        
@njit
def shift_MO_loc_centers_random(peak_inds, xedges, yedges): 
    l = np.min(np.array([xedges[1]-xedges[0], yedges[1]-yedges[0]])) * 0.5
    npeaks = peak_inds.shape[0]
    dr = np.random.rand(npeaks, 2) * 2*l - l
    shifted_centers = np.zeros((npeaks,2))
    for n in range(npeaks):
        i,j = peak_inds[n]
        R0 = bin_center(i,j,xedges,yedges)
        shifted_centers[n,:] = R0 + dr[n,:]
    return shifted_centers

@njit
def correct_peaks(sites, pos, rho, xedges, yedges, side, shift_centers=True):
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

        if shift_centers:
            shift_MO_loc_centers_random(np.array([[edge_ind, peak_ind]] ),xedges,yedges)
        else:
            sites = bin_centers([(edge_ind,peak_ind)],xedges,yedges)
    return sites

# @njit
def generate_site_list(pos,M,L,R,energies,nbins=20,threshold_ratio=0.60, shift_centers=False):
    centres = np.zeros(2) #setting centers = [0,0] allows us to use np.vstack when constructing centres array
    ee = []
    inds = []
    for n in range(M.shape[1]):
        cc, rho, xedges, yedges = get_MO_loc_centers(pos,M,n,nbins,threshold_ratio,return_gridify=True,shift_centers=shift_centers)
        if n in L:
            print(n)
            cc = correct_peaks(cc, pos, rho, xedges, yedges,'L',shift_centers=shift_centers)
        
        elif n in R:
            print(n)
            cc = correct_peaks(cc, pos, rho, xedges, yedges,'R',shift_centers=shift_centers)
        

        centres = np.vstack([centres,cc])
        ee.extend([energies[n]]*cc.shape[0])
        inds.extend([n]*cc.shape[0]) #this will help us keep track of which centers belong to which MOs
    return centres[1:,:], np.array(ee), np.array(inds) #get rid of initial [0,0] entry in centres

@njit
def generate_site_list_opt(pos,M,L,R,energies,nbins=20,threshold_ratio=0.60, shift_centers=False):
    # Assume 10 sites per MO; if we run out of space, extend site energy and position arrays
    out_size = 10*M.shape[1]
    centres = np.zeros((out_size,2)) #setting centers = [0,0] allows us to use np.vstack when constructing centres array
    ee = np.zeros(out_size) 
    inds = np.zeros(out_size, dtype='int')
    nsites = 0
    for n in range(M.shape[1]):
        cc, rho, xedges, yedges = get_MO_loc_centers_opt(pos,M,n,nbins,threshold_ratio,shift_centers=shift_centers)
        if n in L:
            cc = correct_peaks(cc, pos, rho, xedges, yedges,'L',shift_centers=shift_centers)
        
        elif n in R:
            cc = correct_peaks(cc, pos, rho, xedges, yedges,'R',shift_centers=shift_centers)
        
        n_new = cc.shape[0]

        # If arrays are about to overflow, copy everything into bigger arrays
        if nsites + n_new > out_size:
            print("~~~!!!!! Site arrays about to overflow; increasing output size !!!!!~~~")
            out_size += 10*M.shape[1]

            new_centres = np.zeros((out_size,2))
            new_centres[:nsites,:] = centres[:nsites,:]
            centres = new_centres
            
            new_ee = np.zeros(out_size)
            new_ee[:nsites] = ee[:nsites]
            ee = new_ee
            
            new_inds = np.zeros(out_size,dtype='int')
            new_inds[:nsites] = inds[:nsites]
            inds = new_inds

        centres[nsites:nsites+n_new,:] = cc
        ee[nsites:nsites+n_new] = np.ones(n_new) * energies[n]
        inds[nsites:nsites+n_new] = np.ones(n_new,dtype='int') * n

        nsites += n_new
        print(nsites)

        

    return centres[:nsites,:], ee[:nsites], inds[:nsites] #get rid of 'empty' values in output arrays


def assign_AOs(pos, cc, psi=None,init_cc=True):
    """Assigns carbon atoms to localisation centers obtained from `get_MO_loc_centers` using K-means clustering."""
    nclusters = cc.shape[0]
    if init_cc:
        kmeans = MiniBatchKMeans(nclusters,init=cc)
    else:
        kmeans = MiniBatchKMeans(nclusters,init='k-means++')
    
    if psi is not None:
        kmeans = kmeans.fit(pos,sample_weight=np.abs(psi)**2)
    else:
        kmeans = kmeans.fit(pos)

    cluster_cc = kmeans.cluster_centers_
    labels = kmeans.labels_

    return cluster_cc, labels


