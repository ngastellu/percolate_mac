from itertools import combinations_with_replacement
import numpy as np
from scipy.signal import find_peaks
from sklearn.cluster import MiniBatchKMeans
from numba import njit, jit, int32, float64, objmode
from qcnico import qchemMAC as qcm
# from qcnico.graph_tools import components
from percolate import jitted_components

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

    return set(LMOs), set(RMOs)


@njit
def bin_centers(peak_inds,xedges,yedges):
    centers = np.zeros((len(peak_inds),2))
    for k, ij in enumerate(peak_inds):
    # !!! indices of 2d array are swapped with respect to coords!! j = 1st ind = row (y-coord); 2nd ind = column (x-coord) !!!
        i,j = ij 
        centers[k,:] = bin_center(i,j,xedges,yedges)
    return centers

@njit
def bin_center(i,j,xedges,yedges):
    # !!! indices of 2d array are swapped with respect to Cartesian coords!!
    # i = 1st ind = row (y-coord); j = 2nd ind = column (x-coord) !!!
    return np.array([0.5 * (xedges[j] + xedges[j+1]), 0.5 * (yedges[i] + yedges[i+1])])

@njit
def gridifyMO_opt(pos,M,n,nbins):
    pos = pos[:,:2]
    x = pos.T[0]
    y = pos.T[1]
    psi = np.abs(M[:,n])**2
    #_, xedges, yedges = np.histogram2d(x,y,nbins)
    xedges = np.linspace(np.min(x)-0.1,np.max(x)+0.1,nbins+1)
    yedges = np.linspace(np.min(y)-0.1,np.max(y)+0.1,nbins+1)
    rho = np.zeros((nbins,nbins))
    for c, r in zip(psi,pos):
        x, y = r
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

def get_MO_loc_centers(pos, M, n, nbins=20, threshold_ratio=0.50,return_realspace=True,padded_rho=True,return_gridify=False,shift_centers='none'):
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
    # peak_inds = np.roll([key for key in peaks.keys() if peaks[key] > 0],shift=1,axis=1)

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
def get_MO_loc_centers_opt(pos, M, n, nbins=20, threshold_ratio=0.50,shift_centers=False,min_distance=30.0):
    """This function takes in a MO (defined matrix M and index n) and returns a list of hopping sites which correspond to it.
    This version of the function is Numba-compatible, made for running on sets of >1000 MOs.
    
    min_distance is in ANGSTROMS"""

    rho, xedges, yedges = gridifyMO_opt(pos, M, n, nbins)
    dy = yedges[1] - yedges[0]
    min_distance_pixel = int(min_distance/dy)
    nbins = nbins+2 #nbins describes over how many bins the actual MO is discretized; doesn't account for padding
    
    # Loop over gridified MO, identify peaks
    all_peaks = {}
    for i in range(1,nbins-1):
        data = rho[i,:]
        with objmode(peak_inds='intp[:]'):
            peak_inds, _ = find_peaks(data,distance=min_distance_pixel) #about 12 angstroms
        for j in peak_inds:
            peak_val = data[j]
            if peak_val > 1e-4: all_peaks[(i,j)] = peak_val

    threshold = max(all_peaks.values())*threshold_ratio
    peaks = {key:val for key,val in all_peaks.items() if val >= threshold}

    peak_inds = np.array([key for key in peaks.keys() if peaks[key] > 0])

    centers = bin_centers(peak_inds,xedges,yedges)
    ncenters = centers.shape[0]
    if ncenters > 1:
        ikeep = clean_centers(centers,peak_inds,rho,min_dist=min_distance)
        centers = centers[ikeep,:]
        peak_inds = peak_inds[ikeep,:]
    if shift_centers:
        centers = shift_MO_loc_centers_random(peak_inds,xedges,yedges)
    return centers, rho, xedges, yedges

@njit
def jitted_swap_columns(arr,i,j):
    "Exchanges columns `i` and `j` of array `arr`."
    temp = np.zeros(arr.shape[1],dtype='int')
    for k in range(arr.shape[0]):
        temp = arr[k,:]
        temp[i] = arr[k,j]
        temp[j] = arr[k,i]
        arr[k,:] = temp[:]
    return arr

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
        # !!! indices of 2d array are swapped with respect to coords!! j = 1st ind = row (y-coord); 2nd ind = column (x-coord) !!!
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
            shift_MO_loc_centers_random(np.array([[peak_ind, edge_ind]] ),xedges,yedges)
        else:
            sites = bin_centers([(peak_ind,edge_ind)],xedges,yedges)
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
def generate_site_list_opt(pos,M,L,R,energies,nbins=20,threshold_ratio=0.50, minimum_distance=30.0, shift_centers=False):
    # Assume 10 sites per MO; if we run out of space, extend site energy and position arrays
    out_size = 10*M.shape[1]
    centres = np.zeros((out_size,2)) #setting centers = [0,0] allows us to use np.vstack when constructing centres array
    ee = np.zeros(out_size) 
    inds = np.zeros(out_size, dtype='int')
    nsites = 0
    for n in range(M.shape[1]):
        cc, rho, xedges, yedges = get_MO_loc_centers_opt(pos,M,n,nbins,threshold_ratio,min_distance=minimum_distance,shift_centers=shift_centers)
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

def assign_AOs(pos, cc, psi=None,init_cc=True,psi_pow=2,density_threshold=0, flag_empty_clusters=False):
    """Assigns carbon atoms to localisation centers obtained from `get_MO_loc_centers` using K-means clustering."""
    nclusters = cc.shape[0]
    # print('nclusts = ',nclusters)
    if init_cc:
        kmeans = MiniBatchKMeans(nclusters,init=cc,random_state=64)
    else:
        kmeans = MiniBatchKMeans(nclusters,init='k-means++',random_state=64)
    
    if psi is not None:
        density = np.abs(psi)**2
        too_low = (density < density_threshold).nonzero()[0]
        # print(f'K-means: ignoring {too_low.shape[0]} / {pos.shape[0]} carbons (eps = {density_threshold})')
        psi[too_low] = 0
        kmeans = kmeans.fit(pos,sample_weight=np.abs(psi)**psi_pow)
    else:
        kmeans = kmeans.fit(pos)

    cluster_cc = kmeans.cluster_centers_
    labels = kmeans.labels_

    if flag_empty_clusters:
        # print(cc)
        cc_labels = kmeans.predict(cc)
        slabels = set(labels)
        scclabels = set(cc_labels)
        if len(scclabels) == 0:
            print('[assign_AOs] !!!! All labels have been flagged !!!!')
        empty_labels = slabels - scclabels
        # print('Nb of flagged of labels = ', len(empty_labels))
        
        return cluster_cc, labels, empty_labels
    else:
        return cluster_cc, labels


def assign_AOs_naive(pos, cc):
    N = pos.shape[0]
    min_dists = np.ones(N) * np.inf
    labels = np.zeros(N,dtype='int')
    for k, r in enumerate(cc):
        sq_dists = ((pos - r[None,:])**2).sum(axis=1)
        smaller_bools = (sq_dists < min_dists).nonzero()[0]
        min_dists[smaller_bools] = sq_dists[smaller_bools]
        labels[smaller_bools] = k

    return labels


# @njit
def site_radii_naive(pos, M, n, centers, density_threshold=0,psi_pow=2):
    psi = M[:,n]
    if density_threshold > 0:
        density = psi ** 2
        mask = density > density_threshold
        psi = psi[mask]
        pos = pos[mask,:]

        psi /= density.sum()

    radii = np.zeros(centers.shape[0])
    for k,R in enumerate(centers):
        dR = np.linalg.norm(pos - R, axis=1)
        radii[k] = np.sum(dR * (psi**psi_pow))
    
    return radii


    



# @njit
def site_radii(pos, M, n, labels, hyperlocal='sites', density_threshold=0, flagged_labels=None,max_r=50,return_labels=False):
    valid_hl_types = ['sites', 'radii', 'all', 'none']
    if hyperlocal not in valid_hl_types:
        print(f'''[sites_radii] WARNING: specified `hyperlocal` arg \'{hyperlocal}\'is invalid; 
              should be one of the following options: {valid_hl_types}.\n Reverting to default setting: \'sites\'.''')
        hyperlocal = 'sites'
    unique_labels = np.unique(labels)
    nsites = unique_labels.shape[0]
    centers = np.zeros((nsites,2))
    radii = np.zeros(nsites)
    N = pos.shape[0]

    if flagged_labels is not None and len(flagged_labels) > 0:
        flagged_radii = np.zeros(len(flagged_labels))
        m = 0
    
    if density_threshold > 0:
        density = np.abs(M[:,n])**2
        density_mask = (density > density_threshold) 
        # print(f'[site_radii] Ignoring {N - density_mask.sum()} atoms out of {N} total.')

    if density_threshold > 0:
        density = np.abs(M[:,n])**2
        density_mask = (density > density_threshold) 

    for k, l in enumerate(unique_labels):
        mask = (labels==l)  
        if density_threshold > 0:
            mask *= density_mask
        if hyperlocal == 'all':
            centers[k,:] = qcm.MO_com_hyperlocal(pos[mask,:],M[mask,:],n)#,eps_rho=density_threshold)
            radii[k] = qcm.MO_rgyr_hyperlocal(pos[mask,:], M[mask,:], n)#,eps_rho=density_threshold)
        elif hyperlocal == 'sites':
            centers[k,:] = qcm.MO_com_hyperlocal(pos[mask,:],M[mask,:],n)#,eps_rho=density_threshold)
            radii[k] = qcm.MO_rgyr(pos[mask,:], M[mask,:], n, renormalise=True)#,eps_rho=density_threshold)
        elif hyperlocal == 'radii':
            centers[k,:] = qcm.MO_com(pos[mask,:],M[mask,:],n, renormalise=True)#,eps_rho=density_threshold)
            radii[k] = qcm.MO_rgyr_hyperlocal(pos[mask,:], M[mask,:], n)#,eps_rho=density_threshold)
        else: #no hyperlocal
            centers[k,:] = qcm.MO_com(pos[mask,:],M[mask,:],n, renormalise=True)#,eps_rho=density_threshold)
            radii[k] = qcm.MO_rgyr(pos[mask,:], M[mask,:], n, center_of_mass=centers[k,:],renormalise=True)#,eps_rho=density_threshold)
        
        if flagged_labels is not None and l in flagged_labels:
            # if l labels an empty cluster (i.e. devoid of a priori loc centers) and the radius is too big, ignore that site
            rrr = radii[k]
            flagged_radii[m] = rrr
            if radii[k] > max_r: 
                radii[k] = 0

    if flagged_labels is not None and len(flagged_labels) > 0:
        print('flagged radii = ', flagged_radii)
        # print(f'MAX flagged radius = {np.max(flagged_radii)} ; MEAN flagged radius = {np.mean(flagged_radii)}')
        
    # filter nans induced by wavefunction re-normalisation
    inan = np.any(np.isnan(centers),axis=1) + np.isnan(radii)

    # filter zero radii/centers due to density thresholding
    izero = np.all(centers==0,axis=1) + (radii == 0)
    trash_mask = inan + izero

    print(f'Removing {trash_mask.sum()} site/radius pairs')
    
    if return_labels:
        return centers[~trash_mask,:], radii[~trash_mask], unique_labels[~trash_mask]
    else:
        return centers[~trash_mask,:], radii[~trash_mask]

def generate_sites_radii_list(pos,M,L,R,energies,nbins=100,threshold_ratio=0.30, minimum_distance=20.0, shift_centers=False, hyperlocal='sites',
                              flag_empty_clusters = False, radii_rho_threshold=0, max_r=50,return_labelled_atoms=False):
    
    N = M.shape[0]

    out_size = 10*M.shape[1]
    centres = np.zeros((out_size,2)) #setting centers = [0,0] allows us to use np.vstack when constructing centres array
    radii = np.zeros(out_size)
    ee = np.zeros(out_size) 
    inds = np.zeros(out_size, dtype='int')
    
    if return_labelled_atoms:
        all_labels = np.zeros((M.shape[1],N),dtype='int')
    
    nsites = 0

    for n in range(M.shape[1]):
        print(f'**** Getting sites for MO {n} ****')
        cc, rho, xedges, yedges = get_MO_loc_centers_opt(pos,M,n,nbins,threshold_ratio,min_distance=minimum_distance,shift_centers=shift_centers)
        if n in L:
            cc = correct_peaks(cc, pos, rho, xedges, yedges,'L',shift_centers=shift_centers)
        
        elif n in R:
            cc = correct_peaks(cc, pos, rho, xedges, yedges,'R',shift_centers=shift_centers)
        
        psi = M[:,n]
        # with objmode(labels_kmeans=)
        if flag_empty_clusters:
            _ , labels_kmeans, flagged_l = assign_AOs(pos, cc, psi, psi_pow=4, flag_empty_clusters=flag_empty_clusters)
            print('[generate_sites_radii_list] Flagged labels = ', flagged_l)
        else:
            _ , labels_kmeans = assign_AOs(pos, cc, psi, psi_pow=4, flag_empty_clusters=flag_empty_clusters)
            flagged_l = None
        
        if return_labelled_atoms:
            final_sites, rr, unique_labels = site_radii(pos,M,n,labels_kmeans,hyperlocal=hyperlocal,density_threshold=radii_rho_threshold,flagged_labels=flagged_l,max_r=max_r,return_labels=True)
            unique_labels = set(unique_labels)
            for k in range(N):
                if labels_kmeans[k] not in unique_labels:
                    labels_kmeans[k] = -1
            # labels_kmeans[labels_kmeans not in unique_labels] = -1
        else:
            final_sites, rr = site_radii(pos,M,n,labels_kmeans,hyperlocal=hyperlocal,density_threshold=radii_rho_threshold,flagged_labels=flagged_l,max_r=max_r)
        
        print('FINAL SITES = ', final_sites)
        print('FINAL RADII = ', rr)
        print('\n')

        
        n_new = final_sites.shape[0]

        # If arrays are about to overflow, copy everything into bigger arrays
        if nsites + n_new > out_size:
            print("~~~!!!!! Site arrays about to overflow; increasing output size !!!!!~~~")
            out_size += 10*M.shape[1]

            new_centres = np.zeros((out_size,2))
            new_centres[:nsites,:] = centres[:nsites,:]
            centres = new_centres
             
            new_radii = np.zeros(out_size)
            new_radii[:nsites] = radii[:nsites]
            radii = new_radii

            new_ee = np.zeros(out_size)
            new_ee[:nsites] = ee[:nsites]
            ee = new_ee
            
            new_inds = np.zeros(out_size,dtype='int')
            new_inds[:nsites] = inds[:nsites]
            inds = new_inds


        centres[nsites:nsites+n_new,:] = final_sites
        radii[nsites:nsites+n_new] = rr
        ee[nsites:nsites+n_new] = np.ones(n_new) * energies[n]
        inds[nsites:nsites+n_new] = np.ones(n_new,dtype='int') * n
        if return_labelled_atoms:
            all_labels[n,:] = labels_kmeans

        nsites += n_new
        # print(nsites)

    if return_labelled_atoms: 
        return centres[:nsites,:], radii[:nsites], ee[:nsites], inds[:nsites], all_labels #get rid of 'empty' values in output arrays
    else:
        return centres[:nsites,:], radii[:nsites], ee[:nsites], inds[:nsites] #get rid of 'empty' values in output arrays


def generate_sites_radii_list_naive(pos,M,L,R,energies,nbins=100,threshold_ratio=0.30, minimum_distance=20.0, shift_centers=False,
                               psi_pow=2, radii_rho_threshold=0):
    out_size = 10*M.shape[1]
    centres = np.zeros((out_size,2)) #setting centers = [0,0] allows us to use np.vstack when constructing centres array
    radii = np.zeros(out_size)
    ee = np.zeros(out_size) 
    inds = np.zeros(out_size, dtype='int')
    nsites = 0
    for n in range(M.shape[1]):
        print(f'**** Getting sites for MO {n} ****')
        cc, rho, xedges, yedges = get_MO_loc_centers_opt(pos,M,n,nbins,threshold_ratio,min_distance=minimum_distance,shift_centers=shift_centers)
        if n in L:
            cc = correct_peaks(cc, pos, rho, xedges, yedges,'L',shift_centers=shift_centers)
        
        elif n in R:
            cc = correct_peaks(cc, pos, rho, xedges, yedges,'R',shift_centers=shift_centers)
        
        # with objmode(labels_kmeans=)
        rr = site_radii_naive(pos, M, n, cc, density_threshold=radii_rho_threshold, psi_pow=psi_pow)
        print('CENTERS = ', cc)
        print('RADII = ', rr)
        print('\n')
        
        n_new = rr.shape[0]

        # If arrays are about to overflow, copy everything into bigger arrays
        if nsites + n_new > out_size:
            print("~~~!!!!! Site arrays about to overflow; increasing output size !!!!!~~~")
            out_size += 10*M.shape[1]

            # Allocate larger arrays and copy old arrays into newly allocated arrs
            new_centres = np.zeros((out_size,2))
            new_centres[:nsites,:] = centres[:nsites,:]
            centres = new_centres
             
            new_radii = np.zeros(out_size)
            new_radii[:nsites] = radii[:nsites]
            radii = new_radii

            new_ee = np.zeros(out_size)
            new_ee[:nsites] = ee[:nsites]
            ee = new_ee
            
            new_inds = np.zeros(out_size,dtype='int')
            new_inds[:nsites] = inds[:nsites]
            inds = new_inds

        centres[nsites:nsites+n_new,:] = cc
        radii[nsites:nsites+n_new] = rr
        ee[nsites:nsites+n_new] = np.ones(n_new) * energies[n]
        inds[nsites:nsites+n_new] = np.ones(n_new,dtype='int') * n

        nsites += n_new
        # print(nsites)

    return centres[:nsites,:], radii[:nsites], ee[:nsites], inds[:nsites] #get rid of 'empty' values in output arrays

@njit
def pair_dists_w_inds(points):
    N = points.shape[0]
    ndists = N*(N-1) // 2
    dists = np.zeros(ndists)
    inds = np.zeros((ndists,2),dtype='int')
    k = 0
    for i in range(N):
        for j in range(i):
            dists[k] = np.linalg.norm(points[i,:] - points[j,:])
            inds[k,0] = i
            inds[k,1] = j
            k += 1
    return dists, inds


@njit
def clean_centers(cc,ipeaks,grid_rho,min_dist=30.0):
    """"This function removes sites that are too close to eachother (and therefore likely belong to the same localisation pocket).
        N.B. This function returns INDICES of centers, not the centers themselves."""
    N = cc.shape[0]
    cc_dists_flat, ij = pair_dists_w_inds(cc)
    cc_dists = np.zeros((N,N))
    for d, ij in zip(cc_dists_flat,ij):
        i,j = ij
        cc_dists[i,j] = d
        cc_dists[j,i] = d
    np.fill_diagonal(cc_dists,np.max(cc_dists)*1000)
    adj_mat = (cc_dists < min_dist)
    if np.any(adj_mat):
        connected_sets = jitted_components(adj_mat) # find set of sites that are all within each other's neighbourhood
        site_densities = np.array([grid_rho[i,j] for (i,j) in ipeaks])
        ikeep = np.zeros(len(connected_sets),dtype='int')
        for k, c in enumerate(connected_sets):
            c = list(c)
            densities = np.array([site_densities[n] for n in c])
            ikeep[k] = c[np.argmax(densities)]
        return ikeep
    else:
        return np.arange(N)



def sites_mass(psi,tree,centers,radii):
    masses = np.zeros(radii.shape[0])
    atoms_in_radii = tree.query_ball_point(centers,radii)
    for k, ii in enumerate(atoms_in_radii):
        masses[k] = np.sum(np.abs(psi[ii])**2)
    return masses


# @njit
def all_sites_ipr(M,labels,eps_rho=0):
    """Computes the inverse participation ratios of the sub-MOs derived from ALL of a given structure's
    MOs."""
    nsites = 0
    for ll in labels:
        nsites += np.unique(ll[ll != -1]).shape[0] # filter out -1 ('bad' label) from unique(ll) when evalutating nsites
    iprs = np.zeros(nsites)
    k = 0
    # Loop over MOs
    for n in range(M.shape[1]):
        print('Getting IPRs of sites from MO nb ', n)
        ll = labels[n]
        nclusters = np.unique(ll[ll != -1]).shape[0]
        psi = M[:,n]
        iprs[k:k+nclusters] = sites_ipr(psi,ll,eps_rho=eps_rho)

        k += nclusters

    return iprs
        



# @njit
def sites_ipr(psi,labels,eps_rho=0):
    """Computes the IPRs of the sub-MOs obtained by partitioning a SINGLE MO."""
    if eps_rho > 0:
        psi[psi**2 < eps_rho] = 0
        psi /= (psi**2).sum()

    # Ignore -1 label: corresponds to 'bad sites' (see `generate_site_radii_list`)
    nn = np.unique(labels[labels != -1])
    iprs = np.zeros(nn.shape[0])

    for k, n in enumerate(nn):
        mask = (labels == n)
        sub_psi = psi[mask]
        print(np.sum(psi**2))
        iprs[k] = np.sum(sub_psi**4) / np.sum(sub_psi**2)
    
    return iprs