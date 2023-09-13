#!/usr/bin/env python

from os import path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.animation as animation
from qcnico import plt_utils
from qcnico.coords_io import read_xsf



def make_full_traj(pts,m,q):
    """
    This function 'fleshes out' a hopping trajectory by adding `m` intermediate frames per unit distance 
    in between two consecutive sites in the hopping trajectory.
    Once the electron lands on a site, it stays there for `q` additional timeframes (we don't add this delay
    for the initial site).
    The returned array therefore has shape ((N-1)*m + (N-1)*q, 3), where pts.shape[0] = N.
    The first two elements of each row are the (x,y) coords of the hopper at each timeframe.
    The third element is 1 if the hopper is on a site and 0 if the hopper is transitting between two sites
    (this is useful for the `update` function below).
    """
    npts = pts.shape[0]
    print('npts = ', npts)
    nb_intermediate_frames = (np.ceil(np.linalg.norm(np.diff(pts,axis=0),axis=1)) * m).astype('int')
    full_traj = np.zeros(((nb_intermediate_frames.sum()+(npts-1)*q),3))
    full_traj[0,2] = 1 #initial position is a hopping site
    cnt = 0
    for k in range(npts-1):
        ri = pts[k,:]
        rf = pts[k+1,:]

        M = nb_intermediate_frames[k]
        t = np.linspace(0,1,M,endpoint=True)

        hop_pts = ri + t[:,None]*(rf - ri)
        
        full_traj[cnt:cnt+M,:2] = hop_pts
        full_traj[cnt+M:cnt+M+q,:2] = rf
        full_traj[cnt+M:cnt+M+q,2] = 1
        cnt += M+q
        # yield new_pts 
    return full_traj

def find_moves(traj):
    """This function finds the points at which the walker changes positions"""
    npts = traj.shape[0]
    diffs = np.diff(traj,axis=0)
    newx = diffs[:,0].nonzero()[0]
    newy = diffs[:,1].nonzero()[0]
    return newx, newy
    


def update(frame):
    """In this case, each frame is defined by the position of the random walker."""
    r = full_traj[frame,:2]
    on_site = full_traj[frame,2]
    # print(r)
    ye.set_offsets(r)
    ye.set_color('r')
    if on_site == 1:
        ye.set_sizes([50.0])
    else:
        ye.set_sizes([24.0])
    return ye


def sample_traj(traj):
    """Gets all of the points of the trajectory where the x coord changes."""
    newx, _ = find_moves(traj)
    npts_sampled = newx.shape[0] * 2
    sampled_pts = np.ones((npts_sampled,2)) * -1000 # assume -1000 will not come up organically in our data set
    j = 0
    k = 0
    while k<npts_sampled:
        n = newx[j]
        proposed_pts = traj[[n,n+1],:]
        if np.all(proposed_pts[0,:] == sampled_pts[k-1,:]): #if we have a repeat of the previous pt
            sampled_pts[k+1,:] = proposed_pts[1,:] #only update the second point
        else:
            sampled_pts[[k,k+1],:] = traj[[n,n+1],:]
        k+=2
        j+=1
    
    # Remove all unassigned points
    good = np.all(sampled_pts != -1000, axis=1)
    print(good)
    return sampled_pts[good]



# Get data
datadir=path.expanduser("~/Desktop/simulation_outputs/percolation/40x40/percolate_output")

nn = 150
T = 410
kB = 8.617e-5

posdir = path.join(path.dirname(datadir), 'structures')
Mdir = path.join(path.dirname(datadir), 'MOs_ARPACK')
edir = path.join(path.dirname(datadir), 'eARPACK')
trajdir = '/Users/nico/Desktop/simulation_outputs/percolation/40x40/monte_carlo/marcus/trajectories/local/'
trajfile = trajdir + f'sample-{nn}_traj_{T}K.npy'

posfile = path.join(posdir,f'bigMAC-{nn}_relaxed.xsf')
Mfile = path.join(Mdir,f'MOs_ARPACK_bigMAC-{nn}.npy')
efile = path.join(edir, f'eARPACK_bigMAC-{nn}.npy')
ccfile = path.join(datadir,f'sample-{nn}','cc.npy')
iifile = path.join(datadir,f'sample-{nn}','ii.npy')
eefile = path.join(datadir,f'sample-{nn}','ee.npy')


M = np.load(Mfile)
centres = np.load(ccfile)
MOinds = np.load(iifile)
energies = np.load(eefile)
pos, _ = read_xsf(posfile)

early_inds = np.arange(10)
mid_inds1 = np.arange(300,310)
mid_inds2 = np.arange(600,610)
end_inds = np.arange(1000,1224)

itraj = np.load(trajfile)
traj = centres[itraj]


# j = 0
# k = 0
# while k<npts_sampled:
#     n = newx[j]
#     if n > 2:
#         same_x = np.argsort((n-newy)[n-newy > 0])[:2] # get closest pts to n who precede it in newy ==> they will have the same x, but different y
#     else:
#         same_x = np.argsort((newy-n)[(newy-n)>0])[:2]
#     sampled_pts[[k,k+1],:] = traj[same_x,:]
#     sampled_pts[k+2] = traj[n,:]
#     sampled_pts[k+3] = traj[n+1,:]
#     k+=4
#     j+=1

sampled_pts = sample_traj(traj)

print(np.any(np.all(sampled_pts==0,axis=1)))

full_traj = make_full_traj(sampled_pts,0.5,3)[:300]
print('*******************')
print(full_traj.shape)

rho = np.sum(M[:,np.unique(MOinds)]**2,axis=1)

plt_utils.setup_tex()


rcParams['font.size'] = 20
rcParams['figure.figsize'] = [10,5]
rcParams['figure.dpi'] = 200.0
rcParams['figure.constrained_layout.use'] = True
fig, ax = plt.subplots()
ye = ax.scatter(pos.T[0], pos.T[1], c=rho, s=1.0, cmap='plasma',zorder=1)
fig.patch.set_alpha(0.0)
# cbar = fig.colorbar(ye,ax=ax,orientation='vertical')
ax.set_aspect('equal')
ax.set_xlabel("$x$ [\AA]")
ax.set_ylabel("$y$ [\AA]")
ani = animation.FuncAnimation(fig=fig,func=update,frames=full_traj.shape[0],repeat=False)
ani.save(filename='hop_traj.gif',writer='pillow',savefig_kwargs={"transparent":True,"facecolor":"none"})
plt.show()
