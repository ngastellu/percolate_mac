#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from qcnico import plt_utils


max_dP, avg_dP, argmax_dP = np.load('../full_device_test/2d/conv-10.npy').T


plt_utils.setup_tex(fontsize=25.0)

fig, ax = plt.subplots()

color_max = 'tab:red'
ax.plot(max_dP,'-',color=color_max,lw=0.8)
ax.set_xlabel('Iteration $n$')
ax.set_ylabel('$\max_{i}\{|p^{(n+1)}_{i} - p_{i}^{(n)}|\}$', color=color_max)
ax.tick_params(axis='y',color=color_max)
plt.show()

N = 100 * 100
occurences = np.array([(argmax_dP == n).sum() for n in range(1,N+1)])
pos = np.load('../full_device_test/2d/pos.npy')
sizes = np.ones(N)*10
itrouble = (occurences > 200).nonzero()[0]
sizes[itrouble] = 50.0

fig, ax = plt.subplots()
ye = ax.scatter(*pos.T,c=occurences,cmap='inferno',s=sizes)
cbar = fig.colorbar(ye, ax=ax)
plt.show()


plt.plot(occurences)
plt.show()

# color_avg = 'tab:blue'
# ax2 = ax.twinx()
# ax2.plot(avg_dP,'-',color=color_avg, lw=0.8)
# ax2.set_ylabel('$\langle|\Delta p_{i}|\\rangle$',color=color_avg)
# ax2.tick_params(axis='y',color=color_avg)

plt.show()


