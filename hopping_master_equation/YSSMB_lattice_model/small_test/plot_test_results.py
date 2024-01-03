#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from qcnico.plt_utils import setup_tex, histogram



nseed = 64

conv = np.load(f'conv-{nseed}.npy')


setup_tex()

plt.plot(np.log(conv[:,0]))
plt.show()