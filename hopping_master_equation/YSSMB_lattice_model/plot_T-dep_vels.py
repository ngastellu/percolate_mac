#!/usr/bin/env python 

import numpy as np
import matplotlib.pyplot as plt


temps = np.arange(100,410,10)

vels = np.load('all_velocities.npy')

plt.plot(1000.0/temps, np.log(vels))
plt.show()

plt.plot((1.0/temps)**(1/3), np.log(vels))
plt.show()
