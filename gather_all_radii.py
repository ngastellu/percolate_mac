#!/usr/bin/env python

import numpy as np
import os
import sys

structype = os.path.basename(os.getcwd())
motype = sys.argv[1]

if structype == '40x40':
    lbls = np.arange(1,301)
else:
    lbls = np.load(f'ifiltered_MRO_{structype}.npy')


radii_arrs = []

for n in lbls:
    print(n)
    radii_arrs.append(np.load(f'sample-{n}/sites_data_{motype}/radii.npy'))

np.save(f'all_site_radii_{motype}.npy', np.hstack(radii_arrs))
    

