#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt


N1 = 64
N2 = 32

rec_latt = np.meshgrid(np.arange(1,N1+1), np.arange(1,N2+1), np.arange(1,N2+1))

energies = np.load('correlated_energies.npy')
ft_mgf = np.load('ft_mol_geom_field.npy')