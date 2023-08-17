#!/usr/bin/env python

from os import path
import numpy as np
from pyscf import gto
from qcnico.coords_io import read_xsf


nsample = 150

posdir = path.expanduser("~/Desktop/simulation_outputs/percolation/40x40/structures/")
posfile = posdir + f"bigMAC-{nsample}_relaxed.xsf"

pos,_ = read_xsf(posfile)

mol = gto.Mole()
mol.atom = [['C', tuple(r)] for r in pos]
mol.build()

print(mol)

