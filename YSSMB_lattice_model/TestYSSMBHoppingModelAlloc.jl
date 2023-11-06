module TestYSSMBHoppingModelAlloc

include("./HoppingMasterEquation.jl")

using .HoppingMasterEquation, PyCall

density = 6.9e-6 #in Å^{-3}
temps = collect(100:25:125)
ν = 0.007  # picked this stdev to roughly match the Gaussian DOS in the paper (DOI: 10.1103/PhysRevB.63.085202)
Nx = 32
Nyz = 16 

workdir = "YSSMB_test_alloc"
if !isdir(workdir)
    mkdir(workdir)
end
cd(workdir)

energies, velocities, Pinits, Pfinals = YSSMB_lattice_model(temps,density; νeff = ν, N1=Nx, N2=Nyz, e_corr=false)

py"""import numpy as np
np.save('temps.npy', $(PyObject(temps)))
np.save('energies.npy', $(PyObject(energies)))
np.save('velocities.npy', $(PyObject(velocities)))
np.save('Pinits.npy', $(PyObject(Pinits)))
np.save('Pfinals.npy', $(PyObject(Pfinals)))"""


end