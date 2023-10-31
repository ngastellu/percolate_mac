module TestYSALMBHoppingModel

include("./HoppingMasterEquation.jl")

using .HoppingMasterEquation, PyCall

# ntrials = 1
density = 6.9e-6 #in Å^{-3}
temps = collect(100:10:400)
ν = 0.007  # picked this stdev to roughly match the Gaussian DOS in the paper (DOI: 10.1103/PhysRevB.63.085202)

workdir = "YSALMB_test"
if !isdir(workdir)
    mkdir(workdir)
end
cd(workdir)
energies, velocities, Pinits, Pfinals = YSALMB_lattice_model(temps,density; νeff = ν)

py"""import numpy as np
np.save('temps.npy', $(PyObject(temps)))
np.save('energies.npy', $(PyObject(energies)))
np.save('velocities.npy', $(PyObject(velocities)))
np.save('Pinits.npy', $(PyObject(Pinits)))
np.save('Pfinals.npy', $(PyObject(Pfinals)))"""


end