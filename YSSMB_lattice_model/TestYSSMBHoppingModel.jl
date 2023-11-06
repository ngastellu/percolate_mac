module TestYSSMBHoppingModel

include("./HoppingMasterEquation.jl")

using .HoppingMasterEquation, PyCall

ntrials = 100
density = 6.9e-6 #in Å^{-3}
temps = collect(100:25:400)
ν = 0.007  # picked this stdev to roughly match the Gaussian DOS in the paper (DOI: 10.1103/PhysRevB.63.085202)
Nx = 32
Nyz = 16 

workdir = "YSSMB_test_uncorr"
if !isdir(workdir)
    mkdir(workdir)
end
cd(workdir)

for i=1:ntrials
    sampledir = "sample-$i"
    if !isdir(sampledir)
        mkdir(sampledir)
    end
    cd(sampledir)

    energies, velocities, Pinits, Pfinals = YSSMB_lattice_model(temps,density; νeff = ν, N1=Nx, N2=Nyz)

    py"""import numpy as np
    np.save('temps.npy', $(PyObject(temps)))
    np.save('energies.npy', $(PyObject(energies)))
    np.save('velocities.npy', $(PyObject(velocities)))
    np.save('Pinits.npy', $(PyObject(Pinits)))
    np.save('Pfinals.npy', $(PyObject(Pfinals)))"""
    
    cd("..")
end


end