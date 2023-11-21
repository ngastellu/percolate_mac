module RunYSSMBHoppingModelvarT1D

include("../HoppingMasterEquation.jl")

using .HoppingMasterEquation, PyCall, Random


task_nb = parse(Int, ARGS[1])
# rank = parse(Int,ARGS[2])

Random.seed!(task_nb) # seed RNG with task number so that we have the same disorder realisation at each T

# all_efields = collect(0.0004:0.0004:0.0200)
temeperatures = collect(100:10:400)

# E = [all_efields[rank]]


density = 6.9e-3 #in Å^{-3}
ν = 0.007  # picked this stdev to roughly match the Gaussian DOS in the paper (DOI: 10.1103/PhysRevB.63.085202)

@time energies, velocities, Pinits, Pfinals = YSSMB_lattice_model_varT_1d(temperatures,density; νeff = ν, e_corr=true)

py"""import numpy as np
np.save('energies.npy', $(PyObject(energies)))
np.save('velocities.npy', $(PyObject(velocities)))
np.save('temperatures.npy', $(PyObject(all_efields)))
np.save('Pinits.npy', $(PyObject(Pinits)))
np.save('Pfinals.npy', $(PyObject(Pfinals)))"""   

end
