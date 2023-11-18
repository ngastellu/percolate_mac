module RunYSSMBHoppingModelvarE1D

include("../HoppingMasterEquation.jl")

using .HoppingMasterEquation, PyCall, Random


task_nb = parse(Int, ARGS[1])
rank = parse(Int,ARGS[2])

Random.seed!(task_nb) # seed RNG with task number so that we have the same disorder realisation at each T

all_efields = collect(0.0004:0.0004:0.0200)

E = [all_efields[rank]]


density = 6.9e-2 #in Å^{-3}
ν = 0.007  # picked this stdev to roughly match the Gaussian DOS in the paper (DOI: 10.1103/PhysRevB.63.085202)


energies, velocities, Pinits, Pfinals = YSSMB_lattice_model_varE_1d(E,density; νeff = ν, e_corr=true)

py"""import numpy as np
nn = $rank
np.save(f'E-{nn}.npy', $(PyObject(E)))
np.save(f'energies-{nn}.npy', $(PyObject(energies)))
np.save(f'velocities-{nn}.npy', $(PyObject(velocities)))
np.save(f'Pinits-{nn}.npy', $(PyObject(Pinits)))
np.save(f'Pfinals-{nn}.npy', $(PyObject(Pfinals)))"""   

end
