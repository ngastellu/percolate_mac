module RunYSSMBHoppingModelMPI_varE

include("./HoppingMasterEquation.jl")

using .HoppingMasterEquation, PyCall, MPI, Random

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nprocs = MPI.Comm_size(comm)

task_nb = parse(Int, ARGS[1])

Random.seed!(task_nb) # seed RNG with task number so that we have the same disorder realisation at each T

all_efields = collect(0.0004:0.0004:0.0200)
nE = size(all_temps,1)
nE_per_proc = Int(floor(nE / nprocs))

# if rank < nprocs - 1
#     temps = all_temps[1+(rank*nT_per_proc) : (rank+1)*nT_per_proc]
# else
#     temps = all_temps[1+(rank*(nT_per_proc):end)]
# end

E = [all_temps[rank+1]]


density = 6.9e-6 #in Å^{-3}
ν = 0.007  # picked this stdev to roughly match the Gaussian DOS in the paper (DOI: 10.1103/PhysRevB.63.085202)


energies, velocities, Pinits, Pfinals = YSSMB_lattice_model_varE(E,density; νeff = ν, e_corr=false)

py"""import numpy as np
nn = $rank
np.save(f'E-{nn}.npy', $(PyObject(E)))
np.save(f'energies-{nn}.npy', $(PyObject(energies)))
np.save(f'velocities-{nn}.npy', $(PyObject(velocities)))
np.save(f'Pinits-{nn}.npy', $(PyObject(Pinits)))
np.save(f'Pfinals-{nn}.npy', $(PyObject(Pfinals)))"""   

end
