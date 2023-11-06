module RunYSSMBHoppingModelMPI

include("./HoppingMasterEquation.jl")

using .HoppingMasterEquation, PyCall, MPI, Random

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nprocs = MPI.Comm_size(comm)

task_nb = parse(Int, ARGS[1])

Random.seed!(task_nb) # seed RNG with task number so that we have the same disorder realisation at each T

all_temps = collect(100:10:400)
nT = size(all_temps,1)
nT_per_proc = Int(floor(nT / nprocs))

# if rank < nprocs - 1
#     temps = all_temps[1+(rank*nT_per_proc) : (rank+1)*nT_per_proc]
# else
#     temps = all_temps[1+(rank*(nT_per_proc):end)]
# end

T = [all_temps[rank+1]]


density = 6.9e-6 #in Å^{-3}
ν = 0.007  # picked this stdev to roughly match the Gaussian DOS in the paper (DOI: 10.1103/PhysRevB.63.085202)


energies, velocities, Pinits, Pfinals = YSSMB_lattice_model(T,density; νeff = ν, e_corr=false)

py"""import numpy as np
nn = $rank
np.save(f'T-{nn}.npy', $(PyObject(T)))
np.save(f'energies-{nn}.npy', $(PyObject(energies)))
np.save(f'velocities-{nn}.npy', $(PyObject(velocities)))
np.save(f'Pinits-{nn}.npy', $(PyObject(Pinits)))
np.save(f'Pfinals-{nn}.npy', $(PyObject(Pfinals)))"""   

end
