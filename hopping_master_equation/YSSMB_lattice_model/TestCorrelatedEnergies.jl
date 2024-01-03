module TestCorrelatedEnergies

include("../HoppingMasterEquation.jl")

using .HoppingMasterEquation, PyCall

nsamples = 200

N1 = 16
N2 = 8

pos = zeros(N1,N2,N2,3)
a = 10
T = 300
K = 0.0034
ν = 0.3

Ω = ((N1-1) * a) * ((N2-1) * a) * ((N2-1) * a)

for i=1:N1
    for j=1:N2
        for k=1:N2
            pos[i,j,k,:] = [i,j,k]
        end
    end
end

pos = reshape(pos, N1*N2*N2, 3) .* a


for i=1:nsamples
    println(i)
    energies, Φ = generate_correlated_esites(pos, a, N1, N2, Ω, T, K, ν)
    energies = real(energies)

    py"""import numpy as np
    nn = $i
    NN = $nsamples
    n1 = $N1
    n2 = $N2
    np.save(f'corr_energies_{n1}x{n2}x{n2}/correlated_energies-{nn}.npy', $(PyObject(energies)))

    if nn == NN - 1:
        np.save(f'lattice_{n1}x{n2}x{n2}.npy', $(PyObject(pos)))
        np.save(f'ft_mol_geom_field_{n1}x{n2}x{n2}-{nn}.npy', $(PyObject(Φ)))
    """
end
end