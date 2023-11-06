module TestCorrelatedEnergies

include("./HoppingMasterEquation.jl")

using .HoppingMasterEquation, PyCall

N1 = 64
N2 = 32

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

energies, Φ = generate_correlated_esites(pos, a, N1, N2, Ω, T, K, ν)

py"""import numpy as np
np.save('correlated_energies.npy', $(PyObject(energies)))
np.save('ft_mol_geom_field.npy', $(PyObject(Φ)))
"""

end