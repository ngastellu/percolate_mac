module TestCorrelatedEnergies

include("./HoppingMasterEquation.jl")

using .HoppingMasterEquation, PyCall

N = 32

pos = zeros(N,N,N,3)
a = 10
T = 300
K = 0.0034
ν = 0.3

Ω = ((N-1) * a)^3

for i=1:N
    for j=1:N
        for k=1:N
            pos[i,j,k,:] = [i,j,k] .-1
        end
    end
end

pos = reshape(pos, N*N*N, 3) .* a

energies = generate_correlated_esites(pos, a, Ω, T, K, ν)

py"""import numpy as np
np.save('correlated_energies.npy', $(PyObject(energies)))
"""

end