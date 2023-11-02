module TestCorrelatedEnergies

include("./HoppingMasterEquation.jl")

using .HoppingMasterEquation, PyCall

N1 = 4
N2 = 2

pos = zeros(N1,N2,N2,3)
a = 10
T = 300
K = 0.0034
ν = 0.3

Ω = ((N1-1) * a) * ((N2-1) * a) * ((N2-1) * a)

for i=1:N1
    for j=1:N2
        for k=1:N2
            pos[i,j,k,:] = [i,j,k] .-1
        end
    end
end

pos = reshape(pos, N1*N2*N2, 3) .* a

energies = generate_correlated_esites(pos, a, N1, N2, Ω, T, K, ν)

py"""import numpy as np
np.save('correlated_energies.npy', $(PyObject(energies)))
"""

end