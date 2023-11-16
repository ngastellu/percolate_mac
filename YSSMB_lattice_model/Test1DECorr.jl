module Test1DECorr

include("../HoppingMasterEquation.jl")

using .HoppingMasterEquation, PyCall


nsamples = 300

a = 10
T = 300
K = 0.0034
ν = 0.3

for N ∈ [32, 64, 128, 256, 512]
    for i=1:nsamples
        println(i)
        energies, Φ = generate_correlated_esites_1d(a, N, T, K, ν;even=false)
        # energies = real(energies)

        py"""import numpy as np
        import os 
        nn = $i
        n = $N
        if not os.path.exists(f'corr_energies_1d_asymm_{n}/'):
            os.mkdir(f'corr_energies_1d_asymm_{n}')
        if not os.path.exists(f'ft_mol_geom_field_1d_asymm_{n}'):
            os.mkdir(f'ft_mol_geom_field_1d_asymm_{n}') 
        np.save(f'corr_energies_1d_asymm_{n}/correlated_energies-{nn}.npy', $(PyObject(energies)))
        np.save(f'ft_mol_geom_field_1d_asymm_{n}/mgf_ft-{nn}.npy', $(PyObject(Φ)))
        """
    end
end

end