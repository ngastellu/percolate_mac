module Test1DECorr

include("../HoppingMasterEquation.jl")

using .HoppingMasterEquation, PyCall


nsamples = 300

a = 10
T = 300
K = 0.0034
ν = 0.3

N = 1024
for i=1:nsamples
    println(i)
    energies, Φ = generate_correlated_esites_1d(a, N, T, K, ν)
    # energies = real(energies)

    py"""import numpy as np
    import os 
    nn = $i
    n = $N
    ecorr_dir = f'corr_energies_1d_complex_fftshift_{n}/'
    mgfft_dir = f'ft_mol_geom_field_1d_complex_fftshift_{n}/'
    if not os.path.exists(ecorr_dir):
        os.mkdir(ecorr_dir)
    if not os.path.exists(mgfft_dir):
        os.mkdir(mgfft_dir) 
    np.save(ecorr_dir + f'correlated_energies-{nn}.npy', $(PyObject(energies)))
    np.save(mgfft_dir + f'mgf_ft-{nn}.npy', $(PyObject(Φ)))
    """
end

end
