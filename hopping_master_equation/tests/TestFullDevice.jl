module TestFullDevice

include("../RunFullDevice.jl")

using Random, PyCall, .RunFullDevice

    for rnseed ∈ [0, 42, 64, 78]
        # rnseed = 64
            println("********** $rnseed **********")
            Random.seed!(rnseed) # seed RNG with task number so that we have the same disorder realisation at each T
    
    
            T = 300
    
            # nocc = 50
            ν = 0.007  # picked this stdev to roughly match the Gaussian DOS in the paper (DOI: 10.1103/PhysRevB.63.085202)
            Nx = 10
            Ny = 10
            Nz = 10
            lattice_dims = (Nx,Ny,Nz)
            ΔV = 1.0
            T = 300
            dos_type = "gaussian"
            save_each = 1
            
            energies, velocities, Pinits, Pfinals, rates, conv, Pt = run_full_device_lattice(lattice_dims,ΔV,T,dos_type,ν; cutoff_ratio=1.0)
    
            py"""import numpy as np
            nn= $rnseed
            dd = $d
            np.save(f'full_device_test/{dd}d/energies-{nn}.npy', $(PyObject(energies)))
            np.save(f'full_device_test/{dd}d/velocities-{nn}.npy', $(PyObject(velocities)))
            np.save(f'full_device_test/{dd}d/Pinits-{nn}.npy', $(PyObject(Pinits)))
            np.save(f'full_device_test/{dd}d/Pfinals-{nn}.npy', $(PyObject(Pfinals)))
            np.save(f'full_device_test/{dd}d/rates-{nn}.npy', $(PyObject(rates)))
            np.save(f'full_device_test/{dd}d/conv-{nn}.npy', $(PyObject(conv)))
            np.save(f'full_device_test/{dd}d/Pt-{nn}.npy', $(PyObject(Pt)))
            """   
        end
end