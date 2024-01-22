module TestFullDevice

include("../RunFullDevice.jl")

using Random, PyCall, .RunFullDevice

    for rnseed ∈ [10]
        # rnseed = 64
            println("********** $rnseed **********")
            Random.seed!(rnseed) # seed RNG with task number so that we have the same disorder realisation at each T
    
    
            T = 300
    
            # nocc = 50
            # ν = 0.007  # picked this stdev to roughly match the Gaussian DOS in the paper (DOI: 10.1103/PhysRevB.63.085202)
            ν = 0.007
            Nx = 100
            Ny = 100
            Nz = 20
            lattice_dims = (Nx,Ny)#,Nz)
            d = size(lattice_dims,1)
            ΔV = 0
            T = 300
            dos_type = "gaussian"
            save_each = 1
            
            energies, pos, current, Pinits, Pfinals, conv = run_full_device_lattice(lattice_dims,ΔV,T,dos_type,ν; rcut_ratio=sqrt(3),save_each=-1,maxiter=10000)
    
            py"""import numpy as np
            nn= $rnseed
            dd = $d
            np.save(f'full_device_test/{dd}d/pos.npy', $(PyObject(pos)))
            np.save(f'full_device_test/{dd}d/energies-{nn}.npy', $(PyObject(energies)))
            np.save(f'full_device_test/{dd}d/J-{nn}.npy', $(PyObject(current)))
            np.save(f'full_device_test/{dd}d/Pinits-{nn}.npy', $(PyObject(Pinits)))
            np.save(f'full_device_test/{dd}d/Pfinals-{nn}.npy', $(PyObject(Pfinals)))
            np.save(f'full_device_test/{dd}d/conv-{nn}.npy', $(PyObject(conv)))
            #np.save(f'full_device_test/{dd}d/Pt-{nn}.npy', $(PyObject(Pt)))
            """   
        end
end