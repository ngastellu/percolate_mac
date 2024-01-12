module TestFullDevice

include("../RunFullDevice.jl")

using Random, PyCall, .RunFullDevice

    rnseed = parse(Int, ARGS[1])
    println("********** $rnseed **********")
    Random.seed!(rnseed) # seed RNG with task number so that we have the same disorder realisation at each T


    T = 300

    # nocc = 50
    # ν = 0.007  # picked this stdev to roughly match the Gaussian DOS in the paper (DOI: 10.1103/PhysRevB.63.085202)
    ν = 0.007
    Nx = 104
    Ny = 100
    Nz = 100
    lattice_dims = (Nx,Ny)#,Nz)
    d = size(lattice_dims,1)
    ΔV = 0.01
    T = 300
    dos_type = "gaussian"

    
    energies, pos, current, Pinits, Pfinals, conv, Pt = run_full_device_lattice(lattice_dims,ΔV,T,dos_type,ν; rcut_ratio=sqrt(3),save_each=100,maxiter=1000000)

    py"""import numpy as np
    nn= $rnseed
    dd = $d
    np.save(f'pos.npy', $(PyObject(pos)))
    np.save(f'energies-{nn}.npy', $(PyObject(energies)))
    np.save(f'J-{nn}.npy', $(PyObject(current)))
    np.save(f'Pinits-{nn}.npy', $(PyObject(Pinits)))
    np.save(f'Pfinals-{nn}.npy', $(PyObject(Pfinals)))
    np.save(f'conv-{nn}.npy', $(PyObject(conv)))
    np.save(f'Pt-{nn}.npy', $(PyObject(Pt)))
    """   
end