module TestInitFullDevice

    include("../FullDeviceUtils.jl")
    include("../Utils.jl")

    using PyCall, Random, .FullDeviceUtils, .Utils

    Random.seed!(0)
    
    ν = 0.007  # picked this stdev to roughly match the Gaussian DOS in the paper (DOI: 10.1103/PhysRevB.63.085202)
    Nx = 20
    Ny = 20
    Nz = 20
    a = 5
    eL = 0
    eR = 0
    T = 300
    E0 = 1.0
    rcut = a

    pos2 = create_2d_lattice(Nx,Ny,a;full_device=true)
    pos3 = create_3d_lattice(Nx,Ny,Nz,a;full_device=true)

    nn2 = get_neighbour_lists_fdev(pos2,a,Ny;max_nn_estimate=10)
    nn3 = get_neighbour_lists_fdev(pos3,a,Ny*Nz;max_nn_estimate=20)


    ld2 = (Nx,Ny)
    ld3 = (Nx,Ny,Nz)

    ig = ghost_inds_3d(pos3,ld3...,a;full_device=true)

    e2 = initialise_energies_fdev(pos2,ld2,a,eL,eR,E0,"gaussian",ν)
    e3 = initialise_energies_fdev(pos3,ld3,a,eL,eR,E0,"gaussian",ν; ghost_inds = ig)

    p2 = initialise_p_fdev(ld2,e2,T)
    p3 = initialise_p_fdev(ld3,e3,T)



    py"""
    import numpy as np
    np.save('full_device_test/2d/init/pos.npy', $(PyObject(pos2)))
    np.save('full_device_test/2d/init/energies.npy', $(PyObject(e2)))
    np.save('full_device_test/2d/init/prob.npy', $(PyObject(p2)))
    np.save('full_device_test/2d/init/nn2.npy', $(PyObject(nn2)))
    np.save('full_device_test/3d/init/pos.npy', $(PyObject(pos3)))
    np.save('full_device_test/3d/init/energies.npy', $(PyObject(e3)))
    np.save('full_device_test/3d/init/prob.npy', $(PyObject(p3)))
    np.save('full_device_test/3d/init/ighost.npy', $(PyObject(ig)))
    """

end