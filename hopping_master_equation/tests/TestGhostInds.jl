module TestGhostInds

    include("../Utils.jl")
    using .Utils

    Nx = 10 
    Ny = 5
    Nz = 8
    a = 10

    # pos = create_2d_lattice(Nx,Ny,a)
    # N = size(pos,1)
    # for i=1:N
    #     println(pos[i,:])
    # end

    pos = create_3d_lattice(Nx,Ny,Nz,a)

end