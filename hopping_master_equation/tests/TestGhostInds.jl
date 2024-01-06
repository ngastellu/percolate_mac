module TestGhostInds

    include("../Utils.jl")
    using .Utils

    Nx = 10 
    Ny = 5
    Nz = 8
    a = 10
    full_device = true

    pos = create_3d_lattice(Nx,Ny,Nz,a)
    if full_device
        pos[:,1] = pos[:,1] .- 2.0*a
    end
    gg = ghost_inds_3d(pos,Nx,Ny,Nz,a;full_device=full_device)

    for ij in eachrow(gg)
        i,j = ij
        println("Ghost inds =  ($i,$j) ---> ($(pos[i,:]), $(pos[j,:]))")
    end

end