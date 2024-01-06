module TestLattices

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
    nghosts = 0
    nghosts_estimate = Nx*Ny + Nx*Nz + Ny*Nz - (Nx + Ny + Nz) + 1
    for i=1:Nx*Ny*Nz
        if pos[i,1] == (Nx-1)*a || pos[i,2] == (Ny-1)*a || pos[i,3] == (Nz-1)*a 
            println(pos[i,:])
            global nghosts += 1
        end
    end
    println(nghosts)
    println(nghosts_estimate)
    println("Error in nghosts estimate = $(nghosts - nghosts_estimate)")

end