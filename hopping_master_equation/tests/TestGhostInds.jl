module TestGhostInds

    include("../Utils.jl")
    using .Utils

    Nx = 10 
    Ny = 10
    Nz = 10
    a = 10
    full_device = true

    N = Nx*Ny*Nz

    pos = create_3d_lattice(Nx,Ny,Nz,a)
    if full_device
        pos[:,1] = pos[:,1] .- 2.0*a
    end
    gg = ghost_inds_3d(pos,Nx,Ny,Nz,a;full_device=full_device)

    for ij in eachrow(gg)
        i,j = ij
        println("Ghost inds =  ($i,$j) ---> ($(pos[i,:]), $(pos[j,:]))")
    end

    println("\n-----------------\n")

    for i=1:N
        check = gg .- i
        image_check = findall(iszero, check[:,2])
        if size(image_check,1) > 0
            ghost_inds = [gg[k,1] for k ∈ image_check]
            println("$i -----> $image_check")
            println("Ghost inds = $ghost_inds")
        end
    end


end