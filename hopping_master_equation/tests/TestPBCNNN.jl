module TestPBCNNN
    include("../Utils.jl")

    using .Utils

    a = 10

    pos2 = create_2d_lattice(5,5,a)
    pos3 = create_3d_lattice(5,5,5,a)

    println("***** Running 2D test *****")
    innn2 = get_nnn_inds(pos2,a;pbc="full")

    println("***** Running 3D test *****")
    innn3 = get_nnn_inds(pos3,a;pbc="full")
    
end    
