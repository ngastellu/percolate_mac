module RunFullDevice

    include("./Utils.jl")
    include("./FullDeviceUtils.jl")
    include("./YSSMBSolver.jl")

    using .Utils, .FullDeviceUtils, .YSSMBSolver, PyCall

    export run_full_device_lattice

    const kB = 8.617333262e-5 # Boltzmann constant in eV/K
    const e = 1.0 # positron charge

    function run_full_device_lattice(lattice_dims,ΔV,T,dos_type, dos_param;a=10, eL=0, eR=0,
        α_ratio=10.0, rcut_ratio=10, maxiter=1000000, save_each=0, restart_threshold=-1)
        
        N = prod(lattice_dims)
        d = size(lattice_dims,1)
        @assert d ∈ (2,3) "Only 2D and 3D implemented. (d=$d dimensions specified)"
        # Lattice creation functions are defined such that pos is sorted by ascending x-coord
        println("Creating lattice...")
        if d == 2
            pos = create_2d_lattice(lattice_dims..., a; full_device=true)
            edge_size = lattice_dims[2]
            println("Done!")
            ighost = [0]
        else
            pos = create_3d_lattice(lattice_dims..., a; full_device=true)
            edge_size = lattice_dims[2] * lattice_dims[3]
            println("Done!")
            println("Getting ghost inds...")
            ighost = ghost_inds_3d(pos,lattice_dims...,a;full_device=true)
            println("Done!")
            println("\n----- Ghost inds array -----")
            for ij ∈ eachrow(ighost)
                println(ij)
            end
            println("----------------\n")
        end

          
        ΔX = (lattice_dims[1]-4) * a # there are 4 rows of lattice are electrode sites
        E0 = ΔV / ΔX

        rcut = rcut_ratio * a
        println("Getting neighbour lists...")
        ineighbours = build_neighbour_lists_fdev(pos, rcut, edge_size)
        println("Done!")

        α = α_ratio / a
        β = 1.0/(kB*T)

        println("Initialising...")
        energies = initialise_energies_fdev(pos, lattice_dims, a, eL, eR, E0, dos_type, dos_param; ghost_inds=ighost)
        P0 = initialise_p_fdev(lattice_dims, energies, T)
        println("Done!")
        println("∑ P0 = $(sum(P0))")

        # println("Getting rates...")
        # rates = hop_rates_fdev(energies, pos, T, edge_size, α)
        # println("Done!")

        println("Solving master equation...")
        solve_out =  solve_otf(P0, energies, pos, ineighbours, β, α; full_device=true, ighost=ighost, lattice_dims=lattice_dims,
                            save_each=save_each, restart_threshold=restart_threshold,maxiter=maxiter)
        
        if save_each > 0
            converged, Pfinal, conv, Pt = solve_out
        else
            converged, Pfinal, conv = solve_out
        end

        println("Done!")
        println("∑ Pfinal = $(sum(Pfinal))")
        println("Computing carrier velocity...")
        J = current_density_otf(Pfinal,energies,pos,ineighbours,β,α,lattice_dims,a)
        println("J = $J")
        println("Done!")

        if save_each > 0
            return energies, pos, J, P0, Pfinal, conv, Pt
        else
            return energies, pos, J, P0, Pfinal, conv
        end
        
    end 
end