module FullDevice

    include("./Utils.jl")
    include("./FullDeviceUtils.jl")
    include("./YSSMBSolver.jl")

    using .Utils, .FullDeviceUtils, .YSSMBSolver, Random

    function run_full_device_lattice(lattice_dims,ΔV,T,dos_type, dos_param;a=10, eL=0, eR=0,
        α_ratio=10.0,  save_each=0, restart_threshold=-1)
        d = size(lattice_dims,1)
        @assert d ∈ (2,3) "Only 2D and 3D implemented. (d=$d dimensions specified)"
        # Lattice creation functions are defined such that pos is sorted by ascending x-coord
        if d == 2
            pos = create_2d_lattice(lattice_dims..., a)
        else
            pos = create_3d_lattice(lattice_dims..., a)
        end

        pos = pos .- [2*a,0,0] # set x coord of first row of organic sites to 0
        ΔX = (lattice_dims[1]-2) * a # last two rows of lattice are electrode sites
        E0 = ΔV / ΔX

        α = α_ratio / a

        println("Initialising...")
        energies = initialise_energies_fdev(pos, lattice_dims, eL, eR, E0, dos_type, dos_param)
        P0 = initialise_p_fdev(lattice_dims, energies, T)
        println("Done!")
        println("∑ P0 = $(sum(P0))")

        println("Getting rates...")
        rates = hop_rates_fdev(energies, pos, T, edge_size, α)
        println("Done!")

        println("Solving master equation...")
        solve_out =  solve(P0, rates; save_each=save_each, restart_threshold=restart_threshold)
        if save_each > 0
                converged, Pfinal, conv, Pt = solve_out
        else
            converged, Pfinal, conv = solve_out
        end
        println("Done!")
        println("∑ Pfinal = $(sum(Pfinal))")
        println("Computing carrier velocity...")
        vs = carrier_velocity(rates,Pfinal,pos)
        println("vs = $(norm(vs))")
        println("Done!")

        if save_each > 0
            return energies, vs, P0, Pfinal, rates, conv, Pt
        else
            return energies, vs, P0, Pfinal, rates, conv
        end
        
    end 

    # ******* MAIN *******

    for rnseed ∈ [0, 42, 64, 78]
        # rnseed = 64
            println("********** $rnseed **********")
            Random.seed!(rnseed) # seed RNG with task number so that we have the same disorder realisation at each T
    
    
            T = 300
    
            # nocc = 50
            ν = 0.007  # picked this stdev to roughly match the Gaussian DOS in the paper (DOI: 10.1103/PhysRevB.63.085202)
            d = 2
            n1 = 20
            n2 = 20
    
    
            energies, velocities, Pinits, Pfinals, rates, conv, Pt = YSSMB_lattice_model_singleT(T,nocc; N1=n1, N2=n2, E0 = 0.0, νeff = ν, save_each=1, dim=d,
            pbc="full", restart_threshold=10000)
    
            py"""import numpy as np
            nn= $rnseed
            dd = $d
            np.save(f'{dd}d/energies-{nn}.npy', $(PyObject(energies)))
            np.save(f'{dd}d/velocities-{nn}.npy', $(PyObject(velocities)))
            np.save(f'{dd}d/Pinits-{nn}.npy', $(PyObject(Pinits)))
            np.save(f'{dd}d/Pfinals-{nn}.npy', $(PyObject(Pfinals)))
            np.save(f'{dd}d/rates-{nn}.npy', $(PyObject(rates)))
            np.save(f'{dd}d/conv-{nn}.npy', $(PyObject(conv)))
            np.save(f'{dd}d/Pt-{nn}.npy', $(PyObject(Pt)))
            """   
        end

end