module RunYSSMBHoppingModelTest

    include("../../YSSMBSolver.jl")
    include("../../Utils.jl")

    using .YSSMBSolver, .Utils, PyCall, Random, LinearAlgebra

    function YSSMB_lattice_model_singleT(T, nocc; νeff= 0.3, N1=64, N2=32, K=0.0034, E0=1, dim=3, pbc="none",
        save_each=-1, restart_threshold=5)
        a = 10 # lattice constant in Å

        if dim == 3
            pos = zeros(N1,N2,N2,3)
            println("Defining lattice positions...")
            for i=1:N1
                for j=1:N2
                    for k=1:N2
                        pos[i,j,k,:] = [i-1,j-1,k-1] * a # start at origin
                    end
                end
            end
            pos = reshape(pos,N1*N2*N2,3)
        elseif dim == 2
            pos = zeros(N1,N2,2)
            println("Defining lattice positions...")
            for i=1:N1
                for j=1:N2
                    pos[i,j,:] = [i-1,j-1] * a # start at origin
                end
            end
            pos = reshape(pos, N1*N2,2)
        elseif dim == 1
            pos = collect(0:N1-1)
        end

        println("Done!")
        dX = a * (N1-1)
        println("Getting NNN inds...") 
        nnn_inds = get_nnn_inds(pos,a;pbc=pbc) #for each site, list of inds nearest neighbours and next-nearest neighbours
        println("Done!")
        if dim == 3
            Ω = (N1-1) * (N2-1) * (N2-1) * (a^3) #lattice volume in Å        
        elseif dim == 2
            Ω = (N1-1)*(N2-1)*(a^2)
        end
        N = size(pos,1)
        # nocc = Int(floor(density * Ω))
    
        if dim == 3
            E = [E0,0,0] /dX
        elseif dim == 2
            E = [E0,0] /dX
        else
            E = E0 / dX
        end


        # println("************** $T **************")
        converged = false
        P0 = zeros(N)
        Pfinal = zeros(N)
        Pt = zeros(1000000÷save_each, N)
        conv = 0
        rates = 0
        energies = 0 


        while !converged
            # P0 = initialise_FD(N,nocc)
            energies = randn(N) * νeff .- [dot(E, r) for r in eachrow(pos)]
            P0 = initialise_FD(energies, T)
            println("∑ P0 = $(sum(P0))")
            println("Computing rate matrix...")
            rates = miller_abrahams_YSSMB(pos,energies,nnn_inds, T)
            println("Done!")
            println("Iteratively solving master equation...")
            solve_out = solve(P0, rates; save_each=save_each, restart_threshold=restart_threshold)
            if save_each > 0
                converged, Pfinal, conv, Pt = solve_out
            else
                converged, Pfinal, conv = solve_out
            end
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

        nocc = 50
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
