module OldDrivers

    include("./YSSMBSolver.jl")
    include("./Utils.jl")

    using .YSSMBSolver, .Utils

    # function lattice_hopping_model(pos, L_inds, R_inds, temps, E, μL, ntrials)
    function lattice_hopping_model(pos, temps, E, μL, ntrials)
        N = size(pos,1)
        ntemps = size(temps,1)
        occs_init = zeros(ntemps, N)
        occs_final = zeros(ntemps, N)
        energies_saved = zeros(N)
        velocities_norm = zeros(ntrials, ntemps)

        for i=1:ntrials
            energies = randn(N)
            # energies = rand(N) * (2 * E0) .- E0

            # velocities, occs, occs0, conv = run_HME(energies, pos, L_inds, R_inds, temps, E, μL;ϵ=1e-7)
            velocities, occs, occs0, conv = run_HME(energies, pos, temps, E, μL;ϵ=1e-7)

            for j=1:ntemps
                velocities_norm[i,j] = norm(velocities[j,:])
            end

            if i == ntrials #save only occupations of the last disorder realisation (averaging over different realisations doesn't really make sense)
                occs_init = occs0
                occs_final = occs
                energies_saved = energies
            end
        end

        velocities_avg = sum(velocities_norm,dims=1)/ntrials

        return velocities_avg, occs_init, occs_final, energies_saved

    end

    # function run_HME(energies, pos, L_inds, R_inds, temps, E, μL; maxiter=1000000, ϵ=1e-8)
    function run_HME(energies, pos, temps, E, μL; maxiter=1000000, ϵ=1e-8)
        nb_temps = size(temps,1)
        N = size(energies, 1)
        d = size(pos, 2)
        Pfinal = zeros(nb_temps,N)
        Pinit = zeros(nb_temps,N)
        conv = zeros(maxiter)
        velocities = zeros(nb_temps, d)
        for k=1:nb_temps
            T = temps[k]
            println("\n\n ************ T = $T *************")
            K = miller_abrahams_asymm(energies, pos, T, E)
            P0 = initialise_P(energies, pos, L_inds, R_inds, T, E, μL)
            Pinit[k,:] = P0
            # Pfinal[k,:], conv = solve(P0, K, L_inds, R_inds; maxiter=maxiter, ϵ=ϵ)
            Pfinal[k,:], conv = solve(P0, K; maxiter=maxiter, ϵ=ϵ)
            velocities[k,:] = carrier_velocity(K, Pfinal, pos)
        end
        println("Done solving master equation!")
        return velocities, Pfinal, Pinit, conv
    end

    function YSSMB_lattice_model(temps, density; νeff = 0.3, N1=64, N2=32, e_corr = true, K=0.0034)
        nb_temps = size(temps,1)
        a = 10 # lattice constant in Å

        pos = zeros(N1,N2,N2,3)
        println("Defining lattice positions...")
        for i=1:N1
            for j=1:N2
                for k=1:N2
                    pos[i,j,k,:] = [i-1,j-1,k-1] * a # start at origin
                end
            end
        end

        pos = reshape(pos, N1*N2*N2, 3)
        println("Done!")
        dX = a * (N1-1)
        println("Getting NNN inds...") 
        innn = get_innn_3d(pos,a) #for each site, list of inds nearest neighbours and next-nearest neighbours
        println("Done!")
        Ω = (N1-1) * (N2-1) * (N2-1) * (a^3) #lattice volume in Å        
        N = size(pos,1)
        nocc = Int(floor(density * Ω))
        if !e_corr
            energies = randn(N) * νeff
        end
        E = [1,0,0] /dX

        Pfinals = zeros(nb_temps,N)
        Pinits = zeros(nb_temps,N)
        velocities = zeros(nb_temps, 3)

        for (n,T) ∈ enumerate(temps)
            println("************** $T **************")
            P0 = initialise_random(N,nocc)
            println("∑ P0 = $(sum(P0))")
            Pinits[n,:] = P0
            if e_corr
                energies = generate_correlated_esites(pos,a,N1,N2,Ω,T,K,νeff)
            end
            println("Computing rate matrix...")
            rates = miller_abrahams_YSSMB(pos,energies,nnn_inds, T, E)
            println("Done!")
            println("Iteratively solving master equation...")
            Pfinal, conv = solve(P0, rates)
            println("Done!")
            println("∑ Pfinal = $(sum(Pfinal))")
            Pfinal ./= sum(Pfinal)
            Pfinals[n,:] = Pfinal
            println("Computing carrier velocity...")
            vs = carrier_velocity(rates,Pfinal,pos)
            println("Done!")
            velocities[n,:] = vs
        end

        return energies, velocities, Pinits, Pfinals
    end



    function YSSMB_lattice_model_varE(efields, density; νeff = 0.3, N1=64, N2=32, e_corr = true, K=0.0034, T=300)
        nb_efields = size(efields,1)
        a = 10 # lattice constant in Å

        pos = zeros(N1,N2,N2,3)
        println("Defining lattice positions...")
        for i=1:N1
            for j=1:N2
                for k=1:N2
                    pos[i,j,k,:] = [i-1,j-1,k-1] * a # start at origin
                end
            end
        end

        pos = reshape(pos, N1*N2*N2, 3)
        println("Done!")
        dX = a * (N1-1)
        println("Getting NNN inds...") 
        nnn_inds = get_nnn_inds_3d(pos,a) #for each site, list of inds nearest neighbours and next-nearest neighbours
        println("Done!")
        Ω = (N1-1) * (N2-1) * (N2-1) * (a^3) #lattice volume in Å        
        N = size(pos,1)
        nocc = Int(floor(density * Ω))
        if !e_corr
            energies = randn(N) * νeff
        end

        Pfinals = zeros(nb_efields,N)
        Pinits = zeros(nb_efields,N)
        velocities = zeros(nb_efields, 3)

        for (n,E) ∈ enumerate(efields)
            println("************** $E **************")
            P0 = initialise_random(N,nocc)
            println("∑ P0 = $(sum(P0))")
            Pinits[n,:] = P0
            if e_corr
                energies = generate_correlated_esites(pos,a,N1,N2,Ω,T,K,νeff)
            end
            println("Computing rate matrix...")
            Ef = [E,0,0] / dX
            rates = miller_abrahams_YSSMB(pos,energies,nnn_inds, T, Ef)
            println("Done!")
            println("Iteratively solving master equation...")
            Pfinal, conv = solve(P0, rates)
            println("Done!")
            println("∑ Pfinal = $(sum(Pfinal))")
            Pfinal ./= sum(Pfinal)
            Pfinals[n,:] = Pfinal
            println("Computing carrier velocity...")
            vs = carrier_velocity(rates,Pfinal,pos)
            println("Done!")
            velocities[n,:] = vs
        end

        return energies, velocities, Pinits, Pfinals
    end


    function YSSMB_lattice_model_varE_1d(efields, density; νeff = 0.3, N=1024, e_corr = true, K=0.0034, T=300)
        nb_efields = size(efields,1)
        a = 10 # lattice constant in Å

        pos = collect(1:N) * a

        dX = a * (N-1)
        println("Getting NNN inds...") 
        nnn_inds = zeros(Int,N,4)
        # Enforce PBC
        nnn_inds[1,:] = [N-1,N,2,3]
        nnn_inds[2,:] = [N,1,3,4]
        nnn_inds[N-1,:] = [N-3,N-2,N,1]
        nnn_inds[N,:] = [N-2,N-1,1,2]

        for i=3:N-2
        nnn_inds[i,:] = [-2, -1, 1, 2] .+ i 
        end

        println("Done!")
        Ω = dX
        N = size(pos,1)
        nocc = Int(floor(density * Ω))
        println("nocc = $nocc")
        if !e_corr
            energies = randn(N) * νeff
        end

        Pfinals = zeros(nb_efields,N)
        Pinits = zeros(nb_efields,N)
        velocities = zeros(nb_efields,1)

        for (n,E) ∈ enumerate(efields)
            println("************** $E **************")
            P0 = initialise_random(N,nocc)
            println("∑ P0 = $(sum(P0))")
            Pinits[n,:] = P0
            if e_corr
                energies, Φ = generate_correlated_esites_1d(a,N,T,K,νeff)
            end
            println("Computing rate matrix...")
            Ef = E / dX
            rates = miller_abrahams_YSSMB(pos,energies,nnn_inds, T, Ef; _1d=true)
            println("Done!")
            println("Iteratively solving master equation...")
            Pfinal, conv = solve(P0, rates)
            while sum(Pfinal) == -N #if `solve` didn't converge and got restarted
                P0 = initialise_random(N,nocc)
                energies, Φ = generate_correlated_esites_1d(a,N,T,K,νeff)
                rates = miller_abrahams_YSSMB(pos,energies,nnn_inds, T, Ef; _1d=true)
                Pfinal, conv = solve(P0, rates)
            end
            println("Done!")
            println("∑ Pfinal = $(sum(Pfinal))")
            Pfinal ./= sum(Pfinal)
            Pfinals[n,:] = Pfinal
            println("Computing carrier velocity...")
            vs = carrier_velocity(rates,Pfinal,pos)
            println("Done!")
            velocities[n,:] = vs
        end

        return energies, velocities, Pinits, Pfinals
    end

    function YSSMB_lattice_model_varT_1d(temperatures, density; νeff = 0.3, N=1024, a=10, e_corr = true, K=0.0034, E=9.78e-5)
        nb_Ts = size(temperatures,1)

        pos = collect(1:N) * a

        dX = a * (N-1)
        println("Getting NNN inds...") 
        nnn_inds = zeros(Int,N,4)
        # Enforce PBC
        nnn_inds[1,:] = [N-1,N,2,3]
        nnn_inds[2,:] = [N,1,3,4]
        nnn_inds[N-1,:] = [N-3,N-2,N,1]
        nnn_inds[N,:] = [N-2,N-1,1,2]

        for i=3:N-2
            nnn_inds[i,:] = [-2, -1, 1, 2] .+ i 
        end

        println("Done!")
        Ω = dX
        N = size(pos,1)
        nocc = Int(floor(density * Ω))
        println("nocc = $nocc")
        if !e_corr
            energies = randn(N) * νeff
        end

        Pfinals = zeros(nb_Ts,N)
        Pinits = zeros(nb_Ts,N)
        velocities = zeros(nb_Ts,1)

        for (n,T) ∈ enumerate(temperatures)
            println("************** $T **************")
            P0 = initialise_random(N,nocc)
            println("∑ P0 = $(sum(P0))")
            Pinits[n,:] = P0
            if e_corr
                energies, Φ = generate_correlated_esites_1d(a,N,T,K,νeff)
            end
            println("Computing rate matrix...")
            rates = miller_abrahams_YSSMB(pos,energies,nnn_inds, T, E; _1d=true)
            println("Done!")
            println("Iteratively solving master equation...")
            Pfinal, conv = solve(P0, rates)
            while sum(Pfinal) == -N #if `solve` didn't converge and got restarted
                P0 = initialise_random(N,nocc)
                energies, Φ = generate_correlated_esites_1d(a,N,T,K,νeff)
                rates = miller_abrahams_YSSMB(pos,energies,nnn_inds, T, E; _1d=true)
                Pfinal, conv = solve(P0, rates)
            end
            println("Done!")
            println("∑ Pfinal = $(sum(Pfinal))")
            Pfinal ./= sum(Pfinal)
            Pfinals[n,:] = Pfinal
            println("Computing carrier velocity...")
            vs = carrier_velocity(rates,Pfinal,pos)
            println("Done!")
            velocities[n,:] = vs
        end

        return energies, velocities, Pinits, Pfinals
    end

end