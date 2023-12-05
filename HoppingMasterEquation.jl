module HoppingMasterEquation

    using LinearAlgebra, Random, StatsBase, FFTW

    export run_HME, lattice_hopping_model, YSSMB_lattice_model, generate_correlated_esites, 
    YSSMB_lattice_model_varE, generate_correlated_esites_1d, YSSMB_lattice_model_varE_1d,
    YSSMB_lattice_model_varT_1d, miller_abrahams_YSSMB, get_nnn_inds, initialise_random,
    solve, carrier_velocity

    const kB = 8.617333262e-5 # Boltzmann constant in eV/K
    const e = 1.0 # positron charge

    function fermi_dirac(energy,T)
        β = 1.0/(kB*T)
        return 1.0/( 1 + exp(β*energy))
    end

    function initialise_P(energies, pos, L_inds, R_inds, T, E, μL)
        N = size(energies, 1)
        d = size(pos,2) # spatial dimension
        Pinit = zeros(N)
        nL = size(L_inds,1)
        # nR = size(R_inds,1)
        # mincoord_E = Inf
        # maxcoord_E = -Inf
        # minpos = zeros(d)
        # maxpos = zeros(d)
        # for i=1:nL
        #     Ecoord = dot(E,pos[L_inds[i],:])
        #     if Ecoord < mincoord_E
        #         mincoord_E = Ecoord
        #         minpos = pos[L_inds[i],:]
        #     end
        # end
        # for i=1:nR
        #     Ecoord = dot(E,pos[R_inds[i],:])
        #     if Ecoord > maxcoord_E
        #         maxcoord_E = Ecoord
        #         maxpos = pos[R_inds[i],:]
        #     end
        # end
        # μR = μL - e*dot(E,maxpos-minpos) # Make sure lead voltages are consistent with imposed elecrtic field

        for i=1:N
            # if i ∉ (L_inds ∪ R_inds)
            #     Pinit[i] = fermi_dirac(energies[i] - e*dot(E, pos[i,:]), T) # !!! Assuming NEGATIVE charge !!!
            # elseif i ∈ L_inds
            if i ∈ L_inds
                Pinit[i] = fermi_dirac(energies[i]-μL,T)
            else
                Pinit[i] = fermi_dirac(energies[i] - e*dot(E, pos[i,:]), T) # !!! Assuming NEGATIVE charge !!!
            end
            # else # ⟺ i ∈ R_inds
            #     Pinit[i] = fermi_dirac(energies[i]-μR,T)
            # end
            # Pinit[i] = fermi_dirac(energies[i] - e*dot(E, pos[i,:]), T) # !!! Assuming NEGATIVE charge !!!
        end
        return Pinit
    end

    function miller_abrahams_asymm(energies, pos, T, E; α=1.0/30.0)
        N = size(energies,1)
        β = 1.0/(kB*T)
        K = zeros(N,N)

        for i=1:N
            energies[i] -= e * dot(E,pos[i,:])
        end

        for i=1:N
            for j=1:N
                ΔE = energies[i] - energies[j]
                ΔR = norm(pos[i,:] - pos[j,:])
                if ΔE < 0
                    K[i,j] = exp(-2*α*ΔR)
                else
                    K[i,j] = exp(-2*α*ΔR - β*ΔE)
                end

                if i == j
                    K[i,j] = 0
                end
            end
        end
        return K
    end

    function miller_abrahams_YSSMB(pos, energies, innn, T, E; a=10.0, ω0=1.0, _1d=false)
        N = size(pos,1)
        Γ = 5
        β = 1.0/(kB*T)
        println("β = $β")

        K = zeros(N,N)

        for i=1:N
            for j ∈ innn[i,:]
                if j == 0
                    break
                end
                if _1d
                    ΔR = abs(pos[j] - pos[i])
                    K[i,j] = ω0 * exp(-2Γ*ΔR/a) * exp(β*(energies[i] - energies[j] - e*E*ΔR)/2) 
                else
                    ΔR = pos[j,:] - pos[i,:]
                    K[i,j] = ω0 * exp(-2Γ*norm(ΔR)/a) * exp(β*(energies[i] - energies[j] - e*dot(E,ΔR))/2) 
                end
            end
        end
        return K
    end
    
    # function iterate_implicit(Pold, rates, L_inds, R_inds)
    function iterate_implicit(Pold, rates)
        norms = sum(rates,dims=2)
        N = size(rates,1)
        Pnew = zeros(N)
        for i=1:N
            # if i ∈ (L_inds ∪ R_inds) #impose fixed occupations at the boundaries of the system
            # if i ∈ L_inds #impose fixed occupations at the boundaries of the system
                # Pnew[i] = Pold[i]
                # continue
            # end
            sum_top = 0
            sum_bot = 0
            for j=1:i
                sum_top += Pnew[j]*rates[j,i]
                sum_bot += (rates[j,i] - rates[i,j])*Pnew[j]
            end
            for j=i+1:N
                sum_top += Pold[j]*rates[j,i]
                sum_bot += (rates[i,j] - rates[j,i])*Pold[j]
            end
            Pnew[i] = (sum_top/norms[i]) / (1 - sum_bot/norms[i])
        end
        return Pnew
    end

    # function solve(Pinit, rates, L_inds, R_inds; maxiter=1000000, ϵ=1e-6)
    function solve(Pinit, rates; maxiter=1000000, ϵ=1e-6, restart_threshold=5, save_each=-1)
        # println("Solving master equation...")
        N = size(Pinit,1)
        cntr = 1
        restart_cntr = 0
        Pnew = Pinit
        converged = false
        conv = zeros(maxiter,3)
        if save_each > 0
            Ps = zeros(maxiter ÷ save_each,N)
            Ps[1,:] = Pinit
            # println("******")
            # println(sum(Ps[1,:]))
            # println("******")
        end
        while cntr ≤ maxiter && !converged
            # if cntr > 2
            #     break
            # end
            println(cntr)
            Pold = Pnew
            # Pnew = iterate_implicit(Pold,rates, L_inds, R_inds)
            Pnew = iterate_implicit(Pold,rates)
            if (save_each > 0) && (cntr % save_each == 0)
                Ps[cntr ÷ save_each, :] = Pnew
            end
            ΔP = abs.(Pnew-Pold)
            converged = all(ΔP .< ϵ)
            conv[cntr, 1] = maximum(ΔP)
            conv[cntr,2] = sum(ΔP)/N
            conv[cntr,3] = argmax(ΔP)
            println("[max(ΔP), ⟨ΔP⟩, argmax(ΔP)] = $(conv[cntr,:])")
            println("\n")
            if cntr > 1
                if conv[cntr,1] > conv[cntr-1,1]
                    restart_cntr += 1
                else
                    restart_cntr = 0 # reset to zero if max(ΔP) decreases on consecutive iterations
                end
            end
            if restart_cntr ≥ restart_threshold
                return converged, ones(N) .* -1, conv 
            end
            cntr += 1
        end
        
        conv = conv[1:cntr-1,:]

        if save_each > 0
            Ps = Ps[1:cntr-1,:]
            return converged, Pnew, conv, Ps
        else
            return converged, Pnew, conv
        end
        
    end

    function carrier_velocity(rates, occs, pos)
        N = size(rates,1)
        d = size(pos,2)
        v = zeros(d)
        for i=1:N
            for j=1:N
                ΔR = pos[j,:] - pos[i,:] # !!! Assuming NEGATIVE charge carriers !!!
                v += rates[i,j]*occs[i]*(1-occs[j])*ΔR
            end
        end
        return v
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


    function initialise_random(N,nocc)
        P0 = zeros(N)
        occ_inds = sample(1:N, nocc; replace=false)
        P0[occ_inds] .= 1
        return P0
    end

    function get_nnn_inds(pos,a;pbc=false)
        # Given a cubic lattice whose positions are stored in array Nxd (where 1 ≤ d ≤ 3) `pos`, generate the list of
        # nearest and next-nearest neighbour indices for each lattice points.
        # Points in the bulk of the sample will have 4 (d=1), 8 (d=2), or 18 (d=3) nearest/next-nearest neighbours, 
        # sites on/nearest the edges will have less. For these edge/edge-adjacent sites, the "empty" entries will be
        # assigned the value 0.
        N = size(pos, 1)
        d = size(pos, 2)
        @assert d ∈ (1,2,3) "Lattice positions must be 1-,2-, or 3-dimensional. Here d = $(d)."
        if d == 3 || d == 2
            if d==3
                innn = zeros(Int,N,18)
            else # d==2
                innn = zeros(Int,N,8)
            end
            if pbc # to enforce PBC, must know max size of lattice in each direction
                # Assume positions start at a and end at (L-1)*a along each direction
                sorted_cols = [sort(col) for col in eachcol(pos)]
                L = ([col[end] - col[1] for col in sorted_cols])'
            end
            for i=1:N
                if pbc
                    if d==2
                        Δ = (pos .- pos[i,:]') .% L
                    else
                        Δ = (pos .- pos[i,:]')     # this handles the situation where
                        for r in eachrow(Δ)        # edge sites get spurious extra neighbours
                            for i=1:3              # when PBC are turned on in 3D
                                if abs(r[i]) == L[i]
                                    r[i] += a
                                end
                            end
                        end
                        Δ = Δ .% L
                    end
                else
                    Δ = pos .- pos[i,:]'
                end
                norms = sqrt.(sum(abs2,Δ;dims=2))
                ii = findall(0 .< vec(norms) .≤ sqrt(2)*a)
                nii = size(ii,1)
                if nii < size(innn,2)
                    println("Site $(pos[i,:]) has $nii NNN instead of $(size(innn,2)). The NNNs are:")
                    for k in ii
                        println(pos[k,:])
                    end
                end
                # println("size of ii = $(size(ii))")
                # if pbc
                    # println("Working on site $i:  $(pos[i,:])")
                    # for k in ii
                        # println("NNN $(k) = $(pos[k,:])")
                    # end
                # end
                for (j,n) in enumerate(ii)
                    innn[i,j] = n
                end
            end
        else #d ==1
            innn = zeros(Int,N,4)
            if pbc
                # Enforce PBC
                innn[1,:] = [N-1,N,2,3]
                innn[2,:] = [N,1,3,4]
                innn[N-1,:] = [N-3,N-2,N,1]
                innn[N,:] = [N-2,N-1,1,2]
            else
                innn[1,:] = [2,3,0,0]
                innn[2,:] = [1,3,4,0]
                innn[N-1,:] = [N-3,N-2,N,0]
                innn[N,:] = [N-2,N-1,0,0]
            end
    
            for i=3:N-2
                innn[i,:] = [-2, -1, 1, 2] .+ i 
            end 
        end
        return innn
    end

    function generate_correlated_esites(pos, a, N1, N2, Ω, T, K, ν)
        # reciprocal_lattice = (2π/(a^2)) .* pos
        # reciprocal_lattice = reciprocal_lattice ./ [N1 N2 N2]
        Φ = zeros(N1,N2,N2)
        β = 1.0/(kB*T)
        # for (k,q) in enumerate(eachrow(reciprocal_lattice))
        #     σ = 1 / (β*K*norm(q)^2) # neglect factor of Ω bc we end up dividing by Ω when we FT and we wanna avoid overflow
        #     Φ[k] = randn() * σ 
        # end
        for i=1:N1
            for j=1:N2
                for k=1:N2
                    q = norm([i,j,k] ./ [N1, N2, N2]) * (2π/a)
                    σ = 1 / (β*K*(q^2)) # neglect factor of Ω bc we end up dividing by Ω when we FT and we wanna avoid overflow
                    Φ[i,j,k] = randn() * σ
                end
            end
        end
        println("max(Φ) = $(maximum(Φ))")
        # ϕ = zeros(N)
        # for n=1:N
        #     r = pos[n,:]
        #     if r == [0,0,0]
        #         ϕ[n] = sum(Φ)
        #         println(ϕ[n])
        #     else
        #         println(r)
        #         dotprods = [dot(r,q) for q in eachrow(reciprocal_lattice)]
        #         phases = exp.(-im .* dotprods)
        #         ϕ[n] = sum(Φ .* phases)
        #     end
        # end 
        ϕ = fft(Φ)
        return ν .*  ϕ, Φ
    end


    function generate_correlated_esites_1d(a, N, T, K, ν)
            n = Int(N/2) + 1
            Φ = zeros(ComplexF64, n)
            β = 1.0/(kB*T)
            Q = vcat(collect(-n:-1),collect(1:n))
            q = abs.(collect(-n:-1)) .* (2π/n*a)
            for j=1:n
                σ = 1 / (β*K*(q[j]^2)) 
                f = randn() * σ
                φ = randn()
                Φ[j] = f * exp(im*φ)
                # Φ[N-j+1] = f * exp(-im*φ) # enforce Hermitian symmetry to get real energies after FT
            end
            ϕ = irfft(fftshift(Φ),N)
            return ν .*  ϕ, Φ
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