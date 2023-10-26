module HoppingMasterEquation

    using LinearAlgebra, Random

    export run_HME, lattice_hopping_model

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
            # println(energies[i] - e*dot(E, pos[i,:]))
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
    
    function iterate_implicit(Pold, rates, L_inds, R_inds)
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
                sum_bot += (rates[j,i] - rates[i,j])*Pold[j]
            end
            Pnew[i] = (sum_top/norms[i]) / (1 - sum_bot/norms[i])
        end
        return Pnew
    end

    function solve(Pinit, rates, L_inds, R_inds; maxiter=1000000, ϵ=1e-6)
        N = size(Pinit,1)
        # println(N)
        cntr = 1
        Pnew = Pinit
        converged = false
        conv = zeros(maxiter,3)
        while cntr ≤ maxiter && !converged
            println(cntr)
            Pold = Pnew
            Pnew = iterate_implicit(Pold,rates, L_inds, R_inds)
            ΔP = abs.(Pnew-Pold)
            converged = all(ΔP .< ϵ)
            conv[cntr, 1] = maximum(ΔP)
            conv[cntr,2] = sum(ΔP)/N
            conv[cntr,3] = argmax(ΔP)
            println(conv[cntr,:])
            cntr += 1
        end
        return Pnew, conv
    end

    function carrier_velocity(rates, occs, pos)
        println(size(occs[1,:]))
        println(size(occs[2,:]))
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

    function run_HME(energies, pos, L_inds, R_inds, temps, E, μL; maxiter=1000000, ϵ=1e-8)
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
            # println(K)
            P0 = initialise_P(energies, pos, L_inds, R_inds, T, E, μL)
            Pinit[k,:] = P0
            # println(P0)
            Pfinal[k,:], conv = solve(P0, K, L_inds, R_inds; maxiter=maxiter, ϵ=ϵ)
            velocities[k,:] = carrier_velocity(K, Pfinal, pos)
        end
        return velocities, Pfinal, Pinit, conv
    end


    function lattice_hopping_model(pos, L_inds, R_inds, temps, E, μL, ntrials)

        N = size(pos,1)
        ntemps = size(temps,1)
        occs_init = zeros(ntemps, N)
        occs_final = zeros(ntemps, N)
        energies_saved = zeros(N)
        velocities_norm = zeros(ntrials, ntemps)

        for i=1:ntrials
            energies = randn(N)
            # energies = rand(N) * (2 * E0) .- E0
    
            velocities, occs, occs0, conv = run_HME(energies, pos, L_inds, R_inds, temps, E, μL;ϵ=1e-7)
    
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

end