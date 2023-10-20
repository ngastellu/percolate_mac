module HoppingMasterEquation

    using LinearAlgebra

    export run_HME

    const kB = 8.617333262e-5 # Boltzmann constant in eV/K
    const e = 1.0 # positron charge

    function fermi_dirac(energy,T)
        β = 1.0/(kB*T)
        return exp(β*energy)
    end

    function initialise_P(energies, pos, T, E)
        N = size(energies, 1)
        Pinit = zeros(N)
        for i=1:N
            # println(energies[i] - e*dot(E, pos[i,:]))
            Pinit[i] = fermi_dirac(energies[i] - e*dot(E, pos[i,:]), T) # !!! Assuming NEGATIVE charge !!!
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
    
    function iterate_implicit(Pold, rates)
        norms = sum(rates,dims=2)
        N = size(rates,1)
        Pnew = zeros(N)
        for i=1:N
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

    function solve(Pinit, rates; maxiter=1000000, ϵ=1e-6)
        N = size(Pinit,1)
        # println(N)
        cntr = 1
        Pnew = Pinit
        converged = false
        conv = zeros(maxiter,2)
        while cntr ≤ maxiter && !converged
            println(cntr)
            Pold = Pnew
            Pnew = iterate_implicit(Pold,rates)
            ΔP = abs.(Pnew-Pold)
            converged = all(ΔP .< ϵ)
            conv[cntr, 1] = maximum(ΔP)
            conv[cntr,2] = sum(ΔP)/N
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

    function run_HME(energies, pos, temps, E; maxiter=1000000, ϵ=1e-8)
        nb_temps = size(temps,1)
        N = size(energies, 1)
        d = size(pos, 2)
        Pfinal = zeros(N)
        conv = zeros(maxiter)
        velocities = zeros(nb_temps, d)
        for k=1:nb_temps
            T = temps[k]
            K = miller_abrahams_asymm(energies, pos, T, E)
            # println(K)
            P0 = initialise_P(energies, pos, T, E)
            # println(P0)
            Pfinal, conv = solve(P0,K; maxiter=maxiter, ϵ=ϵ)
            velocities[k,:] = carrier_velocity(K, Pfinal, pos)
        end
        return velocities, Pfinal, conv
    end
end