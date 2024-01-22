module YSSMBSolver

    include("./Utils.jl")

    using .Utils

    export solve, solve_otf

    # function iterate_implicit(Pold, rates, L_inds, R_inds)
    function iterate_implicit(Pold, rates; full_device=false, electrode_inds=0, pbc_on=false, ghost_inds=0)
        norms = sum(rates,dims=2)
        N = size(rates,1)
        Pnew = zeros(N)

        if full_device
            @assert electrode_inds != 0 "Electrode indices must be specified for full-device simulations!"
        end

        # if pbc_on
        #     @assert ghost_inds != 0 "Indices of ghost sites must be specified for PBC to be enforced!"
        # end

        for i=1:N
            if full_device
                if i ∈ electrode_inds #impose fixed occupations at the boundaries of the system
                # if i ∈ L_inds #impose fixed occupations at the boundaries of the system
                    Pnew[i] = Pold[i]
                    continue
                end
            end
            if pbc_on
                if i ∈ ghost_inds[:,1] # do not update ghost sites during implicit iteration  
                    continue
                end
            end
            sum_top = 0
            sum_bot = 0
            for j=1:i
                sum_top += Pnew[j]*rates[j,i]
                sum_bot += (rates[i,j] - rates[j,i])*Pnew[j]
            end
            for j=i+1:N
                sum_top += Pold[j]*rates[j,i]
                sum_bot += (rates[i,j] - rates[j,i])*Pold[j]
            end
            Pnew[i] = (sum_top/norms[i]) / (1 - sum_bot/norms[i])
        end
        if ghost_inds != 0
            for ij ∈ ghost_inds
                i,j = ij
                Pnew[i] = Pnew[j] # set probabilities at ghost sites equal to that of their periodic images
            end
        end
        return Pnew
    end


    function update_sums(i,j,energies,pos,Pold,Pnew,sum_top,sum_bot,sum_rates,β,α)
        @assert i != j "No site is its own neighbour! Error for i = $i."
        ei = energies[i]
        ej = energies[j]
        ri = pos[i,:]
        rj = pos[j,:]
        if j < i
            Pj = Pnew[j]
        else
            Pj = Pold[j]
        end
        ω_ji =  MA_asymm_hop_rate(ej,ei,rj,ri,β,α)
        ω_ij = MA_asymm_hop_rate(ei,ej,ri,rj,β,α) 
        sum_top += Pj * ω_ji
        sum_bot += Pj * (ω_ij - ω_ji)
        sum_rates += ω_ji
        return sum_top, sum_bot, sum_rates
    end


    function iterate_implicit_otf(Pold, energies, pos, ineigh, β, α; full_device=false, ighost=0,lattice_dims=0)
        # Implicit iteration solver where the hopping rates are computed on-the-fly ('otf').
        # Partial or full PBC are assumed here.

        N = size(Pold, 1)
        Pnew = zeros(N)

        if full_device
            @assert lattice_dims != 0 "lattice_dimensions must be specified for full-device simulations!"
            d = size(lattice_dims,1)
            if d == 2
                Nx, Ny = lattice_dims
                edge_size = Ny
            else
                Nx, Ny, Nz = lattice_dims
                edge_size = Ny * Nz
            end
        end

        for i=1:N
            ghost_inds = 0
            if full_device
                # if i ∈ electrode_inds #impose fixed occupations at the boundaries of the system
                if i ≤ 2*edge_size || i > N - 2*edge_size #impose fixed occupations at the boundaries of the system
                    Pnew[i] = Pold[i]
                    continue
                end
            end

            sum_top = 0
            sum_bot = 0
            sum_rates = 0

            # Apply PBC
            if d == 2
                if full_device
                    if i % Ny == 0 #if site is a ghost site
                        continue
                    elseif i % Ny == 1 #if site is the periodic image of a ghost site
                        ighost = i + Ny - 1
                        # println("Site $i = $(pos[i,:]) ---> Ghost site $ighost = $(pos[ighost,:])")
                        for j ∈ ineigh[ighost,:]
                            if j == 0
                                break
                            end
                            sum_top, sum_bot, sum_rates = update_sums(i,j,energies,pos,Pold,Pnew,sum_top,sum_bot,sum_rates,β,α)
                        end
                        ghost_inds = [ighost]
                    end
                else
                    if i > N - Ny || i % Ny == 0 #if site is a ghost site
                        continue
                    elseif i ≤ Ny #site is on the x = 0 edge of lattice
                        if i == 1
                            ghost_inds = [N - Ny + 1, N]
                        else
                            ghost_inds = [i + N - Ny]
                        end
                        
                    elseif i % Ny == 1 #site is on the y = 0 edge of lattice
                        ghost_inds = [i + Ny - 1]
                    end
                    for g ∈ ghost_inds
                        for j ∈ ineigh[g,:]
                            if j == 0
                                break
                            end
                            sum_top, sum_bot, sum_rates = update_sums(i,j,energies,pos,Pold,Pnew,sum_top,sum_bot,sum_rates,β,α)
                        end
                    end
                end

            else # d = 3
                check = ighost .- i
                if any(iszero, check[:,1]) #if site is a ghost site, ignore it
                    # println("\n*** Ghost site! i = $i ---> $(pos[i,:]) ***")
                    continue
                end
                image_check = findall(iszero, check[:,2]) #check if site is the image site to any ghost sites
                if size(image_check,1) > 0
                    # println("\n--- Image site! i = $i ---> $(pos[i,:]) ---")
                    ghost_inds = [ighost[k,1] for k ∈ image_check]
                    for g ∈ ghost_inds
                        # println("\tghost site $g ---> $(pos[g,:])")
                        for j ∈ ineigh[g,:]
                            if j == 0
                                break
                            end
                            sum_top, sum_bot, sum_rates = update_sums(i,j,energies,pos,Pold,Pnew,sum_top,sum_bot,sum_rates,β,α)
                        end
                    end
                end
            end

            for j ∈ ineigh[i,:]
                if j == 0
                    break
                end
                sum_top, sum_bot, sum_rates = update_sums(i,j,energies,pos,Pold,Pnew,sum_top,sum_bot,sum_rates,β,α)
            end

            numerator = sum_top / sum_rates
            denominator = 1 - (sum_bot / sum_rates)
            Pnew[i] = numerator / denominator
            
            # Apply PBC; update P of ghost sites corresponding to i, if any
            if ghost_inds != 0
                for k ∈ ghost_inds
                    Pnew[k] = Pnew[i]
                    Pold[k] = Pnew[i] # if k > i, this will allow the update of other probabilities of sites j < k to use the updated P[k]
                end
            end
        end
        return Pnew
    end

    function solve_otf(Pinit, energies, pos, ineigh, β, α; full_device=false, ighost=0, lattice_dims=0, 
        maxiter=1000000, ϵ=1e-6, restart_threshold=-1, save_each=-1)
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
        end
        while cntr ≤ maxiter && !converged
            Pold = Pnew
            Pnew = iterate_implicit_otf(Pold,energies,pos,ineigh,β,α;full_device=full_device, ighost=ighost, lattice_dims=lattice_dims)
            if (save_each > 0) && (cntr % save_each == 0)
                Ps[cntr ÷ save_each, :] = Pnew
            end
            ΔP = abs.(Pnew-Pold)
            converged = all(ΔP .< ϵ)
            conv[cntr, 1] = maximum(ΔP)
            conv[cntr,2] = sum(ΔP)/N
            conv[cntr,3] = argmax(ΔP)
            println("$cntr  [max(ΔP), ⟨ΔP⟩, argmax(ΔP)] = $(conv[cntr,:])")
            # println("\n")
            if cntr > 1
                if conv[cntr,1] > conv[cntr-1,1]
                    restart_cntr += 1
                else
                    restart_cntr = 0 # reset to zero if max(ΔP) decreases on consecutive iterations
                end
            end
            if restart_threshold > 0 && restart_cntr ≥ restart_threshold
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
        end
        while cntr ≤ maxiter && !converged
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
            # println("[max(ΔP), ⟨ΔP⟩, argmax(ΔP)] = $(conv[cntr,:])")
            # println("\n")
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
    
end