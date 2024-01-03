module YSSMBSolver

    export solve

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