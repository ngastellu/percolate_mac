module Utils

    using LinearAlgebra, Random, StatsBase, FFTW

    export initialise_P, initialise_FD, initialise_random, miller_abrahams_asymm, miller_abrahams_YSSMB,
            carrier_velocity, get_nnn_inds, get_Efermi

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

    function get_Efermi(energies,nocc)
        N = size(energies,1)
        @assert 0 < nocc ≤ N "Number of charge carriers must be smaller than the number of sites."
        e_sorted = sort(energies)
        return 0.5 * (e_sorted[nocc] + e_sorted[nocc+1])
    end

    function initialise_FD(energies, eF, T)
        N = size(energies, 1)
        P = zeros(N)
        for i=1:N
            P[i] = fermi_dirac(energies[i]-eF, T)
        end
        return P
    end

    function miller_abrahams_asymm(energies, pos, T; α=1.0/30.0)
        N = size(energies,1)
        β = 1.0/(kB*T)
        K = zeros(N,N)

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

    function miller_abrahams_YSSMB(pos, energies, innn, T; a=10.0, ω0=1.0, _1d=false)
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
                    K[i,j] = ω0 * exp(-2Γ*ΔR/a) * exp(β*(energies[i] - energies[j])/2) 
                else
                    ΔR = pos[j,:] - pos[i,:]
                    K[i,j] = ω0 * exp(-2Γ*norm(ΔR)/a) * exp(β*(energies[i] - energies[j])/2) 
                end
            end
        end
        return K
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
                norms[i] = 1000 #avoid counting self as neighbour
                ii = findall(0 .≤ vec(norms) .≤ sqrt(2)*a)
                nii = size(ii,1)
                # ----- PRINT ERRORS IN NN COUNT -----
                if nii < size(innn,2)
                    println("Site $(pos[i,:]) has $nii NNN instead of $(size(innn,2)). The NNNs are:")
                    for k in ii
                        println(pos[k,:])
                    end
                end
                # println("size of ii = $(size(ii))")
                # if pbc
                #     println("Working on site $i:  $(pos[i,:])")
                #     for k in ii
                #         println("NNN $(k) = $(pos[k,:])")
                #     end
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

    function get_neighbour_inds(pos,rcut; pbc=false)
        N = size(pos,1)
        nb_neighbours = 50
        nnn_inds = zeros(N,nb_neighbours)
        if pbc # to enforce PBC, must know max size of lattice in each direction
            # Assume positions start at a and end at (L-1)*a along each direction
            sorted_cols = [sort(col) for col in eachcol(pos)]
            L = ([col[end] - col[1] for col in sorted_cols])'
        end
        for i=1:N
            Δ = pos .- pos[i,:]'
            if pbc
                Δ = Δ .% L
            end
            Δ = norm(Δ)
        end
    end

end