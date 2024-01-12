module Utils

    using LinearAlgebra, Random, StatsBase, FFTW

    export fermi_dirac, initialise_P, initialise_FD, initialise_random, miller_abrahams_asymm, miller_abrahams_YSSMB,
            carrier_velocity, get_nnn_inds, get_Efermi, create_2d_lattice, create_3d_lattice, MA_asymm_hop_rate,
            ghost_inds_3d, get_neighbours, build_neighbour_lists, current_density_otf

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

    function MA_asymm_hop_rate(ei,ej,ri,rj,β,α;L=Inf,ω0=1e12)
        ΔE = ej - ei # hopping from site i to site j
        ΔR = norm((ri - rj) .% L)
        
        if ΔE < 0
            ω = ω0 * exp(-2*α*ΔR)
        else
            ω = ω0 * exp(-2*α*ΔR - β*ΔE)
        end
        return ω
    end

    function miller_abrahams_asymm(energies, pos, T; α=1.0/30.0, L=Inf)
        N = size(energies,1)
        β = 1.0/(kB*T)
        K = zeros(N,N)

        for i=1:N
            for j=1:N
                ΔE = energies[i] - energies[j]
                ΔR = norm((pos[i,:] - pos[j,:]) .% L)
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

    function current_density_otf(occs, energies, pos, ineigh, β, α, lattice_dims, a;e=1.0)
        # Assumes the electric field is in the x-direction
        N = size(occs,1)
        d = size(pos,2)
        v = zeros(d)
        J = 0
        lattice_volume = prod(lattice_dims .- 1) * (a^d)
        for i=1:N
            ei = energies[i]
            ri = pos[i,:]
            for j ∈ ineigh[i,:]
                if j == 0
                    break
                end                
                ej = energies[j]
                rj = pos[j,:]
                Δx = rj[1] - ri[1] # !!! Assuming NEGATIVE charge carriers !!
                ω_ij = MA_asymm_hop_rate(ei,ej,ri,rj,β,α)
                J += ω_ij*occs[i]*(1-occs[j])*Δx
            end
        end
        return J * e / lattice_volume
    end

    function create_3d_lattice(Nx,Ny,Nz,a;full_device=false)
        pos = zeros(Nz,Ny,Nx,3)
        for i=1:Nz
            for j=1:Ny
                for k=1:Nx
                    pos[i,j,k,:] = [k-1,j-1,i-1] .* a # this will sort final array along its first axis
                end
            end
        end
        N = Nx*Ny*Nz
        pos = reshape(pos,N,3)
        if full_device
            pos = pos .- [2*a,0,0]' # set x coord of first row of organic sites to 0
            # set position of outermost rows of electrode sites equal to that of the innermost rows
            edge_size = Ny*Nz
            pos[1:edge_size,1] .+= a
            pos[N-edge_size+1:N,1] .-= a
        end
        return pos
    end

    function create_2d_lattice(Nx,Ny,a;full_device=true)
        pos = zeros(Ny,Nx,2)
        for i=1:Ny
            for j=1:Nx
                pos[i,j,:] = [j-1,i-1].* a # this will sort final array along its first axis
            end
        end
        N = Nx*Ny
        pos = reshape(pos,N,2)
        if full_device
            pos = pos .- [2*a,0]' # set x coord of first row of organic sites to 0
            # set position of outermost rows of electrode sites equal to that of the innermost rows
            edge_size = Ny
            pos[1:edge_size,1] .+= a
            pos[N-edge_size+1:N,1] .-= a
        end
        return pos
    end


    function initialise_random(N,nocc)
        P0 = zeros(N)
        occ_inds = sample(1:N, nocc; replace=false)
        P0[occ_inds] .= 1
        return P0
    end

    function isghost(r,Nx,Ny,Nz,a;full_device=false)
        if full_device
            return findall(iszero, r[2:3] .- ([Ny,Nz] .-1 ) .* a) .+ 1
        else
            return findall(iszero, r .- ([Nx,Ny,Nz].-1) .* a )
        end
    end

    function ghost_inds_3d(pos,Nx,Ny,Nz,a;full_device=false)
        # Determines indices of edge sites which are periodic images of each other, depending on
        # the type of PBC are enforced.
        # If full_device is true, only PBC along y and z are enforced.
        # Otherwise PBC along all three axes are enforced.
        # Assumes coords go from 0 to (N-1)*a along each axis.

        N = Nx * Ny * Nz

        if full_device
            edge_size = Ny * Nz
            L = ([Ny,Nz] .- 1) .* a
            Nx_org = Nx - 4
            org_pos = view(pos,2*edge_size+1:N-2*edge_size,:)
            nghosts = Nx_org * (Ny + Nz - 1)
            ghost_inds = zeros(Int, nghosts, 2)
            k = 1
            for (i,r) in enumerate(eachrow(org_pos))
                i += 2*edge_size
                zero_axes = findall(iszero, r[2:3]) .+ 1
                if size(zero_axes,1) > 0
                    ghost_axes = isghost(r,Nx,Ny,Nz,a;full_device=full_device)
                    if size(ghost_axes,1) > 0
                        # println("Ghost site! $i ---> $(pos[i,:])")
                        ighost = i
                        target = r[:] # using [:] copies r instead of operating on a view
                        for d in ghost_axes
                            target[d] = 0
                        end
                        # println("Target = $target")
                        Δ = [r2 - target for r2 ∈ eachrow(org_pos)]
                        hits = findall(iszero,Δ)
                        # println("Hits: ")
                        # for h in hits
                            # h+= 2*edge_size
                            # println("$h ---> $(pos[h,:])")
                        # end
                        # print('\n')
                        j = hits[1]
                        ireal = j + 2*edge_size # 'real' site, the one whose occ prob we actually solve for
                    else
                        # println("Image site! $i ---> $(pos[i,:])")
                        ireal = i  # 'real' site, the one whose occ prob we actually solve for
                        target = r[:] # using [:] copies r instead of operating on a view
                        for d ∈ zero_axes
                            target[d] = L[d-1]
                        end
                        # println("Target = $target")
                        Δ = [r2 - target for r2 ∈ eachrow(org_pos)]
                        hits = findall(iszero,Δ)
                        # println("Hits: ")
                        # for h in hits
                            # h+= 2*edge_size
                            # println("$h ---> $(pos[h,:])")
                        # end
                        # print('\n')
                        j = hits[1]
                        ighost = j + 2*edge_size # fictitious site, the one which mirrors the 'real' site
                    end
                    ghost_inds[k,1] = ighost
                    ghost_inds[k,2] = ireal
                    k += 1
                end
            end

        else
            nghosts = Nx*Ny + Nx*Nz + Ny*Nz - (Nx + Ny + Nz) + 1
            @assert Nx*Ny*Nz > nghosts "Lattice too small! Must have:
             Nx*Ny*Nz ($(Nx*Ny*Nz)) > Nx*Ny + Nx*Nz + Ny*Nz ($nghosts)"
            L = ([Nx,Ny,Nz] .- 1) .* a
            ghost_inds = zeros(Int,nghosts,2)
            k = 1
            for (i,r) in enumerate(eachrow(pos))
                zero_axes = findall(iszero, r)
                if size(zero_axes,1) > 0
                    ghost_axes = isghost(r,Nx,Ny,Nz,a;full_device=false)
                    if size(ghost_axes,1) > 0 # ignore sites with forms like [(Nx-1)*a, 0, z], [x, (Ny-1)*a, 0], etc.
                        ighost = i # fictitious site, the one which mirrors the 'real' site
                        target = r[:] # using [:] copies r instead of operating on a view
                        for d ∈ ghost_axes
                            target[d] = 0
                        end
                        Δ = [r2 - target for r2 ∈ eachrow(pos)]
                        j = findall(iszero,Δ)[1]
                        ireal = j # 'real' site, the one whose occ prob we actually solve for
                    else
                        ireal = i  # 'real' site, the one whose occ prob we actually solve for
                        target = r[:] # using [:] copies r instead of operating on a view
                        for d ∈ zero_axes
                            target[d] = L[d]
                        end
                        Δ = [r2 - target for r2 ∈ eachrow(pos)]
                        j = findall(iszero,Δ)[1]
                        ighost = j # fictitious site, the one which mirrors the 'real' site
                    end
                    ghost_inds[k,1] = ighost
                    ghost_inds[k,2] = ireal
                    k += 1
                end
            end
        end
        return ghost_inds
    end

    function get_neighbours(r,pos_array,rcut)
        ΔR = pos_array .- r'
        ΔR = vec(sqrt.(sum(abs2,ΔR;dims=2)))
        ii = findall(ΔR .≤ rcut)
        return ii
    end


    function build_neighbour_lists(pos,rcut; max_nn_estimate=50) # Might be worth using a kD-tree for this...
        # Creates ineighbours, a N * nneighbours matrix, where ineighbours[i,:] = indices of site i's 
        # neighbours. If site i has m < nneighbours neighbours, ineighbours[i,:m+1:nneighbours] = 0.
        N = size(pos,1)
        ineighbours = zeros(Int,N,max_nn_estimate)
        max_nn = 0
        for i=1:N
            ΔR = pos .- pos[i,:]' 
            ΔR = vec(sqrt.(sum(abs2,ΔR;dims=2)))
            ΔR[i] = 1000 #avoid counting self as neighbour
            ii = findall(ΔR .≤ rcut)
            println(ii)
            nb_neighbs = size(ii,1)
            @assert nb_neighbs ≤ max_nn_estimate "Atom $i has $nb_neighbs neighbours! Expected at most max_nn = $(max_nn_estimate)."
            if nb_neighbs > max_nn
                max_nn = nb_neighbs
            end
            ineighbours[i,1:nb_neighbs] = ii
        end
        return ineighbours[:,1:max_nn] # get rid of useless zero entries
    end

    function get_nnn_inds(pos,a;pbc="none")
        # Given a cubic lattice whose positions are stored in array Nxd (where 1 ≤ d ≤ 3) `pos`, generate the list of
        # nearest and next-nearest neighbour indices for each lattice points.
        
        # * If no PBC are enforced (i.e. L = Inf):
        # Points in the bulk of the sample will have 4 (d=1), 8 (d=2), or 18 (d=3) nearest/next-nearest neighbours, 
        # sites on/nearest the edges will have less. For these edge/edge-adjacent sites, the "empty" entries will be
        # assigned the value 0.
        
        # * If full PBC are enforced (i.e. L = [Lx, Ly, Lz])
        # There is no edge/bulk distinction; all sites are bulk sites.
        
        # * If partial PBC are enforced (i.e. L = [Inf, Ly, Lz])
        # PBC are only applied along the y and z directions for now (this assumes the E-field is in
        # the x direction).

        N = size(pos, 1)
        d = size(pos, 2)
        @assert d ∈ (1,2,3) "Lattice positions must be 1-,2-, or 3-dimensional. Here d = $(d)."
        @assert pbc ∈ ("none", "full", "partial") "PBC argument must one of three values: \"none\" (default), 
                                                \"full\", or \"partial\"."
        if d == 3 || d == 2
            if d==3
                innn = zeros(Int,N,18)
            else # d==2
                innn = zeros(Int,N,8)
            end
            if pbc != "none" # to enforce PBC, must know max size of lattice in each direction
                # Assume positions start at a and end at (L-1)*a along each direction
                sorted_cols = [unique(sort(col)) for col in eachcol(pos)]
                L = ([col[end-1] - col[1] for col in sorted_cols])'
                if pbc == "partial"
                    L[1] = Inf #this basically nullifies the PBC along x
                end
                println("-------- L = $L --------")
            end
            for i=1:N
                if pbc != "none"
                    if d==2
                        Δ = (pos .- pos[i,:]') .% L
                    else
                        Δ = (pos .- pos[i,:]')     # this handles the situation where
                        # for r in eachrow(Δ)        # edge sites get spurious extra neighbours
                        #     for i=1:3              # when PBC are turned on in 3D
                        #         if abs(r[i]) == L[i]
                        #             r[i] += a
                        #         end
                        #     end
                        # end
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
                if nii > size(innn,2)
                    println("Site $(pos[i,:]) has $nii NNN instead of $(size(innn,2)). The NNNs are:")
                    # for k in ii
                    #     println(pos[k,:])
                    # end
                end
                println("size of ii = $(size(ii))")
                if pbc != "none"
                    println("Working on site $i:  $(pos[i,:])")
                    for k in ii
                        println("NNN $(k) = $(pos[k,:]); ΔR = $(norms[k])")
                    end
                end
                for (j,n) in enumerate(ii)
                    innn[i,j] = n
                end
            end
        else #d ==1
            innn = zeros(Int,N,4)
            if pbc != "none"
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

end