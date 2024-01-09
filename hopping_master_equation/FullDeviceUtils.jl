module FullDeviceUtils

    include("./Utils.jl")

    using .Utils, Random

    export initialise_energies_fdev, initialise_p_fdev, get_neighbour_lists_fdev

    function swapcols!(arr::AbstractMatrix,i::Integer,j::Integer)
        # Swaps columns i and j of a matrix in-place.
        N = size(arr,1)
        for k=1:N
            arr[k,i], arr[k,j] = arr[k,j], arr[k,i]
        end
    end

    function initialise_p_fdev(lattice_dims, energies, T)
        N = size(energies,1)
        d = size(lattice_dims,1)
        if d == 2
            Nx, Ny = lattice_dims
            edge_size = Ny
        else
            Nx, Ny, Nz = lattice_dims
            edge_size = Ny * Nz
        end

        Pinit = zeros(N)
        for i=1:N
            if i ≤ edge_size || i ≥ N-edge_size + 1
                Pinit[i] = 0 # perpetually empty electrode site
            elseif edge_size < i ≤ 2*edge_size || N-(2*edge_size) + 1 ≤ i ≤ N-edge_size
                Pinit[i] = 1 # perpetually occupied electrode site
            else
                Pinit[i] = fermi_dirac(energies[i],T)
            end
        end

        return Pinit

    end

    function initialise_energies_fdev(pos, lattice_dims, a, eL, eR, E0, dos_type, dos_param;ghost_inds=0)
        # Sets site energies (already accounting for static electric field). 
        # * eL, eR: Work functions of the left and right electrodes (i.e. energies of the electrode sites)
        # * E0: electric field strength (E = [E0, 0, 0] is assumed)
        # * dos_type: "uniform" or "gaussian"
        # * dos_param: width of the DOS if uniform DOS, stdev if Gaussian DOS (μ = 0 assumed)

        d = size(lattice_dims,1)
        N = prod(lattice_dims)

        @assert dos_type ∈ ("uniform", "gaussian") "dos_type must \"uniform\" or \"gaussian\" (currently set to $dos_type)"
        if d == 3
            @assert ghost_inds != 0 "Ghost indices must be specified to enforce PBC in 3D calculations."
        end

        if d == 2
            Nx, Ny = lattice_dims
            edge_size = Ny
        else
            Nx, Ny, Nz = lattice_dims
            edge_size = Ny * Nz
        end

        energies = zeros(N)

        for i=1:N
            if i ≤ 2*edge_size # left electrode sites
                energies[i] = eL
            elseif 2*edge_size < i ≤ N-(2*edge_size) # organic sites
                @assert 0 ≤ pos[i,1] ≤ (Nx-2) * a "xcoord of organic site should be ∈ [0,$((Nx-2)*a)]! (x = $(pos[i,1]))"
                 
                if dos_type == "gaussian"
                    energies[i] = randn() * dos_param - E0 * pos[i,1]
                else
                    energies[i] = rand()*dos_param - dos_param - E0 * pos[i,1]
                end
            else
                energies[i] = eR
            end
        end

        # Handle PBC cases
        if d == 2
            for i=3:(Nx-2)
                ii = i * Ny
                energies[ii] = energies[ii-Ny + 1] # this works bc sites are stored in ascending order of x-coordinate
            end
        else # 3D cases, sites are sorted the same as 2D but the extra dimension complicates things; resorting to hacky soln
            for ij in eachrow(ghost_inds)
                i, j = ij
                energies[i] = energies[j] # ghost_inds keeps track of which sites are periodic images of each other
            end
        end 

        return energies

    end

    function hop_rates_fdev(energies,pos,T,edge_size,α)
        # We assume an asymmetrical Miller-Abrahams hopping model.
        N = size(energies,1)
        β = 1.0/(kB*T)
        K = zeros(N,N)

        # Hopping between left electrode and adjacent organic sites
        for i=1:2*edge_size
            for j=2*edge_sites+1:3*edge_size
                K[i,j] = MA_asymm_hop_rate(energies[i], energies[j], pos[i], pos[j], β, α)
                K[j,i] = MA_asymm_hop_rate(energies[j], energies[i], pos[j], pos[i], β, α)
            end
        end

        # Hopping between right electrode and adjacent organic sites
        for i=N-(2*edge_size)+1:N
            for j=N-3*edge_size+1:N-2*edge_size
                K[i,j] = MA_asymm_hop_rate(energies[i], energies[j], pos[i], pos[j], β, α)
                K[j,i] = MA_asymm_hop_rate(energies[j], energies[i], pos[j], pos[i], β, α)
            end
        end

        # Hopping between organic sites
        for i=2*edge_size+1:N-(2*edge_size)
            for j=i+1:N-(2*edge_size)
                K[i,j] = MA_asymm_hop_rate(energies[i], energies[j], pos[i], pos[j], β, α)
                K[j,i] = MA_asymm_hop_rate(energies[j], energies[i], pos[j], pos[i], β, α)
            end
        end
        return K
    end

    function get_neighbours(r,pos_array,rcut;iremove=0)
        ΔR = pos_array .- r'
        ΔR = vec(sum(abs2,ΔR;dims=2))
        if iremove > 0 #if r is contained in pos_array, remove it from list of neighbours
            ΔR[iremove] += rcut*2
        end
        ii = findall(ΔR .≤ rcut)
        return ii
    end

    
    function build_neighbour_lists_fdev(pos,rcut,edge_size;max_nn_estimate=50) 
        # Creates ineighbours, a N * nneighbours matrix, where ineighbours[i,:] = indices of site i's 
        # neighbours. If site i has m < nneighbours neighbours, ineighbours[i,:m+1:nneighbours] = 0.
        # Since this is for full device calculations, electrode sites are treated a bit differently;
        # left-electrode sites are added to neighbour list of x=0 organic sites and right-electrode
        # sites are added to the neighbour lists of the x=(Nx-2)*a organic sites.

        N = size(pos,1)
        Norg = N - 4*edge_size # nb of organic sites
        ineighbours = zeros(Int,N,max_nn_estimate)
        max_nn = 0
        pos_org = view(pos, 1+2*edge_size:N-2*edge_size,:) #positions of organic sites only

        for i=1:N
            r = pos[i,:]
            if i ≤ edge_size || i > N - 2*edge_size #electrode site
                ii = get_neighbours(r,pos_org,rcut) .+ 2*edge_size
            else #organic site
                iorg = i - 2*edge_size
                ii = get_neighbours(r,pos_org[1+iorg:end,:],rcut;iremove=iorg) .+ 2*edge_size 
            end

            nb_neighs = sum(!iszero, ineighbours[i,:])
            nb_neighbs_new = size(ii,1) + nb_neighbs
            if nb_neighbs_new > max_nn
                max_nn = nb_neighbs_new
            end
            ineighbours[i,1+nb_neighs:nb_neighbs_new] = ii
            
            for j in ii # Add i to neighbour list of its neighbours
                nb_neighbs_j = sum(!iszero,ineighbours[j,:])
                ineighbours[j,nb_neighbs_j+1] = i
            end
        end

        # Organic sites
        for i=1+2*edge_size:N-(2*edge_size)
            i_fullpos = i+2*edge_size #index of site in full positions array
            ΔR = pos_org .- pos_org[i,:]' 
            ΔR = vec(sqrt.(sum(abs2,ΔR;dims=2)))
            ΔR[i] = 1000 #avoid counting self as neighbour
            ii = findall(ΔR .≤ rcut) .+ 2*edge_size
            if i ≤ edge_size
                ΔR_L = posL .- pos_org[i,:]'
                ΔR_L = vec(sqrt.(sum(abs2,ΔR_L;dims=2)))
                iiL = findall(ΔR_L .≤ rcut)
                ii = cat(iiL,  ii, dims=1)
            elseif i > Norg - edge_size
                ΔR_R = posR .- pos_org[i,:]'
                ΔR_R = vec(sqrt.(sum(abs2,ΔR_R;dims=2)))
                iiR = findall(ΔR_R .≤ rcut) .+ (N - 2*edge_size)
                ii = cat(ii, iiR, dims=1)
            end
            # println(ii)
            nb_neighbs = size(ii,1)
            @assert nb_neighbs ≤ max_nn_estimate "Atom $i_fullpos has $nb_neighbs neighbours! Expected at most max_nn = $(max_nn_estimate)."
            if nb_neighbs > max_nn
                max_nn = nb_neighbs
            end
            ineighbours[i,1:nb_neighbs] = ii
        end
        return ineighbours[:,1:max_nn] # get rid of useless zero entries
    end

end