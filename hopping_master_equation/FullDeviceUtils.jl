module FullDeviceUtils

    include("./Utils.jl")
    include("./YSSMBSolver.jl")

    using .Utils, .YSSMBSolver

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

    end

    function initialise_energies_fdev(pos, lattice_dims, eL, eR, E0, dos_type, dos_param)
        # Sets site energies (already accounting for static electric field). 
        # * eL, eR: Work functions of the left and right electrodes (i.e. energies of the electrode sites)
        # * E0: electric field strength (E = [E0, 0, 0] is assumed)
        # * dos_type: "uniform" or "gaussian"
        # * dos_param: width of the DOS if uniform DOS, stdev if Gaussian DOS (μ = 0 assumed)

        d = size(lattice_dims,1)
        N = prod(lattice_dims)

        @assert dos_type ∈ ("uniform", "gaussian") "dos_type must \"uniform\" or \"gaussian\" (currently set to $dos_type)"

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
                if dos_type == "gaussian"
                    energies[i] = randn() * dos_param - E0 * pos[i,1]
                else
                    energies[i] = rand()*dos_param - dos_param - E0 * pos[i,1]
                end
            else
                energies[i] = eR
            end
        end

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
            for j=2*edge_size+1:N-(2*edge_size)
                if i == j
                    K[i,j] = 0
                else
                    K[i,j] = MA_asymm_hop_rate(energies[i], energies[j], pos[i], pos[j], β, α)
            end
        end
        return K
    end


    function run_full_device(lattice_dims,E0,T;a=10)
        d = size(lattice_dims,1)
        @assert d ∈ (2,3) "Only 2D and 3D implemented. (d=$d dimensions specified)"
        if d == 2
            pos = create_2d_lattice(lattice_dims..., a)
        else
            pos = create_3d_lattice(lattice_dims..., a)
        end

        # sort pos by x value (by construction, pos is sorted along its last column)
        swapcols!(pos,1,d)
        pos = pos .- [3*a,0,0] # set x coord of first row of organic sites to 0
        
    end 

end