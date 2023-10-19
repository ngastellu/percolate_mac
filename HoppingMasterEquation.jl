module HoppingMasterEquation

    using LinearAlgebra


    const kB = 8.617333262e-5 # Boltzmann constant in eV/K
    const e = 1.0 # positron charge

    function fermi_dirac(energy,T)
        β = 1.0/(kB*T)
        return exp(β*energy)
    end

    function initialise_P(energies, pos, T, E)
        return fermi_dirac(energies - e*np.dot(E,pos), T) # !!! Assuming NEGATIVE charge !!!
    end
    




end 