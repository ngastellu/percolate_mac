module TestHME
    include("./HoppingMasterEquation.jl")

    using .HoppingMasterEquation, Random, PyCall


    N = 50
    E0 = 1.0

    pos = zeros(N,N,2)

    for i=1:N
        for j=1:N
            pos[i,j,:] = [i,j]
        end
    end

    pos = reshape(pos, N*N, 2)

    println(typeof(pos))

    energies = rand(N*N) * (2 * E0) .- E0

    E = [1,0] / (N-1)
 
    temps = collect(100:10:400)
    velocities, occs, conv = run_HME(energies, pos, temps, E)

    py"""import numpy as np 
    np.save('velocities_test.npy', np.vstack(($(PyObject(temps)), np.linalg.norm($(PyObject(velocities)),axis=1))))
    np.save('final_occs_test.npy', $(PyObject(occs)))
    np.save('conv_test.npy', $(PyObject(conv)))
    np.save('pos_test.npy', $(PyObject(pos)))
    """

end