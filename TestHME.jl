module TestHME
    include("./HoppingMasterEquation.jl")

    using .HoppingMasterEquation, Random, PyCall, LinearAlgebra

    Random.seed!(64)


    n = 50
    E0 = 1.0
    ntrials = 100

    pos = zeros(n,n,2)

    for i=1:n
        for j=1:n
            pos[i,j,:] = [i-1,j-1]
        end
    end

    pos = reshape(pos, n*n, 2)

    L_inds = findall(pos[:,1] .== 0)
    R_inds = findall(pos[:,1] .== n-1)

    println(typeof(pos))

    E = [1,0] / (n-1)
    μL = 0.0
 
    temps = collect(100:10:400)
    # temps = collect(100:10:120)

    vs, occs0, occs, eners = lattice_hopping_model(pos, L_inds, R_inds, temps, E, μL,ntrials)

    

    py"""import numpy as np 
    np.save('pos_test.npy', $(PyObject(pos)))
    np.save('velocities_test.npy', np.vstack(($(PyObject(temps)), np.linaVlg.norm($(PyObject(vs)),axis=1))))
    # np.save('temps_test.npy', $(PyObject(temps)))
    # np.save('velocities_test.npy', $(PyObject(vs)))
    np.save('final_occs_test.npy', $(PyObject(occs)))
    np.save('initial_occs_test.npy', $(PyObject(occs0)))
    np.save('energies_test.npy', $(PyObject(eners)))
    # np.save('conv_test.npy', $(PyObject(conv)))
    """

end