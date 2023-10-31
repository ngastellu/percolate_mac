module RunHME

    include("./HoppingMasterEquation.jl")

    using .HoppingMasterEquation, PyCall

    nsample = 150

    percolate_datadir = "/Users/nico/Desktop/simulation_outputs/percolation/40x40/percolate_output/sample-$(nsample)/"

    py"""
    import numpy as np
    centers = np.load($percolate_datadir + 'cc.npy')
    site_energies = np.load($percolate_datadir + 'ee.npy')
    """

    centers = PyArray(py"centers"o)
    site_energies = PyArray(py"site_energies"o)

    E = [1 0]/400

    temps = collect(100:10:400)
    velocities, occs, conv = run_HME(site_energies, centers, L_inds, R_inds, temps, E, μL; ϵ=1e-7)

    py"""np.save('velocities.npy', np.vstack(($(PyObject(temps)), np.linalg.norm($(PyObject(velocities)),axis=1))))
    np.save('final_occs.npy', $(PyObject(occs)))
    np.save('conv.npy', $(PyObject(conv)))
    """
    
end