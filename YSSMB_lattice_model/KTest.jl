module KTest

    include("../HoppingMasterEquation.jl")

    using LinearAlgebra, Random, .HoppingMasterEquation, Plots
    
    N1 = 16
    N2=8
    a = 10

    kB = 8.617333262e-5
    
    println("Defining pos")
    
    pos = zeros(N1,N2,N2,3);
    for i=1:N1
        for j=1:N2
            for k=1:N2
                pos[i,j,k,:] = [i,j,k] .*a ;
            end
        end
    end
    
    pos = reshape(pos,N1*N2*N2,3);
    
    println("Getting innn")
    innn = get_nnn_inds_3d(pos, a)
    
    
    println("Getting energies")
    energies = randn(N1*N2*N2)*0.007;
    energies0 = zeros(N1*N2*N2);
    
    dX = (N1 - 1)*a
    E = [1,0,0] / dX
    T = 300
    
    
    println("Getting rate matrices")
    K =miller_abrahams_YSSMB(pos,energies, innn, T, E)
    K0 =miller_abrahams_YSSMB(pos,energies0, innn, T, E)
    
    
    N = size(pos,1)
    plot_data = ones(N*(N-1)÷2,4) .* -1
    k = 1
    
    println("Getting plot_data")
    for i=1:N
        for j=1:i-1
            ΔR = pos[i,:] - pos[j,:]
            if norm(ΔR) ≤ sqrt(2) * a
                println(k)
                plot_data[k,1] = norm(ΔR)
                plot_data[k,2] = ΔR[1]
                plot_data[k,3] = K0[i,j]
                plot_data[k,4] = K[i,j]
                global k+=1
            end
        end
    end
    plot_data = plot_data[plot_data[:,1].> 0,:]
    expected = exp.(-plot_data[:,1] - plot_data[:,2]/(2*dX*kB*T))

    diff = abs.(expected .- plot_data[:,3])
    println(size(diff))
    println(maximum(diff))
    println(maximum(K))
    println(maximum(K0))
    println(maximum(abs.(energies)))


    display(heatmap(collect(1:N),collect(1:N),K))
    
end