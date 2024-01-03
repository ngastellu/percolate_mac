module CorrEnenergies


    const kB = 8.617333262e-5 # Boltzmann constant in eV/K

    function generate_correlated_esites(pos, a, N1, N2, Ω, T, K, ν)
        # reciprocal_lattice = (2π/(a^2)) .* pos
        # reciprocal_lattice = reciprocal_lattice ./ [N1 N2 N2]
        Φ = zeros(N1,N2,N2)
        β = 1.0/(kB*T)
        # for (k,q) in enumerate(eachrow(reciprocal_lattice))
        #     σ = 1 / (β*K*norm(q)^2) # neglect factor of Ω bc we end up dividing by Ω when we FT and we wanna avoid overflow
        #     Φ[k] = randn() * σ 
        # end
        for i=1:N1
            for j=1:N2
                for k=1:N2
                    q = norm([i,j,k] ./ [N1, N2, N2]) * (2π/a)
                    σ = 1 / (β*K*(q^2)) # neglect factor of Ω bc we end up dividing by Ω when we FT and we wanna avoid overflow
                    Φ[i,j,k] = randn() * σ
                end
            end
        end
        println("max(Φ) = $(maximum(Φ))")
        # ϕ = zeros(N)
        # for n=1:N
        #     r = pos[n,:]
        #     if r == [0,0,0]
        #         ϕ[n] = sum(Φ)
        #         println(ϕ[n])
        #     else
        #         println(r)
        #         dotprods = [dot(r,q) for q in eachrow(reciprocal_lattice)]
        #         phases = exp.(-im .* dotprods)
        #         ϕ[n] = sum(Φ .* phases)
        #     end
        # end 
        ϕ = fft(Φ)
        return ν .*  ϕ, Φ
    end


    function generate_correlated_esites_1d(a, N, T, K, ν)
            Φ = zeros(N)
            β = 1.0/(kB*T)
            q = collect(1:N) .* (2π/N*a)
            for j=1:N
                σ = 1 / (β*K*(q[j]^2)) 
                f = randn() * σ
                Φ[j] = f
            end
            ϕ = ifft(fftshift(Φ))
            return ν .*  ϕ, Φ
    end

end