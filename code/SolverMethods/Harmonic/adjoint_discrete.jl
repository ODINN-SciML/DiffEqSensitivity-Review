# Discrete adjoint method

include("harmonic.jl")

function discrete_adjoint_method(u0, tspan, p, dt)
    u = u0
    times = tspan[1]:dt:tspan[2]

    λ = [1.0, 0.0]
    ∂L∂p = zeros(length(p))
    u_store = [u]

    # Forward pass to compute solution
    for t in times[1:(end - 1)]
        u += dt * f(u, p, t)
        push!(u_store, u)
    end

    # Reverse pass to compute adjoint
    for (i, t) in enumerate(reverse(times)[2:end])
        u_memory = u_store[end - i + 1]
        λ += dt * ∂f∂u(u_memory, p, t)' * λ
        ∂L∂p += dt * λ' * ∂f∂p(u_memory, p, t)
    end

    return ∂L∂p
end

dL∂p_disc = discrete_adjoint_method(u0, tspan, p, 0.001)

# Notice that there is still some numerical error in the case of the discrete adjoint
@test vec(dL∂p_disc)≈dLdp_SciML rtol=1e-3
