# Continuous Adjoint Method

include("harmonic.jl")
using RecursiveArrayTools

# Augmented dynamicis
function f_aug(z, p, t)
    u, λ, L = z
    du = f(u, p, t)
    dλ = - ∂f∂u(u, p, t)' * λ
    dL = - λ' * ∂f∂p(u, p, t)
    VectorOfArray([du, vec(dλ), vec(dL)])
end

# Solution of original ODE
prob = ODEProblem(f, u0, tspan, p)
sol = solve(prob, Euler(), dt=0.001)

# Final state 
u1 = sol.u[end]
z1 = VectorOfArray([u1, [1.0, 0.0], zeros(length(p))])

aug_prob = ODEProblem(f_aug, z1, reverse(tspan), p)
u0_, λ0, dLdp_cont = solve(aug_prob, Euler(), dt=-0.001).u[end]


@test dLdp_cont ≈ dLdp_SciML[1]