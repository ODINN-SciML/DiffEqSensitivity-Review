"""
Harmonic oscilator
"""

using OrdinaryDiffEq
using SciMLSensitivity
using Zygote
using Test

ω = 0.2
p = [ω]
u0 = [0.0, 1.0]
tspan = [0.0, 10.0]

# Dynamics
function f(u, p, t)
    du₁ = u[2]
    du₂ = - p[1]^2 * u[1]
    return [du₁, du₂]
end

function f!(du, u, p, t)
    du[1] = u[2]
    du[2] = - p[1]^2 * u[1]
end

# Jacobian ∂f/∂p
function ∂f∂p(u, p, t)
    Jac = zeros(length(u), length(p))
    Jac[2,1] = -2*p[1]*u[1]
    return Jac
end

# Jacobian ∂f/∂u
function ∂f∂u(u, p, t)
    Jac = zeros(length(u), length(u))
    Jac[1,2] = 1
    Jac[2,1] = -p[1]^2
    return Jac
end

# Ground truth gradient

function cost(p)
    prob = ODEProblem(f, u0, tspan, p)
    return solve(prob, Euler(), dt=0.001, save_everystep=false, sensealg=BacksolveAdjoint()).u[end][1]
end
cost(p)

dLdp_SciML = Zygote.gradient(p -> cost(p), p)[1]