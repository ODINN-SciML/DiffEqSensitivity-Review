# using Pkg
# Pkg.activate("../SensitivityEnv")

using SciMLSensitivity
using OrdinaryDiffEq
using Zygote

tspan = [0.0, 1000.0]
u0 = [0.0]

"""
    dyn!

This generates solutions u(t) = (t-θ)^5/5 that can be solved exactly with a 5th order integrator.
"""
function dyn!(du, u, t, p)
    θ = p[1]
    du .= (t .- θ).^4.0
end

p = [1.0] 

prob = ODEProblem(dyn!, u0, tspan, p)
sol  = solve(prob, Tsit5())

# We can see that the time steps increase with non-stop
@show diff(sol.t)

function loss(p, sensealg)
    prob = ODEProblem(dyn!, u0, tspan, p)
    if isnothing(sensealg)
        sol = solve(ODEProblem(dyn!, u0, tspan, p), Tsit5())
    else
        sol = solve(ODEProblem(dyn!, u0, tspan, p), Tsit5(), sensealg=sensealg)
    end
    @show diff(sol.t)
    sol.u[end][1]
end

function grad_true(p)
    θ = p[1]
    t = tspan[2]
    θ^4 - (t - θ)^4
end

"""
An implementation of discrete forward sensitivity analysis through ForwardDiff.jl. 
When used within adjoint differentiation (i.e. via Zygote), this will cause forward differentiation 
of the solve call within the reverse-mode automatic differentiation environment.

https://docs.sciml.ai/SciMLSensitivity/stable/manual/differential_equation_sensitivities/#SciMLSensitivity.ForwardDiffSensitivity
"""
g1 = Zygote.gradient(p -> loss(p, ForwardDiffSensitivity()), p)
@show g1

# g2 = Zygote.gradient(p -> loss(p, ForwardSensitivity()), p)
# @show g2

g3 = Zygote.gradient(p -> loss(p, nothing), p)
@show g3

@show grad_true(p)

# Define customized RK(4) solver with given timesteps to show the divergence of forward sensitivities