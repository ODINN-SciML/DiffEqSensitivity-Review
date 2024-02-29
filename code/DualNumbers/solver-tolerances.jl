using Pkg
Pkg.activate(".")

using SciMLSensitivity
using OrdinaryDiffEq
using Zygote
using ForwardDiff
using Infiltrator






tspan = (0.0, 10.0)
u0 = [0.0]
reltol = 1e-6
abstol = 1e-6

"""
    dyn!

This generates solutions u(t) = (t-θ)^5/5 that can be solved exactly with a 5th order integrator.
"""
function dyn!(du, u, p, t)
    θ = p[1]
    du .= (t .- θ).^4.0
end

p = [1.0] 

prob = ODEProblem(dyn!, u0, tspan, p)
sol  = solve(prob, Tsit5(), reltol=reltol, abstol=abstol)

# We can see that the time steps increase with non-stop
# @show diff(sol.t)

function loss(p, sensealg)
    prob = ODEProblem(dyn!, u0, tspan, p)
    if isnothing(sensealg)
        sol = solve(prob, Tsit5(), reltol=reltol, abstol=abstol)
    else
        sol = solve(prob, Tsit5(), sensealg=sensealg, reltol=reltol, abstol=abstol)
    end
    @show "Number of time steps: ", length(sol.t)
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
# Original AD without correction

condition(u, t, integrator) = true
function printstepsize!(integrator)
    # @infiltrate
    if length(integrator.sol.t) > 1
        # println("Stepsize at step ", length(integrator.sol.t), ":   ", integrator.sol.t[end] - integrator.sol.t[end-1])
    end
end

cb = DiscreteCallback(condition, printstepsize!)

# g1 = Zygote.gradient(p -> loss(p, ForwardDiffSensitivity()), internalnorm = (u,t) -> sum(abs2,u/length(u)), p)
g1 = Zygote.gradient(p -> solve(ODEProblem(dyn!, u0, tspan, p), 
                                Tsit5(), 
                                u0 = u0, 
                                p = p, 
                                sensealg = ForwardDiffSensitivity(), 
                                saveat = 0.1,
                                internalnorm = (u,t) -> sum(abs2, u/length(u)), 
                                callback = cb, 
                                reltol=1e-6, 
                                abstol=1e-6).u[end][1], p)
@show g1

# Forward Sensitivity
# g2 = Zygote.gradient(p -> loss(p, ForwardSensitivity()), p)
# g2 = Zygote.gradient(p -> solve(prob, 
#                                 Tsit5(), 
#                                 sensealg = ForwardSensitivity(), 
#                                 saveat = 0.1,
#                                 callback = cb, 
#                                 reltol=1e-12, 
#                                 abstol=1e-12).u[end][1], p)
# @show g2

# Corrected AD
# g3 = ForwardDiff.gradient(p -> loss(p, nothing), p)
g3 = Zygote.gradient(p -> solve(ODEProblem(dyn!, u0, tspan, p), 
                                Tsit5(), 
                                sensealg = ForwardDiffSensitivity(), 
                                # saveat = 0.1,
                                # callback = cb, 
                                reltol=1e-6, 
                                abstol=1e-6).u[end][1], p)
@show g3

@show grad_true(p)

# Define customized RK(4) solver with given timesteps to show the divergence of forward sensitivities