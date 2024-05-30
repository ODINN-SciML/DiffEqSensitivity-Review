# Forward sensitivity equations

include("harmonic.jl")

function sensitivityequation(u0, tspan, p, dt)
    u = u0
    sensitivity = zeros(length(u), length(p))
    for ti in tspan[1]:dt:tspan[2]
        sensitivity += dt * (∂f∂u(u, p, ti) * sensitivity + ∂f∂p(u, p, ti))
        u += dt * f(u, p, ti)
    end
    return u, sensitivity
end

u, s = sensitivityequation(u0, tspan , p, 0.001)

using OrdinaryDiffEq, ForwardDiff, Test 

s_AD = ForwardDiff.jacobian(p -> solve(ODEProblem(f, u0, tspan, p), Tsit5()).u[end], p)

@test s_AD ≈ s rtol=0.01

### Let's do this with SciMLSensitivity

prob = ODEForwardSensitivityProblem(f!, u0, tspan, p)
sol = solve(prob, Tsit5())
u, dudp = extract_local_sensitivities(sol)

@test dudp[1][:, end] ≈ s_AD rtol=1e-3