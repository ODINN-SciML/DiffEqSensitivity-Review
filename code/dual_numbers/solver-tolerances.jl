using Pkg
Pkg.activate("../SensitivityEnv")

using SciMLSensitivity
using OrdinaryDiffEq


tspan = [0.0, 1000.0]
u0 = [0.0]

# This generates solutions u(t) = t^5/5 that can be solved exactly with a 5th order integrator
function dyn!(du, u, t)
    du .= t^4.0
end

prob = ODEProblem(dyn!, u0, tspan)
sol  = solve(prob, Tsit5())

# We can see that the time steps increase with non-stop
@show diff(sol.t)