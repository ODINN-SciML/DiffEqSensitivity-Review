# Example adapted from https://github.com/ScientificComputingCWI/SemesterProgramme/blob/main/presentations/uncorrected_ad.jl
# based on the issue reported in https://github.com/SciML/SciMLSensitivity.jl/issues/273

using SciMLSensitivity, OrdinaryDiffEq, Zygote, ForwardDiff

reltol = abstol = 1e-12

function fiip(du, u, p, t)
    a = p[1]
    du[1] =  a * u[1] - u[1] * u[2]
    du[2] = -a * u[2] + u[1] * u[2]
end

p = [1.]
u0 = [1.0;1.0]
prob = ODEProblem(fiip, u0, (0.0, 10.0), p);

# Correct gradient computed using 
grad0 = Zygote.gradient(p->sum(solve(prob, Tsit5(), u0=u0, p=p, sensealg = ForwardSensitivity(), saveat = 0.1, abstol=abstol, reltol=reltol)), p)
# grad0 = ([212.71042521681443],)
@show grad0

# Original AD with wrong norm 
grad1 = Zygote.gradient(p->sum(solve(prob, Tsit5(), u0=u0, p=p, sensealg = ForwardDiffSensitivity(), saveat = 0.1, 
                               internalnorm = (u,t) -> sum(abs2,u/length(u)), 
                               abstol=abstol, reltol=reltol)), p)
# grad1 = ([6278.15677493293],)
@show grad1

# Let's see the stepsizes of this solve
sol1 = solve(prob, Tsit5(), u0=u0, p=p, sensealg = ForwardDiffSensitivity(),
                               internalnorm = (u,t) -> sum(abs2,u/length(u)), 
                               abstol=abstol, reltol=reltol)
times1 = sol1.t

# This issue has been fixed in https://github.com/SciML/SciMLSensitivity.jl/issues/273 by changing how the internal norm is computed 
# Corrected AD
grad2 = Zygote.gradient(p->sum(solve(prob, Tsit5(), u0=u0, p=p, sensealg = ForwardDiffSensitivity(), saveat = 0.1, abstol=abstol, reltol=reltol)), p)
@show grad2
# grad2 = ([212.71042521681082],)

# This is the same we will obtain if we manually change the internal norm to be the correct one
# Code based on the fixes introduced in the PR https://github.com/SciML/DiffEqBase.jl/pull/529/files
sse(x::Number) = x^2
sse(x::ForwardDiff.Dual) = sse(ForwardDiff.value(x)) + sum(sse, ForwardDiff.partials(x))

totallength(x::Number) = 1
totallength(x::ForwardDiff.Dual) = totallength(ForwardDiff.value(x)) + sum(totallength, ForwardDiff.partials(x))
totallength(x::AbstractArray) = sum(totallength,x)

grad3 = Zygote.gradient(p->sum(solve(prob, Tsit5(), u0=u0, p=p, sensealg = ForwardDiffSensitivity(), saveat = 0.1, 
                               internalnorm = (u,t) -> sqrt(sum(x->sse(x),u) / totallength(u)), 
                               abstol=abstol, reltol=reltol)), p)
@show grad3
# grad3 = ([212.71042521681392],)

# We can check using other method


# We can also customize controler