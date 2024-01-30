# Let's see how finite differences behaves at the moment of computing the
# derivative of the solution of a differential equation

using OrdinaryDiffEq
using CairoMakie
using ComplexDiff

# Parameters 
u0 = [0.0, 1.0]
p = [20.0]
t₀, t₁ = 0.0, 10.0
tspan = (t₀, t₁)
reltol = 1e-5
abstol = 1e-5

# Function to define the ODE dynamics of the harmonic oscilatior
function oscilatior!(du, u, p, t)
    ω = p[1]
    du[1] = u[2]
    du[2] = - ω^2 * u[1]
    nothing
end

# Real solution of the differential equation as function of ω and its derivative wrt ω
function solution(t, u0, p)
    ω = p[1]
    A₀ = u0[2] / ω
    B₀ = u0[1]
    return A₀ * sin(ω * t) + B₀ * cos(ω * t)
end

function solution_derivative(t, u0, p)
    ω = p[1]
    A₀ = u0[2] / ω
    B₀ = u0[1]
    return A₀ * ( t * cos(ω * t) - sin(ω * t)/ω ) - B₀ * t * sin(ω * t)
end

######### Simple example of how to run the dynamcics ###########

# Solve numerical problem
prob = ODEProblem(oscilatior!, u0, tspan, p)
sol = solve(prob, Tsit5(), reltol=reltol, abstol=abstol)

u_final = sol.u[end][1]

# plot(sol)
# plot!(sol.t, solution.(sol.t, Ref(u0), Ref(p)))
# savefig("solution.png")

######### Numerical Differentiation ###########

function finitediff_numerical(h, t, u0, p)
    p₊ = [p[1] + h]
    p₋ = [p[1] - h]
    return (solution(t, u0, p₊) - solution(t, u0, p₋)) / (2h)
end

function finitediff_solver(h, t, u0, p, reltol, abstol)
    p₊ = [p[1] + h]
    p₋ = [p[1] - h]
    tspan = (0.0, t)
    # Forward model with -h
    prob₋ = ODEProblem(oscilatior!, u0, tspan, p₋)
    sol₋ = solve(prob₋, Tsit5(), reltol=reltol, abstol=abstol)
    # Forward model with +h
    prob₊ = ODEProblem(oscilatior!, u0, tspan, p₊)
    sol₊ = solve(prob₊, Tsit5(), reltol=reltol, abstol=abstol)

    return (sol₊.u[end][1] - sol₋.u[end][1]) /(2h)
end

function complexdiff_solver(h, t, u0, p, reltol, abstol)
    function simple_sol(ω)
        tspan = (0.0, t)
        prob = ODEProblem(oscilatior!, u0, tspan, [ω])
        sol = solve(prob, Tsit5(), reltol=reltol, abstol=abstol)
        return sol.u[end][1]
    end
    return ComplexDiff.derivative(simple_sol, p[1], h)
end

######### Simulation with differerent stepsizes ###########

stepsizes = 2.0.^collect(round(log2(eps(Float64))):1:0)
times = collect(t₀:1.0:t₁)

# True derivative computend analytially
derivative_true = solution_derivative(t₁, u0, p)

# Numerical finite differences solution computed with real analytical solution
# derivative_numerical = finitediff_numerical.(stepsizes, Ref(t₁), Ref(u0), Ref(p))
derivative_numerical = finitediff_numerical.(stepsizes, Ref(t₁), Ref(u0), Ref(p))
error_numerical = abs.((derivative_numerical .- derivative_true)./derivative_true)

# Finite differences with solution from solver and low tolerance
derivative_solver_low = finitediff_solver.(stepsizes, Ref(t₁), Ref(u0), Ref(p), Ref(1e-5), Ref(1e-6))
error_solver_low = abs.((derivative_solver_low .- derivative_true)./derivative_true)

# Finite differences with solution from solver and high tolerance
derivative_solver_high = finitediff_solver.(stepsizes, Ref(t₁), Ref(u0), Ref(p), Ref(1e-12), Ref(1e-12))
error_solver_high = abs.((derivative_solver_high .- derivative_true)./derivative_true)

# Complex step differentiation with solution from solver and high tolerance
# derivative_solver_high_complex = complexdiff_solver.(stepsizes, Ref(t₁), Ref(u0), Ref(p), Ref(1e-12), Ref(1e-12))
# error_solver_high_complex = abs.((derivative_solver_high_complex .- derivative_true)./derivative_true)

# Complex step Differentiation
derivative_complex = ComplexDiff.derivative.(ω -> solution(t₁, u0, [ω]), p[1], stepsizes)
error_complex = abs.((derivative_complex .- derivative_true)./derivative_true)


######### Figure ###########



fig = Figure(resolution=(900, 500)) 
ax = Axis(fig[1, 1], xlabel = L"Stepsize ($\varepsilon$)", ylabel = L"\text{Relative error}", 
          xscale = log10, yscale=log10)

scatter!(ax, stepsizes, error_complex,     xscale=log10, yscale=log10, label=L"\text{Complex step differentiation}", color=:green, zorder=1)
scatter!(ax, stepsizes, error_numerical,   xscale=log10, yscale=log10, label=L"\text{Exact Solution}", zorder=2)
scatter!(ax, stepsizes, error_solver_high, xscale=log10, yscale=log10, label=L"Numerical solution (tol=$10^{-12}$)", color=:red, zorder=3)
# scatter!(ax, stepsizes, error_solver_high_complex, xscale=log10, yscale=log10, label=L"Numerical solution Complex (tol=$10^{-12}$)", color=:violet, zorder=3)
scatter!(ax, stepsizes, error_solver_low,  xscale=log10, yscale=log10, label=L"Numerical solution (tol=$10^{-6}$)", color=:orange, zorder=4)

# Add legend
fig[1, 2] = Legend(fig, ax)

save("finite_differences/finite_difference_derivative.pdf", fig)
