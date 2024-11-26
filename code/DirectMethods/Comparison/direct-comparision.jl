# Let's see how finite differences behaves at the moment of computing the
# derivative of the solution of a differential equation

using OrdinaryDiffEq
using CairoMakie
using ComplexDiff
using Zygote, ForwardDiff, SciMLSensitivity

include("../ComplexStep/complex_solver.jl")

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
    du[2] = -ω^2 * u[1]
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
    return A₀ * (t * cos(ω * t) - sin(ω * t) / ω) - B₀ * t * sin(ω * t)
end

######### Simple example of how to run the dynamcics ###########

# Solve numerical problem
prob = ODEProblem(oscilatior!, u0, tspan, p)
sol = solve(prob, Tsit5(), reltol = reltol, abstol = abstol)

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
    sol₋ = solve(prob₋, Tsit5(), reltol = reltol, abstol = abstol)
    # Forward model with +h
    prob₊ = ODEProblem(oscilatior!, u0, tspan, p₊)
    sol₊ = solve(prob₊, Tsit5(), reltol = reltol, abstol = abstol)

    return (sol₊.u[end][1] - sol₋.u[end][1]) / (2h)
end

######### Simulation with differerent stepsizes ###########

stepsizes = 2.0 .^ collect(round(log2(eps(Float64))):2:0)
times = collect(t₀:1.0:t₁)

# True derivative computend analytially
derivative_true = solution_derivative(t₁, u0, p)

# Numerical finite differences solution computed with real analytical solution
# derivative_numerical = finitediff_numerical.(stepsizes, Ref(t₁), Ref(u0), Ref(p))
derivative_numerical = finitediff_numerical.(stepsizes, Ref(t₁), Ref(u0), Ref(p))
derivative_finitediff_exact = finitediff_numerical.(stepsizes, Ref(t₁), Ref(u0), Ref(p))
error_finitediff_exact = abs.((derivative_numerical .- derivative_true) ./ derivative_true)

# Finite differences with solution from solver and low tolerance
derivative_solver_low = finitediff_solver.(
    stepsizes, Ref(t₁), Ref(u0), Ref(p), Ref(1e-6), Ref(1e-6))
error_finitediff_low = abs.((derivative_solver_low .- derivative_true) ./ derivative_true)

# Finite differences with solution from solver and high tolerance
derivative_solver_high = finitediff_solver.(
    stepsizes, Ref(t₁), Ref(u0), Ref(p), Ref(1e-12), Ref(1e-12))
error_finitediff_high = abs.((derivative_solver_high .- derivative_true) ./ derivative_true)

# Complex step differentiation with solution from solver and high tolerance
u0_complex = ComplexF64.(u0)

derivative_complex_low = complexstep_differentiation.(
    Ref(x -> solve(ODEProblem(oscilatior!, u0_complex, tspan, [x]), Tsit5(), reltol = 1e-6, abstol = 1e-6).u[end][1]),
    Ref(p[1]),
    stepsizes)
error_complex_low = abs.((derivative_true .- derivative_complex_low) ./ derivative_true)
derivative_complex_high = complexstep_differentiation.(
    Ref(x -> solve(ODEProblem(oscilatior!, u0_complex, tspan, [x]), Tsit5(), reltol = 1e-12, abstol = 1e-12).u[end][1]),
    Ref(p[1]),
    stepsizes)
error_complex_high = abs.((derivative_true .- derivative_complex_high) ./ derivative_true)

# Complex step Differentiation
derivative_complex_exact = ComplexDiff.derivative.(
    ω -> solution(t₁, u0, [ω]), p[1], stepsizes)
error_complex_exact = abs.((derivative_complex_exact .- derivative_true) ./ derivative_true)

# Forward AD applied to numerical solver
derivative_AD_low = Zygote.gradient(
    p -> solve(ODEProblem(oscilatior!, u0, tspan, p), Tsit5(), reltol = 1e-6, abstol = 1e-6).u[end][1],
    p)[1][1]
error_AD_low = abs((derivative_true - derivative_AD_low) / derivative_true)

derivative_AD_high = Zygote.gradient(
    p -> solve(ODEProblem(oscilatior!, u0, tspan, p), Tsit5(), reltol = 1e-12, abstol = 1e-12).u[end][1],
    p)[1][1]
error_AD_high = abs((derivative_true - derivative_AD_high) / derivative_true)

######### Figure ###########

color_finitediff = RGBf(192 / 255, 57 / 255, 43 / 255)
color_complex = RGBf(41 / 255, 128 / 255, 185 / 255)
color_AD = RGBf(142 / 255, 68 / 255, 173 / 255)
color_AD_low = RGBf(155 / 255, 89 / 255, 182 / 255)

fig = Figure(resolution = (1200, 400))
ax_low = Axis(fig[1, 1], xlabel = L"Stepsize ($\varepsilon$)",
    ylabel = L"\text{Absolute relative error}", xscale = log10, yscale = log10, title = L"Low Tolerance Solver (tol=$10^{-6}$)", titlesize = 24,
    xlabelsize = 24, ylabelsize = 24, xticklabelsize = 18, yticklabelsize = 18)

ax_high = Axis(fig[1, 2], xlabel = L"Stepsize ($\varepsilon$)",
    xscale = log10, yscale = log10, title = L"High Tolerance Solver (tol=$10^{-12}$)", titlesize = 24,
    xlabelsize = 24, ylabelsize = 24, xticklabelsize = 18, yticklabelsize = 18)

# Plot derivatived of true solution (no numerical solver)
for ax in (ax_low, ax_high)
    lines!(ax, stepsizes, error_finitediff_exact,
        label = L"\text{Finite differences (exact solution)}",
        color = color_finitediff, linewidth = 2, linestyle = :solid)
    lines!(ax, stepsizes, error_complex_exact,
        label = L"\text{Complex step differentiation (exact solution)}",
        color = color_complex, linewidth = 2, linestyle = :solid)
end

# Plot derivatives computed on top of numerical solver with finite differences
scatter!(
    ax_low, stepsizes, error_finitediff_low, label = L"\text{Finite differences}",
    color = color_finitediff, marker = '•', markersize = 40)
scatter!(
    ax_high, stepsizes, error_finitediff_high, label = L"\text{Finite differences}",
    color = color_finitediff, marker = '•', markersize = 40)

# Plot derivatives computed on top of numerical solver with complex step method
scatter!(ax_low, stepsizes, error_complex_low,
    label = L"\text{Complex step differentiation}",
    color = color_complex, marker = '∘', markersize = 40)
scatter!(ax_high, stepsizes, error_complex_high,
    label = L"\text{Complex step differentiation}",
    color = color_complex, marker = '∘', markersize = 40)

# AD
lines!(ax_low, stepsizes, repeat([error_AD_low], length(stepsizes)), linestyle = :solid, 
    color = color_AD, label = L"\text{Forward AD}", linewidth = 3)
lines!(ax_high, stepsizes, repeat([error_AD_high], length(stepsizes)), linestyle = :solid,
    color = color_AD, label = L"\text{Forward AD}", linewidth = 3)

# Add legend
fig[1, 3] = Legend(fig, ax_low)

# Hide y-label axes of middle plot
hideydecorations!(ax_high, grid = false)

!ispath("Figures") && mkpath("Figures")
save("Figures/DirectMethods_comparison.pdf", fig)

######### Benchmark ###########

# It looks like complex step has better performance... both in speed and momory allocation.
# @benchmark derivative_complex_low = complexstep_differentiation.(Ref(x -> solve(ODEProblem(oscilatior!, u0_complex, tspan, [x]), Tsit5(), reltol=1e-6, abstol=1e-6).u[end][1]), Ref(p[1]), [1e-5])
# @benchmark derivative_AD_low = Zygote.gradient(p->solve(ODEProblem(oscilatior!, u0, tspan, p), Tsit5(), reltol=1e-6, abstol=1e-6).u[end][1], p)[1][1]
