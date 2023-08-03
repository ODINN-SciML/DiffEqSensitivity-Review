# Let's see how finite differences behaves at the moment of computing the
# derivative of the solution of a differential equation

using OrdinaryDiffEq
# using Plots
using CairoMakie

function oscilatior!(du, u, p, t)
    ω = p[1]
    du[1] = u[2]
    du[2] = - ω^2 * u[1]
    nothing
end

function solution(t, u0, p)
    # We fix u(t=0) = 0
    ω = p[1]
    return (u0[2] / ω) * sin(ω * t)
end

# Initial condition
u0 = [0.0, 1.0]
p = [20.0]
t₀, t₁ = 0.0, 10.0
tspan = (t₀, t₁)
reltol = 1e-5
abstol = 1e-5

# Solve numerical problem
prob = ODEProblem(oscilatior!, u0, tspan, p)
sol = solve(prob, Tsit5(), reltol=reltol, abstol=abstol)

u_final = sol.u[end][1]

# plot(sol)
# plot!(sol.t, solution.(sol.t, Ref(u0), Ref(p)))
# savefig("solution.png")

# Let's differentiate the final state of the differential equation using finite differences for different stepsizes

function solution_derivative(t, u0, p)
    # Compute derivative with respect to the parameter ω
    ω = p[1]
    return u0[2] * ( t * cos(ω * t) / ω - sin(ω * t) / (ω^2) )
end

function finitediff_real(h, t, u0, p)
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

stepsizes = 2.0.^collect(round(log2(eps(Float64))):1:0)

derivative_true = solution_derivative(t₁, u0, p)

derivative1 = finitediff_real.(stepsizes, Ref(t₁), Ref(u0), Ref(p))
error1 = abs.((derivative1 .- derivative_true)./derivative_true)

derivative2 = finitediff_solver.(stepsizes, Ref(t₁), Ref(u0), Ref(p), Ref(1e-5), Ref(1e-6))
error2 = abs.((derivative2 .- derivative_true)./derivative_true)

derivative3 = finitediff_solver.(stepsizes, Ref(t₁), Ref(u0), Ref(p), Ref(1e-12), Ref(1e-12))
error3 = abs.((derivative3 .- derivative_true)./derivative_true)

# Makie Figure
fig = Figure(resolution=(900, 500)) 
ax = Axis(fig[1, 1], xlabel = "Stepsize", ylabel = "Absolute error",
    title = "Absolute Relative of the Derivative", xscale = log2, yscale=log10)
scatter!(ax, stepsizes, error1, xscale=log2, yscale=log10, label="Exact solution")
scatter!(ax, stepsizes, error2, xscale=log2, yscale=log10, label=L"Numerical solution (tol=$10^{-6}$)")
scatter!(ax, stepsizes, error3, xscale=log2, yscale=log10, label=L"Numerical solution (tol=$10^{-12}$)")

# Add legend
fig[1, 2] = Legend(fig, ax)

save("finite_difference_derivative.pdf", fig)
