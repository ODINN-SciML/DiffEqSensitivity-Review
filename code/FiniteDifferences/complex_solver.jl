using OrdinaryDiffEq
using CairoMakie
using ComplexDiff
using Zygote, ForwardDiff, SciMLSensitivity

function oscilatior!(du_complex::Array{Complex{Float64}}, u::Array{Complex{Float64}}, p, t)
    ω = p[1]
    du_complex[1] = u[2]
    du_complex[2] = - ω^2 * u[1]
    nothing
end

du_complex = Array{Complex{Float64}}([0.0])
u0_complex = Array{Complex{Float64}}([0.0, 1.0])
p_complex = Array{Complex{Float64}}([20.]) .+ 0.1im

function complexstep_differentiation(f::Function, p::Float64, h::Float64)
    p_complex = p .+ h * im
    res = f(p_complex)
    return imag(res) / h
end

# sol = solve(ODEProblem(oscilatior!, u0_complex, tspan, p_complex), Tsit5(), reltol=1e-6, abstol=1e-6)
deriv = complexstep_differentiation(x -> solve(ODEProblem(oscilatior!, u0_complex, tspan, [x]), Tsit5(), reltol=1e-6, abstol=1e-6).u[end][1], 20., 0.001)
# deriv = ComplexDiff.derivative(x -> solve(ODEProblem(oscilatior!, u0_complex, tspan, x), Tsit5(), reltol=1e-6, abstol=1e-6), [20.], 0.1)
