"""
Harmonic oscilator with sensitivity equations

"""

ω = 0.2
p = [ω]
u0 = [0.0, 1.0]
tspan = [0.0, 10.0]

# Dynamics
function f(u, p, t)
    du₁ = u[2]
    du₂ = -p[1]^2 * u[1]
    return [du₁, du₂]
end

function f!(du, u, p, t)
    du[1] = u[2]
    du[2] = -p[1]^2 * u[1]
end

# Jacobian ∂f/∂p
function ∂f∂p(u, p, t)
    Jac = zeros(length(u), length(p))
    Jac[2, 1] = -2 * p[1] * u[1]
    return Jac
end

# Jacobian ∂f/∂u
function ∂f∂u(u, p, t)
    Jac = zeros(length(u), length(u))
    Jac[1, 2] = 1
    Jac[2, 1] = -p[1]^2
    return Jac
end

# Explicit Euler method
function sensitivityequation(u0, tspan, p, dt)
    u = u0
    sensitivity = zeros(length(u), length(p))
    for ti in tspan[1]:dt:tspan[2]
        sensitivity += dt * (∂f∂u(u, p, ti) * sensitivity + ∂f∂p(u, p, ti))
        u += dt * f(u, p, ti)
    end
    return u, sensitivity
end

u, s = sensitivityequation(u0, tspan, p, 0.001)

using OrdinaryDiffEq, ForwardDiff, Test

s_AD = ForwardDiff.jacobian(p -> solve(ODEProblem(f, u0, tspan, p), Tsit5()).u[end], p)
@test s_AD≈s rtol=0.01

### Let's do this with forward sensitivity

function loss(p)
    solve(ODEProblem(f!, u0, tspan, p), Tsit5(), sensealg = ForwardSensitivity())[end]
end

using Zygote, SciMLSensitivity

s_sens = Zygote.jacobian(loss, p)

using Zygote, SciMLSensitivity
s = Zygote.jacobian(
    p -> solve(ODEProblem(f!, u0, tspan, p), Tsit5(), sensealg = ForwardSensitivity())[end],
    p)

# Discrete adjoint method

function discrete_adjoint_method(u0, tspan, p, dt)
    u = u0
    times = tspan[1]:dt:tspan[2]

    λ = [1.0, 0.0]
    ∂L∂θ = zeros(length(p))
    u_store = [u]

    # Forward pass to compute solution
    for t in times[1:(end - 1)]
        u += dt * f(u, p, t)
        push!(u_store, u)
    end

    # Reverse pass to compute adjoint
    for (i, t) in enumerate(reverse(times)[2:end])
        u_memory = u_store[end - i + 1]
        λ += dt * ∂f∂u(u_memory, p, t)' * λ
        ∂L∂θ += dt * λ' * ∂f∂p(u_memory, p, t)
    end

    return ∂L∂θ
end

∂L∂θ = discrete_adjoint_method(u0, tspan, p, 0.001)
