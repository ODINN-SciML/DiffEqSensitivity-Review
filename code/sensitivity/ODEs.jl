# Lotka Volterra with time-dependent interaction

cd(@__DIR__)
using Pkg;
Pkg.activate(".")
Pkg.instantiate()

using Test
using LinearAlgebra
using ForwardDiff # forward-mode AD
using ReverseDiff, Zygote # reverse-mode AD
using OrdinaryDiffEq
using SciMLSensitivity # using the SciML sensitivity tools

# Initial condition
u0 = [1.0, 1.0]
# LV equation parameter. p = [α, β, δ, γ]
p = [1.5, 1.0, 3.0, 1.0]
# integration time
tspan = (0.0, 1.0)
# time discretization
dt = 0.001

# system
function f(u, p, t)
    x, y = u
    α, β, δ, γ = p
    dx = α * x - β * x * y * t
    dy = -δ * y + γ * x * y * t
    [dx, dy]
end

### solving the ODE!

# hand-written Euler solver
function euler(f, u0, tspan, p, dt)
    u = u0
    t = tspan[1]:dt:tspan[2] # time grid
    for ti in t[1:end-1]
        u = u .+ f(u, p, ti) .* dt
    end
    u
end

# SciML

# Setup the ODE problem, then solve
prob = ODEProblem(f, u0, tspan, p)
sol = solve(prob, Euler(), dt = dt)
@test euler(f, u0, tspan, p, dt) ≈ sol.u[end] rtol = 1e-10


### Discrete-forward sensitivity analysis

# hand-written implementation
# Jacobian ∂f/∂u
function fu(u, p, t)
    (x, y, α, β, δ, γ) = (u[1], u[2], p[1], p[2], p[3], p[4])

    J = zeros(length(u), length(u))

    J[1, 1] = α - β * y * t
    J[2, 1] = γ * y * t
    J[1, 2] = - β * x * t
    J[2, 2] = - δ + x * γ * t

    J
end

# Parameter Jacobian ∂f/∂p
function fp(u, p, t)
    (x, y, α, β, δ, γ) = (u[1], u[2], p[1], p[2], p[3], p[4])

    Jp = zeros(length(u), length(p))

    Jp[1, 1] = x
    Jp[2, 1] = 0
    Jp[1, 2] = - x*y*t
    Jp[2, 2] = 0
    Jp[1, 3] = 0
    Jp[2, 3] = - y
    Jp[1, 4] = 0
    Jp[2, 4] = x*y*t

    Jp
end

function DFSA(u0, tspan, p, dt)
    u = u0
    t = tspan[1]:dt:tspan[2] # time grid

    vu = I(length(u))
    vp = zeros(length(u), length(p))

    for ti in t[1:end-1]
        vu = vu + dt * fu(u, p, ti) * vu
        vp = vp + dt * (fu(u, p, ti) * vp + fp(u, p, ti))
        u = u + f(u, p, ti) * dt # Euler step
    end
    u, vu, vp
end

u, vu, vp = DFSA(u0, tspan, p, dt)

vu_AD = ForwardDiff.jacobian(u0 -> euler(f, u0, tspan, p, dt), u0)
vp_AD = ForwardDiff.jacobian(p -> euler(f, u0, tspan, p, dt), p)

@test u ≈ euler(f, u0, tspan, p, dt)
@test vu ≈ vu_AD
@test vp ≈ vp_AD


### Discrete-adjoint sensitivity analysis

# with L = z_T

# custom implementation
function DASA(u0, tspan, p, dt)
    u = u0
    t = tspan[1]:dt:tspan[2] # time grid

    λu = [1.0, 0.0]
    λp = zeros(length(p))
    # forward pass
    us = [u]
    for ti in t[1:end-1]
        u = u + f(u, p, ti) * dt # Euler step
        push!(us, u)
    end

    # reverse pass
    usrev = reverse(us)[2:end]

    for (i, ti) in enumerate(reverse(t)[2:end])
        u = usrev[i]
        λp = λp + dt * vec(λu' * fp(u, p, ti))
        λu = λu + dt * vec(λu' * fu(u, p, ti))
    end

    us[1], vec(λu), vec(λp)
end


u0_, λu, λp = DASA(u0, tspan, p, dt)

λu_AD = ReverseDiff.gradient(u0 -> euler(f, u0, tspan, p, dt)[1], u0)
λp_AD = ReverseDiff.gradient(p -> euler(f, u0, tspan, p, dt)[1], p)

@test λu ≈ λu_AD
@test λp ≈ λp_AD


### Continuous-forward sensitivity analysis (identical to DFSA since non-adaptive)

function f_augmented_CFSA(z, p, t)
    u, vu, vp = z

    du = f(u, p, t)
    dvu = fu(u, p, t) * vu
    dvp = fu(u, p, t) * vp + fp(u, p, t)
    (du, dvu, dvp)
end

y0 = (u0, 1.0 * I(length(u0)), zeros(length(u0), length(p)))
y = euler(f_augmented_CFSA, y0, tspan, p, dt)

@test y[1] ≈ u
@test y[2] ≈ vu
@test y[3] ≈ vp

### Continuous-adjoint sensitivity analysis

function f_augmented_CASA(z, p, t)
    u, λu, λp = z

    du = f(u, p, t)

    dλu = -λu' * fu(u, p, t)
    dλp = -λu' * fp(u, p, t)

    (du, vec(dλu), vec(dλp))
end

uend = sol.u[end]
y0 = (uend, [1.0, 0.0], zeros(length(p)))
u0_, λu, λp = euler(f_augmented_CASA, y0, reverse(tspan), p, -dt)

function cost(u0, p)
    prob = ODEProblem(f, u0, tspan, p)
    u = Array(solve(prob, Euler(), dt=dt, save_everystep=false, sensealg=BacksolveAdjoint()))[1, end]
end
cost(u0, p)

λu_SciML, λp_SciML = Zygote.gradient((u0, p) -> cost(u0, p), u0, p)

@test λu ≈ λu_SciML
@test λp ≈ λp_SciML
