using Plots;
gr();
using Statistics
using LinearAlgebra
using Zygote
using Tullio
using Statistics

nx, ny = 100, 100          # size of the grid
Δx, Δy = 1, 1              # lenght of each element in the grid (meters)
t₁ = 1                     # final time in simulation

D₀ = 1                     # real diffusivity parameters
tolnl = 1e-6               # tolerance of the numerical method (If this is zero, the are no wigles)
itMax₀ = 100               # maximum number of iterations of numerical method
damp = 0.85                # damping factor of numerical method
dτsc = 1.0 / 10.0           # step in implicit numerical method (must be < 1.0)
adaptive = true            # adaptive stepsize

# Stable stepsize
dτ(x) = dτsc * Δx^2 / (4x)

if !adaptive
    Δt = dτ(2D₀)
    @assert Δt / (Δx^2) < 1 / (4 * (2D₀))
end

function heatflow(T, D::Real, p, tol = tolnl, itMax = itMax₀, adaptive = adaptive)
    Δx, Δy, Δt, t₁ = p

    total_iter = 0
    t = 0

    if adaptive
        δt = min(t₁ - t, dτ(D))
    else
        δt = Δt
    end

    while t < t₁
        iter = 1
        Hold = copy(T)
        dTdt = zeros(nx, ny)
        err = Inf

        while iter < itMax + 1 && tol <= err
            Err = copy(T)

            F, dτ = Heat(T, D, (Δx, Δy, δt, t₁), adaptive)

            @tullio ResT[i, j] := -(T[i, j] - Hold[i, j]) / δt +
                                  F[pad(i - 1, 1, 1), pad(j - 1, 1, 1)]

            dTdt_ = copy(dTdt)
            @tullio dTdt[i, j] := dTdt_[i, j] * damp + ResT[i, j]

            T_ = copy(T)
            @tullio T[i, j] := max(0.0, T_[i, j] + dτ * dTdt[i, j])

            Zygote.ignore() do
                Err .= Err .- T
                err = maximum(Err)
                # if err < tol
                #     println("Number of steps: ", iter)
                # end
            end

            iter += 1
            total_iter += 1
        end

        #println("total iterations: ", iter)

        t += δt
    end
    return T
end

function Heat(T, D, p, adaptive)
    Δx, Δy, Δt, t₁ = p

    dTdx_edges = diff(T[:, 2:(end - 1)], dims = 1) / Δx
    dTdy_edges = diff(T[2:(end - 1), :], dims = 2) / Δy

    Fx = -D * dTdx_edges
    Fy = -D * dTdy_edges
    F = .-(diff(Fx, dims = 1) / Δx .+ diff(Fy, dims = 2) / Δy)

    if adaptive
        dτ_ = dτ(D)
    else
        dτ_ = dτ(2D₀)
    end

    return F, dτ_
end

### Generate reference dataset
Δt = 0.1
p = (Δx, Δy, Δt, t₁)

# initial condition
T₀ = [250 * exp(-((i - nx / 2)^2 + (j - ny / 2)^2) / 500) for i in 1:nx, j in 1:ny]

# simulated evolution
T₁ = copy(T₀)
T₁ = heatflow(T₁, D₀, p, 1e-6, 1000, adaptive)

gr()
heatmap(T₀, clim = (0, maximum(T₀)))
savefig("T0_initial_state.png")

heatmap(T₁, clim = (0, maximum(T₀)))
savefig("T1_final_state.png")

@show maximum(T₁ .- T₀)
@show sqrt(sum((T₁ .- T₀) .^ 2) / (nx * ny))

### Automatic Differentiation on the heat equation with respect to D
mse(ŷ, y) = mean(abs2, ŷ .- y)

function loss(T, θ, p)
    uD = θ[1]
    T = heatflow(T, uD, p, adaptive)
    l_H = mse(T, T₁)
    return l_H
end

all_D = LinRange(D₀ / 2, 3D₀ / 2, 50)
all_loss = zeros(0)
all_grad = zeros(0)

for d in all_D
    T = T₀
    loss_uD, back_uD = Zygote.pullback(D -> loss(T, D, p), d)

    println("difusivity: ", d)
    println("loss: ", loss_uD)
    #println("gradient: ", back_uD(1), "\n")

    append!(all_loss, loss_uD)
    append!(all_grad, back_uD(1)[1])
end

plot(all_D, all_loss)
vline!([D₀])
savefig("Loss_function.png")

plot(all_D, all_grad)
hline!([0])
savefig("Gradient.png")
