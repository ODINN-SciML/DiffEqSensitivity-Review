using ComponentArrays
using FreezeCurves
using LinearAlgebra
using OrdinaryDiffEq
using SciMLSensitivity
using Statistics
using Zygote

# plotting
import Plots

const L = 3.34e8 # volumetric latent heat of fusion of water [J/m^3]

const default_p = ComponentVector(
    θwi=0.5,  # total water+ice content; 1.0 would correspond to a block of pure ice/water
    k_s=2.5,  # thermal conductivity of solid material [W/(m.K)]
    k_w=0.57, # thermal conductivity of unfrozen water [W/(m.K)]
    k_i=2.2,  # thermal conductivity of ice [W/(m.K)]
    c_s=2.5e6, # heat capacity of solid material [J/(m^3.K)]
    c_w=4.2e6, # heat capacity of unfrozen water [J/(m^3.K)]
    c_i=1.9e6, # heat capacity of ice [J/(m^3.K)]
    jH_lb=0.05, # lower boundary flux [W/m^2]
)

struct Grid1D{TA}
    edges::TA
    cells::TA
    thick::TA
    dists::TA
end
function Grid1D(grid_edges::AbstractVector)
    cells = (grid_edges[1:end-1] .+ grid_edges[2:end]) ./ 2
    thick = diff(grid_edges)
    dists = diff(cells)
    return Grid1D(grid_edges, cells, thick, dists)
end

function steady_state_init(grid::Grid1D, p; Tsurf=-10.0)
    T0 = Tsurf .+ p.jH_lb / p.k_s.*grid.cells
    H0 = enthalpy.(T0, Ref(p))
    return T0, H0
end

function heatcapacity(θw, p)
    c = (1 - p.θwi)*p.c_s + θw*p.c_w + (1-θw)*p.c_i
    return c
end

function enthalpy(T, p)
    if T < zero(T)
        T*heatcapacity(zero(p.θwi), p)
    else
        T*heatcapacity(p.θwi, p) + L*p.θwi
    end
end

function enthalpyinv(H, p)
    θwi = p.θwi
    θw, Lθ = FreezeCurves.freewater(H, θwi, L)
    C = heatcapacity(θw, p)
    T_f = H / C
    T_t = (H - Lθ) / C
    T = ifelse(
        H < zero(θwi),
        # Case 1: H < 0 -> frozen
        T_f,
        # Case 2: H >= 0
        ifelse(
            H >= Lθ,
            # Case 2a: H >= Lθ -> thawed
            T_t,
            # Case 2b: 0 <= H < Lθ -> phase change
            zero(T_f)
        )
    )
end

function heateq_with_phase_change(
    grid::Grid1D,
    T_ub::F=(u,p,t) -> zero(eltype(u))
) where {F}
    function f(u, p, t)
        Hinv(H) = enthalpyinv(H, p)
        Hinv_res = Hinv.(u)
        T = map(first, Hinv_res)
        θw = map(last, Hinv_res)
        θwi = p.θwi
        k = (1-θwi)*p.k_s .+ θw*p.k_w .+ (1.0.-θw)*p.k_i
        jH = k[1:end-1].*(T[1:end-1] .- T[2:end]) ./ grid.dists
        jH_lb = p.jH_lb
        jH_ub = -2*(T[1] - T_ub(u,p,t)) / grid.thick[1]
        dH = vcat(
            (jH_ub - jH[1]) / grid.thick[1],
            (jH[1:end-1] .- jH[2:end]) ./ grid.thick[2:end-1],
            (jH[end] - jH_lb) / grid.thick[end]
        )
        return zero(u) .+ dH
    end
    return f
end

# upper boundary temperature
T_ub(u,p,t) = 10*sin(2π*t/(24*3600))

p = default_p
grid_edges = vcat(0.0:0.05:5.0, 5.1:0.1:20.0, 20.5:0.5:50.0, 51.0:1.0:100.0)
grid = Grid1D(grid_edges)
T0, H0 = steady_state_init(grid, p; Tsurf=-10.0)
f = heateq_with_phase_change(grid, T_ub)
jac_prototype = Tridiagonal(ones(length(H0)-1), ones(length(H0)), ones(length(H0)-1))
tspan = (0.0, 48*3600.0)
prob = ODEProblem(ODEFunction(f; jac_prototype), H0, tspan, p)
sol = @time solve(prob, SSPRK43(), abstol=1e-2, reltol=1e-6)

H_sol = reduce(hcat, sol.u)
T_sol = enthalpyinv.(H_sol, Ref(p))
Plots.plot(T_sol[1:5:50,:]', leg=nothing)

function loss(p)
    sensealg = sensealg=InterpolatingAdjoint(autojacvec=EnzymeVJP(), checkpointing=true)
    sol_p = solve(prob, SSPRK43(); p, sensealg)
    return mean(sol_p.u[end])/L
end

grad = @time Zygote.gradient(loss, p)
