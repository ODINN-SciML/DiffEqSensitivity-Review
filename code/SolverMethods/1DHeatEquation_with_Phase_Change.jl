using BenchmarkTools
using ComponentArrays
using FreezeCurves
using IfElse
using LinearAlgebra
using OrdinaryDiffEq
using SciMLSensitivity
using Statistics
using Zygote

# plotting
import Plots

const L = 3.34e8 # volumetric latent heat of fusion of water [J/m^3]

const default_p = ComponentVector(
    θwi=0.3,  # total water+ice content; 1.0 would correspond to a block of pure ice/water
    k_s=2.5,  # thermal conductivity of solid material [W/(m.K)]
    k_w=0.57, # thermal conductivity of unfrozen water [W/(m.K)]
    k_i=2.2,  # thermal conductivity of ice [W/(m.K)]
    c_s=2.5e6, # heat capacity of solid material [J/(m^3.K)]
    c_w=4.2e6, # heat capacity of unfrozen water [J/(m^3.K)]
    c_i=1.9e6, # heat capacity of ice [J/(m^3.K)]
    jH_lb=0.1, # lower boundary flux [W/m^2]
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
    T = IfElse.ifelse(
        H < zero(θwi),
        # Case 1: H < 0 -> frozen
        T_f,
        # Case 2: H >= 0
        IfElse.ifelse(
            H >= Lθ,
            # Case 2a: H >= Lθ -> thawed
            T_t,
            # Case 2b: 0 <= H < Lθ -> phase change
            zero(T_f)
        )
    )
    return T, θw
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
        kc = (1-θwi)*p.k_s .+ θw*p.k_w .+ (1.0.-θw)*p.k_i
        k = (kc[1:end-1] .+ kc[2:end])./2
        jH = k.*(T[1:end-1] .- T[2:end]) ./ grid.dists
        jH_lb = p.jH_lb
        jH_ub = 2*(T_ub(u,p,t) - T[1]) / grid.thick[1]
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
T_ub(u,p,t) = 10*sin(2π*t/(24*3600)) - 1.0

p = default_p
grid_edges = vcat(0.0:0.05:5.0, 5.1:0.1:10)
# grid_edges = vcat(0.0:0.05:1.0, 1.1:0.1:10.0)
grid = Grid1D(grid_edges)
T0, H0 = steady_state_init(grid, p; Tsurf=-10.0)
f = heateq_with_phase_change(grid, T_ub)
jac_prototype = Tridiagonal(ones(length(H0)-1), ones(length(H0)), ones(length(H0)-1))
tspan = (0.0, 5*24*3600.0)
prob = ODEProblem(ODEFunction(f; jac_prototype), H0, tspan, p)
sol = @time solve(prob, SSPRK43(), abstol=1e-2, reltol=1e-6, saveat=3600.0)

H_sol = reduce(hcat, sol.u)
T_sol = first.(enthalpyinv.(H_sol, Ref(p)))
Plots.plot(T_sol', leg=nothing)
Plots.heatmap(sol.t./3600.0, grid.cells, T_sol, yflip=true)

buildalg(::Type{InterpolatingAdjoint}; autojacvec, checkpointing, kwargs...) = InterpolatingAdjoint(; autojacvec, checkpointing)
buildalg(::Type{GaussAdjoint}; autojacvec, checkpointing, kwargs...) = GaussAdjoint(; autojacvec, checkpointing)
buildalg(::Type{QuadratureAdjoint}; autojacvec, kwargs...) = QuadratureAdjoint(; autojacvec)
buildalg(::Type{BacksolveAdjoint}; autojacvec, kwargs...) = BacksolveAdjoint(; autojacvec)

function benchmark_sensealg(::Type{algType}, tspan_end, p; saveat=nothing, dealg=SSPRK43(), sensealg_kwargs...) where {algType}
    sensealg = buildalg(algType; sensealg_kwargs...)
    newprob = remake(prob; p=p, tspan=(0.0, tspan_end))
    if isnothing(saveat)
        sol = solve(newprob, dealg)
    else
        sol = solve(newprob, dealg; saveat)
    end
    @assert sol.retcode == ReturnCode.Success
    @assert sol.t[end] == tspan_end
    dgdu(out,u,p,t,i) = (out .= u .- 1.0)
    bench_result = @benchmark adjoint_sensitivities(
        $sol,
        $dealg;
        sensealg=$sensealg,
        t=[$sol.t[end]],
        dgdu_discrete=$dgdu,
        checkpoints=$sol.t,
        abstol=1e-8,
        reltol=1e-8,
    )
    # bench = @benchmark Zygote.gradient($loss, $p)
    return (
        t=tspan_end,
        allocs=bench_result.allocs,
        memory=bench_result.memory,
        runtime_mean=mean(bench_result.times),
        runtime_mid=median(bench_result.times),
        runtime_std=std(bench_result.times),
        alg=string(algType),
        sensealg_kwargs...
    )
end

res_with_chckpointing = benchmark_sensealg(InterpolatingAdjoint, 24*3600.0, p; autojacvec=EnzymeVJP(), checkpointing=true, saveat=600.0)
res_without_checkpointing = benchmark_sensealg(InterpolatingAdjoint, 24*3600.0, p; autojacvec=EnzymeVJP(), checkpointing=false)

configs = [
    (InterpolatingAdjoint, (autojacvec=EnzymeVJP(), checkpointing=false)),
    (InterpolatingAdjoint, (autojacvec=EnzymeVJP(), checkpointing=true)),
    (GaussAdjoint, (autojacvec=EnzymeVJP(), checkpointing=false)),
    (GaussAdjoint, (autojacvec=EnzymeVJP(), checkpointing=true)),
    (QuadratureAdjoint, (autojacvec=EnzymeVJP(), checkpointing=false)),
    (BacksolveAdjoint, (autojacvec=EnzymeVJP(), checkpointing=false)),
]

# target simulation time periods ranging from 1 minute to 30 days
tspans = [60.0, 3600.0, 24*3600.0, 30*24*3600.0]

# lossfunc = buildloss(prob, InterpolatingAdjoint(autojacvec=EnzymeVJP()), tspan=(0.0,24*3600.0))
# Zygote.gradient(lossfunc, p)

results = []
for t in tspans
    for c in configs
        algtype, kwargs = c
        @info "Running benchmark for $algtype with $kwargs and tspan of $t sec."
        saveat = min(t/10.0, 3600.0)
        push!(results, benchmark_sensealg(algtype, t, p; saveat, kwargs...))
    end
end

using DataFrames
results_df = DataFrame(results)
