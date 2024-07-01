using OrdinaryDiffEq, ForwardDiff, Zygote, SciMLSensitivity

function f!(du, u, p, t)
    du[1] = p[1] * (u[1] - u[2])
    du[2] = p[2] * (u[2] - u[1])
end
p = [1.0, 1.0];
u0 = [1.0; 1.0];
prob = ODEProblem(f!, u0, (0.0, 5.0), p)
sol = solve(prob, Tsit5(), saveat = 0.1)

function cost(u0)
    _prob = remake(prob, u0 = u0)
    solve(_prob, Tsit5(), reltol = 1e-12, abstol = 1e-12, saveat = 0.1)[1, :]
end
dx = ForwardDiff.jacobian(cost, u0)[end, :]
#=
2-element Vector{Float64}:
  11013.732897407262
 -11012.732897407262
=#
dx2 = Zygote.jacobian(cost, u0)[1][end, :]
#=
2-element Vector{Float64}:
  11013.732897407262
 -11012.732897407262
=#

function cost(u0)
    _prob = remake(prob, u0 = u0)
    solve(_prob, Tsit5(), reltol = 1e-6, abstol = 1e-6, saveat = 0.1)[1, :]
end
dx = ForwardDiff.jacobian(cost, u0)[end, :]
#=
2-element Vector{Float64}:
  11013.73051284873
 -11012.73051284873
=#

dx2 = Zygote.jacobian(cost, u0)[1][end, :]
#=
2-element Vector{Float64}:
  11013.73051284873
 -11012.73051284873
=#

function cost(u0)
    _prob = remake(prob, u0 = u0)
    solve(_prob, Tsit5(), reltol = 1e-12, abstol = 1e-12, saveat = 0.1)[1, :]
end
p = [1.0001, 1.0];
((cost(p) - cost([1.0, 1.0])) / 0.0001)[end]
# 11013.732900492363
