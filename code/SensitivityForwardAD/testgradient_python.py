import jax
import jax.numpy as jnp
from diffrax import diffeqsolve, ODETerm, Tsit5, SaveAt, PIDController, RecursiveCheckpointAdjoint, DirectAdjoint

from jax import config
config.update("jax_enable_x64", True)

def vector_field(t, u, args):
    x, y = u
    a = args
    dx = a * x - x * y
    dy = -a * y + x * y
    return dx, dy

def run(p0, adjoint = RecursiveCheckpointAdjoint(), tol = 1e-12):
    term = ODETerm(vector_field)
    solver = Tsit5(scan_kind="bounded")
    stepsize_controller = PIDController(rtol=tol, atol=tol)
    t0 = 0
    t1 = 10.0
    ts = jnp.linspace(t0, t1, 101)
    dt0 = 0.1
    y0 = (jnp.array(1.0), jnp.array(1.0))
    saveat = SaveAt(ts=ts)

    sol = diffeqsolve(term, solver, t0, t1, dt0, y0, 
                      adjoint = adjoint,
                      stepsize_controller = stepsize_controller,
                      args=p0,
                      saveat=saveat)
    return sum(sum(sol.ys))

p0 = 1.0
run(p0, RecursiveCheckpointAdjoint())
    
J = jax.jacrev(run)(y0)
# Array(6217.6080555, dtype=float64, weak_type=True)

J = jax.jacrev(lambda y0: run(y0, RecursiveCheckpointAdjoint(), tol = 1e-3))(y0)
# Array(6217.6080555, dtype=float64, weak_type=True)

J = jax.jacfwd(lambda y0: run(y0, DirectAdjoint()))(y0)
# Array(6217.6080555, dtype=float64)

J = jax.jacfwd(lambda y0: run(y0, DirectAdjoint(), tol = 1e-3))(y0)
# Array(6217.6080555, dtype=float64)

y1 = 1.000001
y0 = 1.0
(run(y1) - run(y0)) / .000001
# Array(212.71060729, dtype=float64)