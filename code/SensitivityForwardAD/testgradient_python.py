import jax
import jax.numpy as jnp
from diffrax import diffeqsolve, ODETerm, Tsit5, 
                    PIDController, BacksolveAdjoint, 
                    RecursiveCheckpointAdjoint, DirectAdjoint

from jax import config
config.update("jax_enable_x64", True)

def vector_field(t, u, args):
    x, y = u
    a, b = args
    dx = a * (x - y)
    dy = b * (y - x)
    return dx, dy

def run(y0, adjoint = RecursiveCheckpointAdjoint(), tol = 1e-12):
    term = ODETerm(vector_field)
    solver = Tsit5(scan_kind="bounded")
    stepsize_controller = PIDController(rtol=tol, atol=tol)
    t0 = 0
    t1 = 5.0
    dt0 = 0.1
    p0 = (1.0, 1.0)
    sol = diffeqsolve(term, solver, t0, t1, dt0, y0, 
                      adjoint = adjoint,
                      stepsize_controller = stepsize_controller,
                      args=p0)
    ((x,), _) = sol.ys
    return x

y0 = (jnp.array(1.0), jnp.array(1.0))
J = jax.jacrev(run)(y0)
# (Array(3755.79674193, dtype=float64, weak_type=True), 
# Array(-3754.79674193, dtype=float64, weak_type=True))

J = jax.jacrev(lambda y0: run(y0, RecursiveCheckpointAdjoint(), tol = 1e-3))(y0)
# (Array(3755.79674193, dtype=float64, weak_type=True)
# Array(-3754.79674193, dtype=float64, weak_type=True))

J = jax.jacfwd(lambda y0: run(y0, DirectAdjoint()))(y0)
# (Array(3755.79674193, dtype=float64), Array(-3754.79674193, dtype=float64))

J = jax.jacfwd(lambda y0: run(y0, DirectAdjoint(), tol = 1e-3))(y0)
# (Array(3755.79674193, dtype=float64), Array(-3754.79674193, dtype=float64))






J = jax.jacrev(lambda y0: run(y0, BacksolveAdjoint()))(y0)
# (Array(11013.73289742, dtype=float64, weak_type=True), 
# Array(-11012.73289742, dtype=float64, weak_type=True))

J = jax.jacrev(lambda y0: run(y0, BacksolveAdjoint(), tol = 1e-3))(y0)
# (Array(10869.87401012, dtype=float64, weak_type=True), 
# Array(-10868.87401012, dtype=float64, weak_type=True))

y1 = (jnp.array(1.000001), jnp.array(1.0))
y0 = (jnp.array(1.0), jnp.array(1.0))
(run(y1) - run(y0)) / .000001
# Array(11013.73156938, dtype=float64)