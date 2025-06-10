import jax
import jax.numpy as jnp
import optax
from jax import grad, jit
import numpy as np
from qmcpy import Halton

# determinitic Halton (not scrambled) in any dim
def getHalton(n, d):
    HaltonGenerator = Halton(dimension=d, randomize=None)
    return HaltonGenerator.gen_samples(n)

# random in any dim
def random_nd_points(n, d=5, seed=None):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 1.0, size=(n, d))

# Fibonaccci lattice in dim 2
def fibonacci_rational_lattice(Fn, Fn_minus1):
    i = jnp.arange(Fn, dtype=jnp.float64)
    x = i / Fn
    y = (Fn_minus1 * i) / Fn  # float multiplication and division
    y = y % 1.0               # fractional part in float
    return jnp.stack([x, y], axis=1)

