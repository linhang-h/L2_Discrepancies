import jax
import jax.numpy as jnp
import optax
from jax import grad, jit
import numpy as np
from qmcpy import Halton

def getKronecker(n, phi, placehold = 0, startwith = 0):
    '''
    Generates Kronecker lattice.

    #Args

    n : number of points to sample
    Phi : d - 1 dimensional real vector whose coordinates are linearly independent over rationals
    placehold : the coordinate of lattice that will be linearly increasing.
    startwith : the starting index to be scaled by Phi.

    #returns
    A Kronecker lattice.
    '''
    if jnp.isscalar(phi):
        phi = jnp.array([phi])
        
    d = phi.shape[0]+1
    i_values = jnp.arange(startwith, n + startwith).reshape(n, 1)
    scaled_phi = (i_values * phi) % 1
    points = jnp.zeros((n, d))
    idx = list(range(d))
    idx.pop(placehold)
    points = points.at[:, idx].set(scaled_phi)
    points = points.at[:, placehold].set(jnp.arange(startwith, n + startwith) / n)

    return points


# Fibonacci initilization in any dim
def getFibonacci(n, d, placehold=0):
    phi_base = (jnp.sqrt(5) - 1) / 2
    # Use inverse powers of the golden ratio to ensure irrationality and avoid 0
    phis = jnp.array([1 / phi_base**j for j in range(1, d)])  # d-1 values
    return getKronecker(n, phi=phis, placehold=placehold)

# determintic sobol in any dim
def sobol_nd_deterministic(n, d=5):
    m = int(np.ceil(np.log2(n)))  # Number of points must be a power of 2
    sampler = qmc.Sobol(d=d, scramble=False)  # scramble=False → fully deterministic
    points = sampler.random_base2(m=m)
    return points[:n]

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


