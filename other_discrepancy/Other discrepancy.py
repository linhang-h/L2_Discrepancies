import jax
import jax.numpy as jnp
import optax
from jax import grad, jit
import numpy as np

from jax import config


#Code for the L_2 perio
def perio_L2(L):
    n = L.shape[0]
    diff = L[:, None, :] - L[None, :, :]  
    abs_diff = jnp.abs(diff)
    sq_diff = diff ** 2
    term = 0.5 - abs_diff + sq_diff       
    prod = jnp.prod(term, axis=2)         
    total = jnp.sum(prod) / (n ** 2)
    return jnp.sqrt(total - 1 / 9)

def smooth_perio_L2(X, eps=1e-16):
    n = X.shape[0]

    diff = X[:, None, :] - X[None, :, :]  
    sq_diff = diff ** 2
    abs_approx = jnp.sqrt(sq_diff + eps)  

    term = 0.5 - abs_approx + sq_diff  
    prod = jnp.prod(term, axis=2)      
    total = jnp.sum(prod) / (n ** 2)  

    return jnp.sqrt(total - 1 / 9) 

def general_L2_discrepancy(X):
    """
    Compute real general L2 discrepancy for a set of points X (shape: n x d).
    """
    n, d = X.shape

    # Term 1: (4/3)^d
    term1 = (4 / 3) ** d

    # Term 2: -2/n sum_i prod_j (3/2 - t_ij^2 / 2)
    t_squared = X ** 2
    inner_2 = (3 / 2) - t_squared / 2  # shape (n, d)
    prod_2 = jnp.prod(inner_2, axis=1)  # shape (n,)
    term2 = -2 * jnp.sum(prod_2) / n

    # Term 3: 1/n^2 sum_{i,k} prod_j (2 - max(t_ij, t_kj))
    X_i = X[:, None, :]  # shape (n, 1, d)
    X_k = X[None, :, :]  # shape (1, n, d)
    max_vals = jnp.maximum(X_i, X_k)  # shape (n, n, d)
    prod_3 = jnp.prod(2 - max_vals, axis=2)  # shape (n, n)
    term3 = jnp.sum(prod_3) / (n ** 2)

    return term1 + term2 + term3


def smooth_max_sqrt(a, b, eps=1e-16):
    """
    Smooth approximation of max(a, b) using square root method.
    a, b: JAX arrays of same shape
    eps: small positive value to prevent sqrt(0)
    """
    return 0.5 * (a + b) + 0.5 * jnp.sqrt((a - b)**2 + eps)


def smooth_general_L2_discrepancy(X, eps=1e-16):
    """
    Smoothed general L2 discrepancy for optimization.
    Uses square-root smoothing to approximate max(t_ij, t_kj).

    Parameters:
    - X: (n, d) JAX array
    - eps: small value for smooth max stability

    Returns:
    - scalar discrepancy (smoothed)
    """
    n, d = X.shape

    # Term 1: constant
    term1 = (4 / 3) ** d

    # Term 2: -2/n sum_i prod_j (3/2 - t_ij^2 / 2)
    t_squared = X ** 2
    inner_2 = (3 / 2) - t_squared / 2
    prod_2 = jnp.prod(inner_2, axis=1)
    term2 = -2 * jnp.sum(prod_2) / n

    # Term 3: 1/n^2 sum_{i,k} prod_j (2 - smooth_max(t_ij, t_kj))
    X_i = X[:, None, :]  # shape (n, 1, d)
    X_k = X[None, :, :]  # shape (1, n, d)

    max_vals = smooth_max_sqrt(X_i, X_k, eps=eps)  # shape (n, n, d)
    term_inside = 2.0 - max_vals
    prod_3 = jnp.prod(term_inside, axis=2)
    term3 = jnp.sum(prod_3) / (n ** 2)

    return term1 + term2 + term3


def extreme_L2_discrepancy(X):
    """
    Compute the real extreme L2 discrepancy for a point set X (n x d).
    """
    n, d = X.shape

    # Term 1: 1 / 12^d
    term1 = 1 / (12 ** d)

    # Term 2: -2/n * sum_i prod_j x_ij (1 - x_ij) / 2
    inner_2 = X * (1 - X) / 2  # shape (n, d)
    prod_2 = jnp.prod(inner_2, axis=1)  # shape (n,)
    term2 = -2 * jnp.sum(prod_2) / n

    # Term 3: 1/n^2 * sum_{i,i'} prod_j (min(x_ij, x_i'j) - x_ij * x_i'j)
    X_i = X[:, None, :]  # shape (n, 1, d)
    X_ip = X[None, :, :]  # shape (1, n, d)
    min_vals = jnp.minimum(X_i, X_ip)
    prod_vals = X_i * X_ip
    diff = min_vals - prod_vals
    prod_3 = jnp.prod(diff, axis=2)
    term3 = jnp.sum(prod_3) / (n ** 2)

    return term1 + term2 + term3


def smooth_min_sqrt(a, b, eps=1e-16):
    """
    Smooth approximation of min(a, b) using square root method.
    """
    return 0.5 * (a + b) - 0.5 * jnp.sqrt((a - b) ** 2 + eps)

def smooth_extreme_L2_discrepancy(X, eps=1e-16):
    """
    Smoothed extreme L2 discrepancy for ADAM optimization.
    X: (n, d) point set in [0,1]^d
    eps: small smoothing parameter for sqrt
    """
    n, d = X.shape

    # Term 1: 1 / 12^d
    term1 = 1 / (12 ** d)

    # Term 2: -2/n * sum_i prod_j x_ij (1 - x_ij) / 2
    inner_2 = X * (1 - X) / 2  # shape (n, d)
    prod_2 = jnp.prod(inner_2, axis=1)
    term2 = -2 * jnp.sum(prod_2) / n

    # Term 3: 1/n^2 * sum_{i,i'} prod_j (smooth_min(x_ij, x_i'j) - x_ij * x_i'j)
    X_i = X[:, None, :]   # (n, 1, d)
    X_ip = X[None, :, :]  # (1, n, d)

    smooth_min = smooth_min_sqrt(X_i, X_ip, eps=eps)
    prod_vals = X_i * X_ip
    diff = smooth_min - prod_vals
    prod_3 = jnp.prod(diff, axis=2)
    term3 = jnp.sum(prod_3) / (n ** 2)

    return term1 + term2 + term3

