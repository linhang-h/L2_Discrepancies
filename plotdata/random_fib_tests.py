import jax
import jax.numpy as jnp
import optax
from jax import grad, jit
import numpy as np
import random

import matplotlib.pyplot as plt

def heatmap_from_array(P, ax=None):  
    n = len(P)
    PX = P[:, 0]
    PY = P[:, 1]

    X = [0.001 * i for i in range(1001)]
    Y = [0.001 * i for i in range(1001)]
    Z = [[0 for _ in range(1001)] for _ in range(1001)]

    maxi = 0
    mxii = 0
    mxjj = 0
    for i in range(1001):
        for j in range(1001):
            Z[j][i] = locdisc(X[i], Y[j], P)
            if Z[j][i] > maxi:
                maxi = Z[j][i]
                mxii = i
                mxjj = j

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))

    cs = ax.contourf(X, Y, Z, levels=40, cmap="viridis")
    plt.colorbar(cs, ax=ax, label="Local Discrepancy")
    ax.scatter(PX, PY, c='r', s=50, linewidth=1, alpha=0.7)
    ax.scatter(mxii/1000, mxjj/1000, c='black')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect('equal')

    return ax, maxi

def is_in(p,q):
    for i in range(len(p)):
        if p[i]>q[i]:
            return False
    return True

def is_instrict(p,q):
    for i in range(len(p)):
        if p[i]>=q[i]:
            return False
    return True

def discre(L):
    disc=0
    
    vol=0
    L.append([1.0,1.0])
    f=len(L)
    for i in range(f):
        for j in range(f):
            count=0
            count2=0
            vol=L[i][0]*L[j][1]
            for k in range (f-1):
                if is_in(L[k],[L[i][0],L[j][1]]):
                    count+=1
                if is_instrict(L[k],[L[i][0],L[j][1]]):
                    count2+=1
            temp=count/float(f-1)-vol
            temp2=vol-count2/float(f-1)
            if temp> temp2 and temp> disc:
                disc=temp
            else:
                if temp2> disc and temp2> temp:
                    disc=temp2
    L.remove([1.0,1.0])
    return disc

def Linf_discrepancy(P, grid_size=300):
    # Evaluate discrepancy on a dense grid and take the max value
    xs = np.linspace(0, 1, grid_size)
    ys = np.linspace(0, 1, grid_size)
    X, Y = np.meshgrid(xs, ys)
    points_grid = np.stack([X.ravel(), Y.ravel()], axis=1)

    # For each grid point x, count proportion of P in [0,x1]x[0,x2]
    count = np.sum((P[:, None, 0] <= points_grid[:, 0]) &
                   (P[:, None, 1] <= points_grid[:, 1]), axis=0)
    N = len(P)
    discrepancy = np.abs(count / N - points_grid[:, 0] * points_grid[:, 1])
    return np.max(discrepancy)

def locdisc(x, y, P):
    no = 0
    nc = 0
    n = len(P)
    for i in range(n):
        if P[i][0] < x and P[i][1] < y:
            no += 1
        if P[i][0] <= x and P[i][1] <= y:
            nc += 1
    return max(x*y - no/n, nc/n - x*y)

def plot_points(P, title="Point Set"):
    P_np = jnp.array(P) if isinstance(P, jax.Array) else P
    plt.figure(figsize=(4, 4))
    plt.scatter(P_np[:, 0], P_np[:, 1], c='blue', s=30)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title(title)
    plt.gca().set_aspect('equal')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

def softmax(a, b, tau=1e-16):
    return 0.5 * (a + b + jnp.sqrt((a - b)**2 + tau))

# Smoothed L2 star discrepancy function
def L2_smoothed(P, tau=1e-16):
    n = P.shape[0]
    a = jnp.sum((1 - P[:, 0]**2) * (1 - P[:, 1]**2)) / (2 * n)

    # Compute pairwise (i, j) terms using broadcasting
    x = P[:, 0]
    y = P[:, 1]
    xi, xj = jnp.meshgrid(x, x, indexing='ij')
    yi, yj = jnp.meshgrid(y, y, indexing='ij')

    smax0 = softmax(xi, xj, tau)
    smax1 = softmax(yi, yj, tau)
    b = jnp.sum((1 - smax0) * (1 - smax1)) / (n**2)

    return 1/9 - a + b

golden_ratio = (jnp.sqrt(5) - 1) / 2

def fibonacci(n, gen = golden_ratio, shift = 0):
    i = jnp.arange(n)
    x = (i+shift) / n
    y = ((i+shift) * gen) % 1  
    return jnp.stack([x, y], axis=1)

def L2_discrepancy(P):
    N, d = P.shape

    prod1 = 1. - P ** 2
    prod1 = np.prod(prod1, axis=1)
    sum1 = np.sum(prod1)

    # Broadcasting to compute pairwise max across dimensions
    P1 = P[:, np.newaxis, :]  # shape (N, 1, d)
    P2 = P[np.newaxis, :, :]  # shape (1, N, d)
    pairwise_max = np.maximum(P1, P2)  # shape (N, N, d)

    product = np.prod(1. - pairwise_max, axis=2)
    sum2 = np.sum(product)

    one_div_N = 1. / N
    out = np.sqrt(
        np.power(3., -d)
        - one_div_N * np.power(2., 1. - d) * sum1
        + (1. / np.power(N, 2.)) * sum2
    )

    return out

output_vals = np.zeros([2, 500])

for w in range(500):
    # Use it for initial points
    n = 260
    P_init = fibonacci(n, gen = random.random())

    # ADAM optimizer setup
    learning_rate = 0.001
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(P_init)

    # Define loss function and gradient
    loss_fn = lambda P: L2_smoothed(P)
    loss_and_grad = jax.value_and_grad(loss_fn)

    # Optimization loop
    P = P_init
    objective_history = []

    L2_values = []
    Linf_values = []

    for i in range(200):
        loss_val, grads = loss_and_grad(P)
        updates, opt_state = optimizer.update(grads, opt_state)
        P = optax.apply_updates(P, updates)

        # Clamp to [0, 1]^2
        P = jnp.clip(P, 0.0, 1.0)
        objective_history.append(loss_val)
        
        L2_values.append(float(L2_discrepancy(np.array(P))))
        Linf_values.append(float(Linf_discrepancy(np.array(P))))

    init_points = np.asarray(P_init).astype(np.float64)
    optimized_points = np.asarray(P).astype(np.float64)
    initial_L2_value = L2_discrepancy(init_points)
    initial_Linf_value = discre(init_points.tolist())

    # === Output ===
    output_vals[0, w] = L2_discrepancy(optimized_points)
    output_vals[1, w] = discre(optimized_points.tolist())
    
np.savetxt('random_fib_results.txt', output_vals)