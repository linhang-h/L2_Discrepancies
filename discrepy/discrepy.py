#dependency
import time
import qmcpy
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import jax.lax as lax
from jax import grad, jit
import optax
from qmcpy import Halton
from qmcpy import Lattice
from qmcpy import Sobol
import itertools

# General Utilities------------------------------------

def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()  # More precise than time.time()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(f"Function '{func.__name__}' executed in {end_time - start_time:.6f} seconds")
        return result
    return wrapper


# Visual Utilities------------------------------------

def plot2DArrow(init, end, Ax = None, paint = 'blue'):
    if Ax is None:
        Ax = plt.gca()
    dx = end[:,0] - init[:,0]
    dy = end[:,1] - init[:,1]
    Ax.quiver(init[:,0], init[:,1], dx, dy, angles='xy', scale_units='xy', scale = 1, color = paint , width=0.003)
    Ax.legend()
    plt.show()
    return Ax

def locdisc(x, y, points):
    no = 0
    nc = 0
    n = len(points)
    for i in range(n):
        if points[i][0] < x and points[i][1] < y:
            no += 1
        if points[i][0] <= x and points[i][1] <= y:
            nc += 1
    return max(x*y - no/n, nc/n - x*y)


def heatmap_from_array(points, Ax=None):
    n = len(points)
    PX = points[:, 0]
    PY = points[:, 1]

    X = [0.001 * i for i in range(1001)]
    Y = [0.001 * i for i in range(1001)]
    Z = [[0 for _ in range(1001)] for _ in range(1001)]

    maxi = 0
    mxii = 0
    mxjj = 0
    for i in range(1001):
        for j in range(1001):
            Z[j][i] = locdisc(X[i], Y[j], points)
            if Z[j][i] > maxi:
                maxi = Z[j][i]
                mxii = i
                mxjj = j

    if Ax is None:
        Fig, Ax = plt.subplots(figsize=(7, 6))

    cs = Ax.contourf(X, Y, Z, levels=40, cmap= "viridis")
    plt.colorbar(cs, ax = Ax, label= "Local Discrepancy")
    Ax.scatter(PX, PY, c='r', s=50, linewidth=1, alpha=0.7)
    Ax.scatter(mxii/1000, mxjj/1000, c='black')
    Ax.set_xlabel("x")
    Ax.set_ylabel("y")
    Ax.set_aspect('equal')

    return Ax, maxi


def draw_history(hist, title = "History", label = "L2", xlabel = "Iteration", ylabel = "Discrepancy", Ax = None):
    if Ax is None:
        Fig = plt.figure(figsize=(8, 5))
        Ax = Fig.gca()
    
    Ax.plot(hist, label= label, color = 'blue')
    Ax.set_xlabel(xlabel)
    Ax.set_ylabel(ylabel)
    Ax.set_title(title)
    Ax.legend()
    Ax.grid(True)
    plt.tight_layout()
    plt.show()
    
    return Ax


def draw_the_plot(initial, optimized, 
                  title = "Summary", label1 = "Initial", label2 = "Optimized", arrow = True, Ax = None):
        
    if Ax is None:
        Fig = plt.figure(figsize=(6, 6))
        Ax = Fig.gca()
        
    ip = initial
    fp = optimized
    Ax.scatter(ip[:, 0], ip[:, 1], c ='red', label=label1)
    Ax.scatter(fp[:, 0], fp[:, 1], c ='green', label=label2)

    if arrow: plot2DArrow(ip, fp, Ax = Ax, paint = 'blue')
    Ax.set_title(title)
    Ax.set_aspect('equal')

    Ax.set_xlim(0, 1)
    Ax.set_ylim(0, 1)
    Ax.grid(True)
    Ax.legend()
    plt.tight_layout()
    plt.show()
        
    return Ax


def draw_heatmap(points, title = "Heatmap", Ax = None):
    if Ax is None:
        Fig = plt.figure(figsize = (7, 5))
        Ax = Fig.gca()

    # Heatmap for initial set
    _, max_init = heatmap_from_array(points, Ax=Ax)
    Ax.set_title(title)
    Ax.text(0.5, -0.15, f"Max Local Discrepancy: {max_init:.5f}",
        transform= Ax.transAxes, ha='center', fontsize=11)        
    plt.tight_layout()
    plt.show()
    return Ax

# Functional Utilities------------------------------------

def smooth_max(a, b, tau=1e-16):
    '''
    Smoothed maximum function. Control tau to adjust smoothness.
    '''
    return 0.5 * (a + b + ((a - b)**2 + tau)**(0.5))

def smooth_min_sqrt(a, b, tau=1e-16):
    """
    Smooth approximation of min(a, b) using square root method.
    """
    return 0.5 * (a + b) - 0.5 * jnp.sqrt((a - b)**2 + tau)

def internalize(points):
    '''
    #Args
    point set,
    
    #returns

    the sorted version of each coordinate, (sortedCoordinates)
    and the order indices in which coordinates of each point appears in the sortedCoordinates, (ordidx)
    '''
    d = points.shape[1]
    n = points.shape[0]
    sortedCoordinates = np.apply_along_axis(sorted, axis = 0, arr = points).T
    sortedIndices = np.argsort(points, axis = 0)
    ordidx = np.empty_like(sortedIndices)
    for j in range(d):
        ordidx[sortedIndices[:, j], j] = np.arange(n)     

    return sortedCoordinates, ordidx


def L2Disc(points, smoothing = False, tau = 1e-16):
    
    '''
    L2 Discrepancy, following the Wornock's formula. Written in Numpy setting.
    '''
    n = points.shape[0]
    d = points.shape[1]

    SP1 = jnp.sum(jnp.prod(1 - points**2, axis=1))
    
    x_i = points[:, jnp.newaxis, :]  # shape: (n, 1, d)
    x_j = points[jnp.newaxis, :, :]  # shape: (1, n, d)


    vf_values = lax.cond(
    smoothing,
    lambda _: smooth_max(x_i, x_j, tau=tau),
    lambda _: jnp.maximum(x_i, x_j),
    operand=None
    )

    SP2 = jnp.sum(jnp.prod(1 - vf_values, axis=2))

    return 1/(3**d) - 2**(1-d)/n * SP1 + 1/(n**2) * SP2


def LinfDisc(points, sizelim = 3*300**3, mg_size = 300, method = 'internalize'):
    '''
    L - Inf Discrepancy by definition.
    If using GPU, jax will allcate up to 75% of the memory. Refer to jax document: jax GPU memory allocation.
    Running on CPU, it is dynamic, and depends on RAM using. This is unstable if computational demand is heavy.
    
    method : 'internalize' gives exact L_infinity discrepancy. 'meshgrid' gives approximate L_infinity discrepancy,
    by computing local discrepancies over a suitably large meshgrid. (controllable through mg_size)

    **If the size of n**d is not exceedingly large, 'internalize' is superior in both speed and accuracy.
    However, if n**d is large, one should turn to meshgrid to avoid memory fault (+ for speed) 
    
    sizelim : set limit of the total estimated memory demand. If d * (n**d) exceeds sizelim, we force to switch into meshgrid method.
    Through this argument one can control when to give up internalize method, and manage to avoid memory fault.
    
    mg_size : when using meshgrid, the size of grid. (it creates mg_size**d large grid)

    '''
    n = points.shape[0]
    d = points.shape[1]

    size_est = d * n**d
    if size_est > sizelim:
        method = 'meshgrid'      

    if method == 'internalize':
        #sortedCoordinate, ordidx = funcUtils.internalize(points)
        points = jnp.vstack((points, jnp.ones(d)))

        # Generate grid indices using broadcasting
        mesh = jnp.meshgrid(*points.T, indexing='ij')
        gridix = jnp.stack(mesh, axis=-1).reshape(-1, d)  # Flatten

        # Compute volume
        vol = jnp.prod(gridix, axis=1)

        # Vectorized counts
        opencount = jnp.sum(jnp.all(points < gridix[:, None, :], axis=2), axis=1)
        closedcount = jnp.sum(jnp.all(points <= gridix[:, None, :], axis=2), axis=1)

        # Compute max discrepancy
        disc = jnp.max(jnp.maximum(closedcount / n - vol, vol - opencount / n))
        
        return disc
        
    #Meshgrid <- recommended for relatively large sized point set. Vectorized operations.
    elif method == 'meshgrid':
       dims = [jnp.linspace(0, 1, mg_size) for _ in range(d)]
       grid = jnp.meshgrid(*dims, indexing='xy')
       reshape_grid = jnp.array(grid).reshape(len(grid), -1)
       eval_set = jnp.stack(reshape_grid, axis = 1)
       #print(eval_set.shape)
       comparison = points[jnp.newaxis, :, :] < eval_set[:, jnp.newaxis, :]
       count = jnp.sum(jnp.all(comparison, axis=2), axis=1)
       disc = jnp.max(jnp.abs(count/n - jnp.prod(eval_set, axis = 1)))
       return disc


 #Code for the L_2 perio
def PeriodicL2Disc(points, smoothing = False, tau = 1e-16):
    '''
    Periodic L2 Discrepancy, following the Wornock's formula. Written in Numpy setting. Control tau to adjust smoothness.
    '''
    n = points.shape[0]
    diff = points[:, None, :] - points[None, :, :]
    sq_diff = diff**2

    abs_diff = lax.cond(
    smoothing,
    lambda _: jnp.sqrt(sq_diff + tau),
    lambda _: jnp.abs(diff),
    operand=None
    )
    term = 0.5 - abs_diff + sq_diff
    prod = jnp.prod(term, axis=2)
    total = jnp.sum(prod) / (n**2)

    return total - 1/9



def ExtremeL2Disc(points, smoothing = False, tau = 1e-16):
    """
    Extreme L2 discrepancy. Control tau to adjust smoothness.
    """
    n, d = points.shape

    # Term 1: 1 / 12^d
    term1 = 1 / (12**d)

    # Term 2: -2/n * sum_i prod_j x_ij (1 - x_ij) / 2
    inner_2 = points * (1 - points) / 2  # shape (n, d)
    prod_2 = jnp.prod(inner_2, axis = 1)  # shape (n,)
    term2 = -2 * jnp.sum(prod_2) / n

    # Term 3: 1/n^2 * sum{i,i'} prod_j (min(x_ij, x_i'j) - x_ij * x_i'j)
    points_i = points[:, None, :]  # shape (n, 1, d)
    points_ip = points[None, :, :]  # shape (1, n, d)

    min_vals = lax.cond(
    smoothing,
    lambda _: smooth_min_sqrt(points_i, points_ip, tau = tau),
    lambda _: jnp.minimum(points_i, points_ip),
    operand=None
    )
    prod_vals = points_i * points_ip
    diff = min_vals - prod_vals
    prod_3 = jnp.prod(diff, axis=2)
    term3 = jnp.sum(prod_3) / (n ** 2)

    return term1 + term2 + term3


def smooth_extreme_L2_discrepancy(X, eps=1e-16):

    n, d = X.shape

    # Term 1: 1 / 12^d
    term1 = 1 / (12**d)

    # Term 2: -2/n * sum_i prod_j x_ij (1 - x_ij) / 2
    inner_2 = X * (1 - X) / 2  # shape (n, d)
    prod_2 = jnp.prod(inner_2, axis=1)
    term2 = -2 * jnp.sum(prod_2) / n

    # Term 3: 1/n^2 * sum{i,i'} prod_j (smooth_min(x_ij, x_i'j) - x_ij * x_i'j)
    X_i = X[:, None, :]   # (n, 1, d)
    X_ip = X[None, :, :]  # (1, n, d)

    smooth_min = smooth_min_sqrt(X_i, X_ip, eps=eps)
    prod_vals = X_i * X_ip
    diff = smooth_min - prod_vals
    prod_3 = jnp.prod(diff, axis=2)
    term3 = jnp.sum(prod_3) / (n ** 2)

    return term1 + term2 + term3




def ExtremeL2Disc(points, smoothing = False):
    '''
    Extreme L2 Discrepancy, following the Wornock's formula. Written in Numpy setting.
    '''
    pass

def unitCubeProj(points):
    return jnp.clip(points, 0.0, 1.0)


L2 = lambda points: L2Disc(points, smoothing = False)
L2_Smooth = lambda points: L2Disc(points, smoothing = True)
PeriodicL2 = lambda points: PeriodicL2Disc(points, smoothing = False)
PeriodicL2_Smooth = lambda points: PeriodicL2Disc(points, smoothing = True)
ExtremeL2 = lambda points: ExtremeL2Disc(points, smoothing = False)
ExtremeL2_Smooth = lambda points: ExtremeL2Disc(points, smoothing = True)

Linf = lambda points : LinfDisc(points, sizelim = 300**2, mg_size = 300, method = 'internalize')


# Sequence Generators ------------------------------------

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

    
def getFibonacci(n, d = 2, placehold = 0):
    '''
    Generates first n terms of the fibonacci sequence. Currently 2 - dimensional only
    '''
    phi_1 = (jnp.sqrt(5) - 1)/2
    phi = jnp.array([ (phi_1**j - 1)/(phi_1 -1) for j in range(2,d+1)])
    return getKronecker(n, phi = phi, placehold = placehold, startwith = 0)


def getHalton(n, d = 2):
    HaltonGenerator = Halton(dimension=d)
    return HaltonGenerator.gen_samples(n)


def getRandom(n, d = 2):
    return np.random.rand(n,d)
    

def getSobol(n, d, randomnize = 'LMS_DS', seed = None):
    '''
    Works properly only when n is a power of 2.
    '''
    Sob = Sobol(dimension=d, randomize = randomnize, seed = seed)
    return Sob.gen_samples(n)

def getLattice(n,d, randomnize = 'shift', order = 'natural', seed = None):
    '''
    Works properly only when n is a power of 2.
    '''
    l = Lattice(dimension=d, randomize = randomnize, order = order, seed = seed)
    return l.gen_samples(n)


def getMPMC(n, d = 2):
    pass

def default_compute_additionals(**kwargs):
    return {}

class OptTester:
    '''
    A Wrapper class for optax optimizer that is *iterative, and *use gradient.
    This class helps running various optimizers, options, and experiments for optimization task of discrepancies,
    easily for who are aware of basics of optax, without need of duplication of codes.
    Design of this class sticks to functional programming paradigm.
    '''
    linf = Linf
    l2 = L2

    
    def __init__(self, optimizer, name = "",
                 compute_additionals = default_compute_additionals, dec = lambda x: unitCubeProj(x), enc = lambda x : x):
        '''
        Initializer of class OptTester.
        #Args

        optimizer : optax optimizer object. 
        name : string.

        compute_additionals : 
        A function returning dictionary of keyword arguments and value pair,
        out of gradient, value, current parameter state, and loss function, plus some extra arguments (if needed)
        The signature of compute_additionals should be formatted like the beow:

        def compute_additionals(grad = None, val = None, state = None, lossfn = None, **args):
            ... ...
            return {'karg1' : v1, 'karg2' : v2, ...}
        
        where 'karg1','karg2',... are keyword arguments needed for self.optimizer.update()
        

        enc, dec: A function taking a point set and returning a point set. Performed before and after each step of iterative optimization.
        For example, dec is by default an action of cliping points in [0,1].  
        '''
        self.optimizer = optimizer
        self.name = name
        self._compute_additionals = compute_additionals
        self._dec = dec
        self._enc = enc


    def step(self, state, lossfn, opt_state, extra = None):
        '''
        Perform one step of iterative optimization.

        #Args
        state : parameters to optimize. lossfn : jax - differentiable loss function to optimize. 
        opt_state : optimizer state. 
        extra : extra arguments needed for self.optimizer.update(), other than gradient, value, parameters and loss function.

        #returns
        A tuple of updated state(parameters) and an updated optimizer state.
        '''
        if opt_state is None:
            raise Exception("Not properly initialized")
        
        if extra is None:
            extra = {}
        
        state = self._enc(state)
        
        loss_and_grad = jax.value_and_grad(lossfn)
        lfvalue, lfgrad = loss_and_grad(state)

        additionals = self._compute_additionals(**{'grad':lfgrad, 'val':lfvalue, 'state':state, 'lossfn':lossfn, **extra})

        updates, opt_state = self.optimizer.update(lfgrad, opt_state, params = state, **additionals)
        state = optax.apply_updates(state, updates)
        
        state = self._dec(state)
    
        return state, opt_state


    @property
    def dec(self): return self._dec
    @property
    def enc(self): return self._enc
    @property
    def compute_additionals(self): return self.compute_additionals
    @dec.setter
    def dec(self, dec1): self._dec = dec1
    @enc.setter
    def enc(self, enc1): self._enc = enc1
    @compute_additionals.setter
    def compute_additionals(self, func): self._compute_additionals = func
    @classmethod
    def set_LInf(cls, linf):
        cls.Linf = linf
    
    @classmethod
    def set_L2(cls, l2):
        cls.L2 = l2



    @timeit
    def iterate(self, init_data, lossfn, maxiter = 200, tol = 1e-6,
            linf_record_step = 1, save_history = False, L2 = l2, Summary = True):
        '''
        Perform whole iterative optimization.
        #Args

        init_data : points to optimize. 
        lossfn : jax - differentiable loss function to optimize.
        maxiter : maximum number of iteration.
        tol : tolerance to be used for break condition.

        linf_record_step : integer referring how frequently evalute L_infinity discrepancy. For example, if 2, it evaluates per every 2 steps.
        Helpful for fast execution because L_infinity discrepancy is computationally heavy, especially for higher dimensions.

        save_history : bool flag to save histories or not. Helpful for fast execution as well.

        L2 : A reference function to record history. This is needed because L2 discrepancy is nonsmooth.
        As well as lossfn, one can give different versions (Periodic, Extreme) discrepancies.

        Summary : A bool flag. If True, displays decadance of loss function, and print out the summary of optimization.

        #returns
        tuple of optimized points, history of L2, history of L_infinity discrepancy. histories are meaningful only if save_history = True.
        '''
        #initialization
        inum = 0
        opt_state = self.optimizer.init(init_data)
        state = init_data
        
        #initialization of result lists
        if save_history:
            LInf_history = [OptTester.linf(state)]
        else:
            LInf_history = []
        Loss_history = [lossfn(state)]
        history = []

        if L2 is None: history = [lossfn(state)] 
        else: history =  [L2(state)]


        while inum < maxiter:
            before_step = state
            #functional control of loss func
            state, opt_state = self.step(state, lossfn, opt_state)
            Loss_history.append(lossfn(state))

            if save_history:
                if L2 is None: history.append(lossfn(state))
                else: history.append(L2(state))
                # Since LInf is computationally heavy, one can compute it less frequently
                if inum % linf_record_step == 0: LInf_history.append(OptTester.linf(state))
                  
            inum +=1

            #stop condition if tolerance gap is achieved
            if jnp.linalg.norm(state - before_step) < tol: break
    
        if save_history: LInf_history.append(OptTester.linf(state))

        #summary
        if Summary:
            draw_history(Loss_history, title = self.name, label = "lf", xlabel = "Iteration", ylabel = "Loss", Ax = None)
            # === Output ===
            print("Initial L2 discrepancy:", L2(init_data))
            print("Final L2 discrepancy value:", L2(state))
            print("Optimized Points:\n", state)

        return state, history, LInf_history




# Functions -----------------------------------------------d


def visualize_all(init, optimized, *histories):
    '''
    Perform all the relevant plots. It is combination of draw_the_plot, draw_heatmap, draw_history.
    '''
    Ax1 = draw_the_plot(init, optimized, arrow = True)
    Ax2 = draw_heatmap(optimized)
    Axs = []
    i = 1
    for hist in histories:
        Axs.append(draw_history(hist, label = "history #"+str(i)))
        i +=1

    return Ax1, Ax2, tuple(Axs)


def random_restart(opttester, init_data, lossfn, scale = 0.2, restart_num = 10):
    '''
    Perform iterations of given optimizer, by random restarting #restart_num times.

    #Args
    
    opttester : OptTester.
    init_data : points to optimize.
    lossfn : jax - differentiable loss function to optimizer. 
    scale : magnitude of added random noise. 
    restart_num : number of repetition.
    
    #returns
    final optimized points, total history of L2 discrepancy, total history of L_inf discrepancy
    '''
    data = init_data
    concat_L2hist = []
    concat_Linfhist = []
    key = jax.random.key(42)

    for i in range(restart_num):
        key, subkey = jax.random.split(key)
        data, L2hist, Linfhist, = opttester.iterate(data, lossfn, linf_record_step = 15, save_history = True, Summary = False)
        noise = scale * (jax.random.uniform(key, shape=data.shape) - 0.5) * 2
        data = unitCubeProj(data + noise)
        
        concat_L2hist = concat_L2hist + L2hist
        concat_Linfhist = concat_Linfhist + Linfhist
    
    return data, concat_L2hist, concat_Linfhist

