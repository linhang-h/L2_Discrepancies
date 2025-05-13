#dependency
import qmcpy
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax import grad, jit
import optax
from qmcpy import Halton
import itertools



#visualization utilities
class VisUtils:

    def plot2DArrow(init, end, ax = None):
        if ax is None:
            ax = plt.gca()
        dx = end[:,0] - init[:,0]
        dy = end[:,1] - init[:,1]
        plt.quiver(init[:,0], init[:,1], dx, dy, angles='xy', scale_units='xy', scale = 1, color='blue', width=0.003)
        plt.legend()
        plt.show()

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

    def heatmap_from_array(P, ax=None): 
        import numpy as np
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
                Z[j][i] = VisUtils.locdisc(X[i], Y[j], P)
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



class funcUtils:

    class Grid:
        def __init__(self, param_spaces):
            self._generator = itertools.product(*param_spaces)

        def pop(self):
            return next(self._generator, None) 
        

    def internalize(points):
        '''
        given the information of point set, 
        
        returns:
        
        the sorted version of each coordinate, (sortedCoordinates)
        and the order indices in which coordinates of each point appears in the sortedCoordinates, (ordidx)
        and the 'function' that takes an arbitrary order index and tell if it is actual point or not (ispoint)
        
        in that we save memory inefficiency and duplication of codes.
        '''
        sortedCoordinates = []
        d = points.shape[1]
        n = points.shape[0]
        for i in range(d):
            sortedCoordinates.append(sorted(points[:,i]))

        ordidx = tuple([[sortedCoordinates.index(points[j][i]) for i in range(d)] for j in range(n)])        
    
        def ispoint(ordidx):
            refTable = sortedCoordinates
            recv = [refTable[ordidx[i]] for i in range(len(ordidx))]
            if recv in points:
                return 1
            else:
                return 0
    
        return sortedCoordinates, ordidx, ispoint


    def smooth_max(a, b, tau=1e-4):
        '''
        Smoothened maximum function. Control tau to adjust accuracy.
        '''
        return 0.5 * (a + b + ((a - b)**2 + tau)**(0.5))
    
    
    def L2Disc(points, smoothing = False):
        '''
         L2 Discrepancy, following the Wornock's formula. Written in Numpy setting.
        '''
        n = points.shape[0]
        d = points.shape[1]
        SP1 = np.sum(np.prod(1 - points**2, axis=1))
        SP2 = 0
        for i in range(n):
            for j in range(n):
                cash_k = 1
                for k in range(d):
                    factor = 1
                    if smoothing:
                        factor = (1 - funcUtils.smooth_max(points[i][k],points[j][k]))
                    else:
                        factor = (1 - max(points[i][k],points[j][k]))
                    cash_k *= factor
                SP2 += cash_k
        
        return 1/(3**d) - 2**(1-d)/n * SP1 + 1/(n**2) * SP2
        
    
    def LinfDisc(points):
        '''
        L - Inf Discrepancy by definition.
        '''
        n = points.shape[0]
        d = points.shape[1]

        sc, oid, f = funcUtils.internalize(points)
        disc = 0.0
        grid = funcUtils.Grid([[i for i in range(n)] for j in range(d)])
        while True:
            gridix = grid.pop()
            if gridix is None:
                break
            vol = np.product([points[gridix[j]][j] for j in range(d)])
            threshold = [sc[j].index(points[gridix[j]]) for j in range(d)]
            opencount = np.sum([np.all(oid[i] < threshold) for i in range(n)])
            closedcount = np.sum([np.all(oid[i] <= threshold) for i in range(n)])
            disc = max(disc, opencount/n - vol, vol - closedcount/n)

        return disc

    def unitCubeProj(points):
        return np.clip(points, 0.0, 1.0)


class SpecialSeqs:

    def getKronecker(n, phi, d =2):
        '''
        Generates Kronecker lattice. For now 2-dimensional only.
        '''
        x = np.linspace(0,1,n,endpoint = False)
        y = np.mod( np.arange(n) * phi, 1.0)
        return np.column_stack((x,y))
    
    def getFibonacci(n, d = 2):
        '''
        Generates first n terms of the fibonacci sequence. Currently 2 - dimensional only
        '''
        return SpecialSeqs.getKronecker(n, (np.sqrt(5) - 1)/2, d)
    
    def getHalton(n, d = 2):
        HaltonGenerator = Halton(dimension=d)
        return HaltonGenerator.gen_samples(n)
    
    def getRandom(n, d = 2):
        return np.random.rand(n,d)
    
    def getMPMC(n, d = 2):
        pass



        
    



class WrappedOptimizer:

    def __init__(self, name, optimizer, init_data, loss_fn, **additionals):
        self.optimizer = optimizer
        self.name = name
        self.lossfn = loss_fn
        self._state = init_data
        self.opt_state = self.optimizer.init(init_data)
        self.additionals = additionals
        self._func_value = self.lossfn(self._state)
        self.history = [self._func_value]
        self._dec = None
        self._enc = None
    

    def step(self):
        # basically decorator, but class instance version
        if self._enc is not None:
            self._enc(self)
        lfvalue, lfgrad = jax.value_and_grad(self.lossfn)(self.opt_state)
        updates, opt_state = self.optimizer.update(lfgrad, self.opt_state, params = self.state, **self.additionals) # params = self.state may be optional
        self.state = optax.apply_updates(self.state, updates)
        self.func_value = self.lossfn(self.state)
        self.history.append(self.func_value)
        #basically decorator, but class instance version
        if self._dec is not None:
            self._dec(self)
    

    @property
    def state(self): return self.state
    @property
    def lossfn_value(self): return self.func_value
    @property
    def dec(self): return self.dec
    @property
    def enc(self): return self.enc
    @dec.setter
    def dec(self, dec1): self._dec = dec1
    @enc.setter
    def enc(self, enc1): self._enc = enc1 



def iterate(wopts, milestone = 15, maxiter = 1000, tol = 1e-7,
        trail = 'simple', LInf = True):
    #initialization
    inum = 0
    initial_states = [wopt.state for wopt in wopts]

    figpts = []
    axpts = []

    LInf_history = [[funcUtils.LinfDisc(initial_state)] for initial_state in initial_states]
    
    #initial plots
    for i in range(len(initial_states)):
        figpts.append(plt.figure(figsize=(6, 6)))
        ip = initial_states[i]
        plt.scatter(ip[:, 0], ip[:, 1], c ='red', label='Initial')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.title("Summary")
        plt.gca().set_aspect('equal')
        axpts.append(plt.gca())
        plt.grid(True)
        plt.tight_layout()


    while inum < maxiter:
        before_states = [wopt.state for wopt in wopts] # diff
        for wopt in wopts:
            wopt.step()
        after_states = [wopt.state for wopt in wopts] 

        if inum % milestone == 0:
            #do the milestone work
            if trail == 'continuous':
                    for i in range(len(after_states)):
                        VisUtils.plot2DArrow(before_states[i], after_states[i], ax = axpts[i])
            if LInf:
                for i in range(len(after_states)):
                    LInf_history[i].append(funcUtils.LinfDisc(after_states[i]))
        inum = inum + 1

        #stop condition
        if np.sum([np.linalg.norm(after_states[i] - before_states[i]) for i in range(len(after_states))]) < tol:
            break
    
    final_states = [wopt.state for wopt in wopts]
    L2_history = [wopt.history for wopt in wopts]
    LInf_history.append()

    #final plots
    for i in range(len(initial_states)):
        fp = final_states[i]
        plt.scatter(fp[:, 0], fp[:, 1], c ='green', label='Optimized')
        if trail == 'simple': #Arrow is just poltted between initial and final
            VisUtils.plot2DArrow(initial_states[i], fp, ax = plt.gca())
        plt.legend()
    
    plt.show()


    #history plots(L2)
    plt.figure(figsize=(8, 5))
    for i in range(len(initial_states)):
        plt.plot(L2_history[i], label='L2 discrepancy: '+ wopts[i].name)
    plt.xlabel("Iteration")
    plt.ylabel("Discrepancy")
    plt.title("Discrepancy vs. Iteration")
    plt.legend()    
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    if LInf: # history plots(LInf)
        plt.figure(figsize=(8, 5))
        for i in range(len(initial_states)):
            plt.plot(LInf_history[i], label='LInf discrepancy: '+ wopts[i].name)
        plt.xlabel("Iteration")
        plt.ylabel("Discrepancy")
        plt.title("Discrepancy vs. Iteration")
        plt.legend()    
        plt.grid(True)
        plt.tight_layout()
        plt.show()




    #heatmaps
    for i in range(len(initial_states)):
        fig, axs = plt.subplots(1, 2, figsize=(14, 5))

        # Heatmap for initial set
        _, max_init = VisUtils.heatmap_from_array(initial_states[i], ax=axs[0])
        axs[0].set_title("Initial: " + wopts[i].name)
        axs[0].text(0.5, -0.15, f"Max Local Discrepancy: {max_init:.5f}",
                transform=axs[0].transAxes, ha='center', fontsize=11)
        # Heatmap for optimized set
        _, max_opt = VisUtils.heatmap_from_array(final_states[i], ax=axs[1])
        axs[1].set_title("Optimized: "+ wopts[i].name)
        axs[1].text(0.5, -0.15, f"Max Local Discrepancy: {max_opt:.5f}",
                transform=axs[1].transAxes, ha='center', fontsize=11)

        plt.tight_layout()
    
    plt.show()

    #return final_states



#initialization of optimizers

Kroneckerpoints = SpecialSeqs.getKronecker(np.sqrt(2))
learning_rate = 0.0005
Adamoptimizer = optax.adam(learning_rate)
loss_fn = lambda point: funcUtils.L2_smoothed

SGDBacktracker = optax.chain(
   optax.sgd(learning_rate=1.),
   optax.scale_by_backtracking_linesearch(max_backtracking_steps=15)
) #chain needs more arguments.

#Code example
Adamopt = WrappedOptimizer("Adam on Kronecker", Adamoptimizer, Kroneckerpoints, loss_fn) # for example,
iterate([Adamopt])

    