import numpy as np
import random
import matplotlib.pyplot as plt

def softmax(a, b, delta):
    return 0.5 * (a + b + np.sqrt(delta + (a-b)**2))

def smooth_L2(A, delta, edge_buffer):
    n=len(A[0, :])
    a=0
    for i in range(n):
        a+=(1-A[0, i]**2)*(1-A[1, i]**2)
    a=a/(2*n)
    b=0
    penalties = 0
    for i in range(n):
        penalties += edge_penalty(A[0, i], edge_buffer) + edge_penalty(A[1, i], edge_buffer)
        for j in range(n):
            b+=(1-softmax(A[0, i],A[0, j], delta))*(1-softmax(A[1, i], A[1, j], delta))
    b=b/(n**2)
    b+= penalties
    return(1/9-a+b)

def edge_penalty(x, delta):
    #penalty is currently a bump function, which is nonzero on [0, delta] or [delta, 1]
    if 1-x < delta:
        return (np.exp(1 / (((x-1)/delta)**2 - 1)))
    if x < delta:
        return (np.exp(1 / ((x/delta)**2 - 1)))
    else:
        return 0
    
def numerical_gradient(A, ve, delta, tol, edge_buffer):
    grad = 0*A
    for i in range(len(A[:, 0])):
        for j in range(len(A[0, :])):
            A_step =np.array(A)
            A_step[i,j] = A[i,j] + ve
            grad[i,j] = (smooth_L2(A_step, delta, edge_buffer) - smooth_L2(A, delta, edge_buffer)) / ve
    return grad

def gradient_descent(A, maxiter = 500, delta = 10**(-8), ve = 10**(-8), tol = 10**(-6), edge_buffer = None):
    if edge_buffer == None:
        edge_buffer = 0.1 / len(A[0, :])
    
    state = np.array(A)
    for w in range(maxiter):
        gradient = numerical_gradient(state, ve, delta, tol, edge_buffer)
        if np.linalg.norm(gradient) < tol:
            print(w)
            break
        
        alpha = 1
        c= 10**(-3)
        current_val = smooth_L2(state, delta, edge_buffer)
        alpha = 1
        
        while True:
            new_val = smooth_L2(state -alpha*gradient, delta, edge_buffer)
            if np.any(new_val < 0) or np.any(new_val > 1):
                alpha *= 0.5
            elif new_val <= current_val -c*alpha*(np.linalg.norm(gradient))**2:
                break
            else:
                alpha *= 0.5
            
            if alpha < tol:
                print('Backtrack line search fail')
                break
        
        state = state -alpha*gradient
        state = np.clip(state, 0, 1)
        if w==(maxiter - 1):
            print('Maximum iterations reached')
    return state

def randmatrix(n):
    M = np.zeros([2, n])
    for i in range(n):
        M[0, i] = random.random()
        M[1, i] = random.random()
    return(M)

def L2(A):
    n=len(A[0, :])
    a=0
    for i in range(n):
        a+=(1-A[0, i]**2)*(1-A[1, i]**2)
    a=a/(2*n)
    b=0
    for i in range(n):
        for j in range(n):
            b+=(1-max(A[0, i],A[0, j]))*(1-max(A[1, i], A[1, j]))
    b=b/(n**2)
    return(1/9-a+b)

def arrowgraph(M1, M2, M1_label = 'M1', M2_label = 'M2', graphtitle = 'M1 to M2'):
    #creates a graph of arrows from M1 to M2
    start_x = M1[0, :]
    start_y = M1[1, :]
    direction_x = M2[0, :] - M1[0, :]
    direction_y = M2[1, :] - M1[1, :]

    plt.figure(figsize=(6, 6))
    plt.quiver(start_x, start_y, direction_x, direction_y, angles='xy', scale_units='xy', scale=1, color='blue')
    plt.scatter(start_x, start_y, color='red', label= M1_label)
    plt.scatter(M2[0, :], M2[1, :], color='green', label= M2_label)
    plt.axis('equal')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(graphtitle)
    plt.legend()
    plt.show()

def relative_position(A, tol):
    #creates a matrix of relative positions of points
    A_copy = np.array(A)
    position = 0*A
    n = len(A[0, :])
    
    for i in range(n):
        current_val = min(A_copy[0, :])
        if current_val > 1:
            break
        for j in range(n):
            if np.abs(A[0, j] - current_val) < tol:
                position[0, j] = i+1
                A_copy[0, j] = 2
    
    for i in range(n):
        current_val = min(A_copy[1, :])
        if current_val > 1:
            break
        for j in range(n):
            if np.abs(A[1, j] - current_val) < tol:
                position[1, j] = i+1
                A_copy[1, j] = 2
                
    return position

def compare_positions(M1, M2, tol = 10**(-16), title1 = 'M1', title2 = 'M2'):
    #creates two position graphs, one for M1 and one for M2
    fig, axs = plt.subplots(1, 2)
    positions_1 = relative_position(M1, tol)
    positions_2 = relative_position(M2, tol)

    axs[0].scatter(positions_1[0, :], positions_1[1, :], color = 'red')
    axs[0].set_title(title1)
    axs[0].set_aspect('equal', adjustable='box')
    axs[0].set_xticks(np.arange(min(positions_1[0, :]), max(positions_1[0, :])+1, 1.0))
    axs[0].set_yticks(np.arange(min(positions_1[1, :]), max(positions_1[1, :])+1, 1.0))
    axs[0].grid()

    axs[1].scatter(positions_2[0, :], positions_2[1, :], color = 'green')
    axs[1].set_title(title2)
    axs[1].set_aspect('equal', adjustable='box')
    axs[1].set_xticks(np.arange(min(positions_2[0, :]), max(positions_2[0, :])+1, 1.0))
    axs[1].set_yticks(np.arange(min(positions_2[1, :]), max(positions_2[1, :])+1, 1.0))
    axs[1].grid()

    plt.show()

def quadgraph(M1, M2 = None, M1_title = 'Initialization', M2_title = 'Optimized', max_iterations = 500, title = 'Quadgraph', tol = 10**(-16), include_L2 = True):
    #creates four graphs in a 2 by 2 grid
    
    #if no M2 specifified, runs gradient descent on M1
    if M2 == None:
        M2 = gradient_descent(M1, maxiter = max_iterations)
    
    fig, axs = plt.subplots(2, 2)
    positions_1 = relative_position(M1, tol)
    positions_2 = relative_position(M2, tol)
    
    start_x = M1[0, :]
    start_y = M1[1, :]
    direction_x = M2[0, :] - M1[0, :]
    direction_y = M2[1, :] - M1[1, :]

    #graph of the final point set
    axs[0,0].quiver(start_x, start_y, direction_x, direction_y, angles='xy', scale_units='xy', scale=1, color='blue')
    axs[0,0].scatter(start_x, start_y, color='red', label= M1_title)
    axs[0,0].scatter(M2[0, :], M2[1, :], color='green', label= M2_title)
    axs[0,0].set_aspect('equal', adjustable='box')
    axs[0, 0].set_xticks(np.arange(0, 1.2, 0.2))
    axs[0, 0].set_yticks(np.arange(0, 1.2, 0.2))
    axs[0,0].set_title(M1_title + ' to ' + M2_title)
    
    #arrow graph
    axs[0, 1].scatter(M2[0, :], M2[1, :], color = 'green')
    axs[0, 1].set_title(M2_title + ' Graph')
    axs[0, 1].set_aspect('equal', adjustable='box')
    axs[0, 1].set_xticks(np.arange(0, 1.2, 0.2))
    axs[0, 1].set_yticks(np.arange(0, 1.2, 0.2))

    #M1 position graph
    axs[1, 0].scatter(positions_1[0, :], positions_1[1, :], color = 'red')
    axs[1, 0].set_title(M1_title + ' Positions')
    axs[1, 0].set_aspect('equal', adjustable='box')
    axs[1, 0].set_xticks(np.arange(min(positions_1[0, :]), max(positions_1[0, :])+1, 1.0))
    axs[1, 0].set_yticks(np.arange(min(positions_1[1, :]), max(positions_1[1, :])+1, 1.0))
    axs[1, 0].grid()

    #M2 position graph
    axs[1, 1].scatter(positions_2[0, :], positions_2[1, :], color = 'green')
    axs[1, 1].set_title(M2_title + ' Positions')
    axs[1, 1].set_aspect('equal', adjustable='box')
    axs[1, 1].set_xticks(np.arange(min(positions_2[0, :]), max(positions_2[0, :])+1, 1.0))
    axs[1, 1].set_yticks(np.arange(min(positions_2[1, :]), max(positions_2[1, :])+1, 1.0))
    axs[1, 1].grid()
    
    if include_L2:
        plt.suptitle(title + ': L2 = ' + str(np.sqrt(L2(M2))))
    else: 
        plt.suptitle(title)
    plt.tight_layout()
    plt.show()