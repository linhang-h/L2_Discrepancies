import numpy as np
import matplotlib.pyplot as plt
import random
tol = 10**(-10)

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

def find_vertices(A, k, i): 
    #finds vertices of the parabolas that make up the one variable function along the (k, i) axis-parallel line through the current point
    n = len(A[0, :])
    if k == 0:
        a = (-1 / (2*n))*((A[1,i]**2)-1)
        #b value for leftmost parabola
        first_b = (A[1, i]-1)/(n**2) 
        vertex_set = np.array([-(first_b)/(2*a)]) 
        xks = np.sort(np.delete(A[0, :], i), 0)
        xks_and_one = np.append(xks, 1)
        for l in range(n-1):
            b_so_far = first_b
            for j in range(n):
                if (not j==i) and (xks[l] - A[0, j] > -tol):
                    b_so_far += (2 / (n**2))*(max(A[1, i], A[1, j]) - 1)
            vertex = (-b_so_far / (2*a))
            if (vertex < 1) and (vertex > 0) and (vertex > xks_and_one[l]) and (vertex < xks_and_one[l+1]):
                vertex_set = np.append(vertex_set, vertex)
        return(vertex_set)
    if k == 1:
        a = (-1 / (2*n))*((A[0,i]**2)-1)
        #b value for leftmost parabola
        first_b = (A[0, i]-1)/(n**2) 
        vertex_set = np.array([-(first_b)/(2*a)]) 
        xks = np.sort(np.delete(A[1, :], i), 0)
        xks_and_one = np.append(xks, 1)
        for l in range(n-1):
            b_so_far = first_b
            for j in range(n):
                if (not j ==i) and (xks[l] - A[1, j] > -tol):
                    b_so_far += (2 / (n**2))*(max(A[0, i], A[0, j]) - 1)
            vertex = (-b_so_far / (2*a))
            if (vertex < 1) and (vertex > 0) and (vertex > xks_and_one[l]) and (vertex < xks_and_one[l+1]):
                vertex_set = np.append(vertex_set, vertex)
        return(vertex_set)
    
def axis_parallel_descent(A, iter, coordinate_rule = 'list', search_rule = 'min', meta_rule = 'single', include_coordinate_equality = True, include_extremes = False):
    #coordinate rules: 'list' optimizes parallel to each axis down a list, 'random' selects axes randomly
    #meta rules: 'single' optimizes over one axis and moves, 'search' picks the best improvement across the minima optimizing parallel to each axis
    
    D = np.array(A)
    n = len(A[0, :])
    selected_coordinate = np.array([0, 0])
            
    for w in range(iter):
        if meta_rule == 'single': 
            
            def L2_sub(A, k, i, val):
                Y = np.array(A)
                Y[k, i] = val
                sum = 0
                return(L2(Y)+sum)
            
            if coordinate_rule == "random": 
                selected_coordinate = np.array([random.randint(0, 1), random.randint(0, len(A[0, :])-1)])
        
            if coordinate_rule == "list":
                #just iterates over points: (0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2), etc
                selected_coordinate[1] = int(((w % (2*n)) - (w % 2)) / 2) 
                selected_coordinate[0] = w % 2

            search_set = find_vertices(D, selected_coordinate[0], selected_coordinate[1])

            if include_extremes:
                search_set = np.append(search_set, [0, 1])
                
            if include_coordinate_equality:
                search_set = np.append(search_set, A[selected_coordinate[0], :])
        
            if search_rule == 'min':
                current_min = 10
                current_argmin = 0
                for i in range(len(search_set)):
                    if L2_sub(D, selected_coordinate[0], selected_coordinate[1], search_set[i]) < current_min:
                        current_min = L2_sub(D, selected_coordinate[0], selected_coordinate[1], search_set[i])
                        current_argmin = i
                D[selected_coordinate[0], selected_coordinate[1]] = search_set[current_argmin]
        
        if meta_rule == "search":
            
            def L2_sub(A, k, i, val):
                Y = np.array(A)
                Y[k, i] = val
                sum = 0
                return(L2(Y)+sum)
            
            current_best_coordinate = np.array([0, 0])
            current_best_coord_placement = 0
            current_best_val = 10
            for i in range(2):
                for j in range(n):
                    search_set = find_vertices(D, i, j)
                    
                    if include_extremes:
                        search_set = np.append(search_set, [0, 1])
                
                    if include_coordinate_equality:
                        search_set = np.append(search_set, A[i, :])
        
                    if search_rule == 'min':
                        current_min = 10
                        current_argmin = 0
                        for k in range(len(search_set)):
                            if L2_sub(D, i, j, search_set[k]) < current_min:
                                current_min = L2_sub(D, i, j, search_set[k])
                                current_argmin = k
                                
                    if current_min < current_best_val:
                        current_best_coordinate = np.array([i, j])
                        current_best_coord_placement = search_set[current_argmin]
                        current_best_val = current_min
                        
            D[current_best_coordinate[0], current_best_coordinate[1]] = current_best_coord_placement              
    return(D)