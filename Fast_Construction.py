import numpy as np 

def L2_Square(A):
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
    
def find_vertex(A, k, i):
    #Let f(x) be the function that outputs the L2 value of A when A[k, i] is replaced by x
    #Then f(x) is a continuous piecewise construction of parabolas, with new components after x passes the value of A[k, j] for each 0 =< j =< n-1
    #This function finds the location of the vertex of the parabola segment that A[k, i] is currently located in
    #The vertex may or may not lie within the domain of the peicewise segment
    n = len(A[0, :])
    if k == 0:
        a = (-1 / (2*n))*((A[1,i]**2)-1)
        b_so_far = (A[1, i]-1)/(n**2) 
        for j in range(n):
            if not (j == i):
                if A[0, i] > A[0, j]:
                    b_so_far += (2 / (n**2)) * (max(A[1, i], A[1, j]) - 1)
    if k == 1:
        a = (-1 / (2*n))*((A[0,i]**2)-1)
        b_so_far = (A[0, i]-1)/(n**2) 
        for j in range(n):
            if not (j == i):
                if A[1, i] > A[1, j]:
                    b_so_far += (2 / (n**2)) * (max(A[0, i], A[0, j]) - 1)
    vertex = -b_so_far / (2*a)
    return vertex
    
def fibonacci_points(n, irrational):
    M = np.zeros([2, n])
    for i in range(n):
        M[0, i] = i / n
        M[1, i] = (irrational * i) % 1
    return(M)

golden_ratio = (1+ np.sqrt(5)) / 2

def construct_set(n, maxiter):
    A = fibonacci_points(n, golden_ratio)
    
    #Below constructs rel_x, rel_y
    #The ith coordinate of rel_x is the coordinate j such that A[0, j] has the ith lowest relative position amongst A[0, l]
    #Similar for rel_y
    #This is necessary to figure out if the vertex is actually a good point to move to, or if moving to it would change the relative position of the points (bad)
    rel_x = np.zeros(n)
    rel_y = np.zeros(n)
    x_search = np.sort(A[0, :])
    y_search = np.sort(A[1, :])
    for l in range(n):
        for r in range(n):
            if A[0, r] == x_search[l]:
                rel_x[l] = r
            if A[1, r] == y_search[l]:
                rel_y[l] = r
    
    for w in range(maxiter):
        #This looks a little odd, but just iterates over all [k, i] pairs as w increases
        selected_coordinate = np.array([w % 2, int(((w % (2*n)) - (w % 2)) / 2)])
        
        vertex = find_vertex(A, selected_coordinate[0], selected_coordinate[1])
        
        if selected_coordinate[0] == 0:
            for l in range(n):
                #finds relative position of the chosen coordinate
                if rel_x[l] == selected_coordinate[1]:
                    select_l = l
                    break
            
            #finds the coordinates before and after in relative position
            if (select_l+1) < n:
                next_coordinate = A[0, int(rel_x[select_l + 1])]
            else:
                next_coordinate = 1
        
            if (select_l-1) > 0:
                prior_coordinate = A[0, int(rel_x[select_l - 1])]
            else:
                prior_coordinate = 0

            #if going to the vertex wouldn't change the relative position, it moves A[k, i] there
            if (vertex > prior_coordinate) and (vertex < next_coordinate):
                A[0, selected_coordinate[1]] = vertex
            
            #Else, it just puts it halfway between the two closest coordinates
            else:
                A[0, selected_coordinate[1]] = (next_coordinate + prior_coordinate) / 2
            
        if selected_coordinate[0] == 1:
            for l in range(n):
                if rel_y[l] == selected_coordinate[1]:
                    select_l = l
                    break
            
            if (select_l+1) < n:
                next_coordinate = A[1, int(rel_y[select_l + 1])]
            else:
                next_coordinate = 1
        
            if (select_l-1) > 0:
                prior_coordinate = A[1, int(rel_y[select_l - 1])]
            else:
                prior_coordinate = 0
        
            if (vertex > prior_coordinate) and (vertex < next_coordinate):
                A[1, selected_coordinate[1]] = vertex
            else:
                A[1, selected_coordinate[1]] = (next_coordinate + prior_coordinate) / 2
        
    print(np.sqrt(L2_Square(A)))
    return(A)