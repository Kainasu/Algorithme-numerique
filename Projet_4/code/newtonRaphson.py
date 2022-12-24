
import numpy as np
import copy
import math

def NewtonRaphson(f, J, U, N, epsilon):
    for i in range(N):
        V = np.linalg.lstsq(J(U), -f(U), rcond=None)[0]
        U = U + V
        if (np.linalg.norm(f(U)) < epsilon):
            return (U, True)
    return (U, False)

'''
def f(x):
    newx = copy.deepcopy(x)
    for i in range(len(x)):
        newx[i] = newx[i]**2 -4 
    return newx

def fDeriv(x):
    return 2*x
'''


def NewtonRaphsonBT(f, J, U, N, epsilon):
    k = 0
    Y = f(U)
    while (np.linalg.norm(Y) >= epsilon and k < N):
        k += 1
        V = np.linalg.lstsq(J(U), -Y, rcond=None)[0]
        alpha = 1
        while(np.linalg.norm(f(U + alpha * V)) >= np.linalg.norm(Y)):
            alpha = alpha / 2
            if(alpha < 1e-3):
                return (U, False)
        U = U + V * alpha
        Y = f(U)
    if(k >= N): 
        return (U, False)
    return (U, True)


###### TEST ######

if __name__ == '__main__':
    
    f1 = lambda x: x**2 - 4 
    derivf1 = lambda x: 2*x

    print("For f = x**2 - 4 roots = [-2, 2]")
    print("Without backtracking")
    print("initial point = -1e5, returns ", NewtonRaphson(f1, derivf1, np.array([[-1e5]]), 100, 1e-9))
    print("initial point = 1e5, returns ",NewtonRaphson(f1, derivf1, np.array([[1e5]]), 100, 1e-9))
    print("initial point = 0, returns ",NewtonRaphson(f1, derivf1, np.array([[0]]), 100, 1e-9))


    print("\nWith backtracking")
    print("initial point = -1e5, returns ",NewtonRaphsonBT(f1, derivf1, np.array([[-1e5]]), 100, 1e-9))
    print("initial point = 1e5, returns ",NewtonRaphsonBT(f1, derivf1, np.array([[-1e5]]), 100, 1e-9))
    print("initial point = 0, returns ",NewtonRaphsonBT(f1, derivf1, np.array([[0]]), 100, 1e-9))

    f2 = lambda x: x**2 + 4
    derivf2 = lambda x: 2*x

    print("\nFor f = x**2 + 4 (no roots)")
    print("Without backtracking")
    print("initial point = -1e5, returns ", NewtonRaphson(f2, derivf2, np.array([[-1e5]]), 100, 1e-9))
    print("initial point = 1e5, returns ", NewtonRaphson(f2, derivf2, np.array([[1e5]]), 100, 1e-9))
    print("initial point = 0 (stationnary point), returns ", NewtonRaphson(f2, derivf2, np.array([[0]]), 100, 1e-9))

    print("\nWith backtracking")
    print("initial point = -1e5, returns ",NewtonRaphsonBT(f2, derivf2, np.array([[-1e5]]), 100, 1e-9))
    print("initial point = 1e5, returns ",NewtonRaphsonBT(f2, derivf2, np.array([[1e5]]), 100, 1e-9))
    print("initial point = 0 (stationnary point), returns ",NewtonRaphsonBT(f2, derivf2, np.array([[0]]), 100, 1e-9))

    
    def f3(x):
        n = len(x)
        result = np.zeros([n,1])
        for i in range(n):
            result[i,0] = 10*math.cos(x[i,0])
        return result
    
    def derivf3(x):
        n = len(x)
        result = np.zeros([n,1])
        for i in range(n):
            result[i,0] = -10*math.sin(x[i,0])
        return result


    print("\nFor f = 10*math.cos(x) roots = {pi/2 + kpi | k entier}")
    print("Without backtracking")
    print("initial point = 6, returns ",NewtonRaphson(f3, derivf3, np.array([[6]]), 100, 1e-9))
    print("initial point = -4, returns ",NewtonRaphson(f3, derivf3, np.array([[-4]]), 100, 1e-9))
    print("initial point = 0, returns ",NewtonRaphson(f3, derivf3, np.array([[0]]), 100, 1e-9))

    print("\nWith backtracking")
    print("initial point = 6, returns ",NewtonRaphsonBT(f3, derivf3, np.array([[6]]), 100, 1e-9))
    print("initial point = -4, returns ",NewtonRaphsonBT(f3, derivf3, np.array([[-4]]), 100, 1e-9))
    print("initial point = 0, returns ",NewtonRaphsonBT(f3, derivf3, np.array([[0]]), 100, 1e-9))
    
  
    
    f4 = lambda x: x**3 - 5*x 
    derivf4 = lambda x: 3*x**2 - 5
    
    print("\nFor f = x**3 - 5*x roots = [0, -sqrt(5), sqrt(5)]\n")
    print("Without backtracking")
    print("initial point = -1e5, returns ",NewtonRaphson(f4, derivf4, np.array([[-1e5]]), 100, 1e-9))
    print("initial point = 1e5, returns ",NewtonRaphson(f4, derivf4, np.array([[1e5]]), 100, 1e-9))
    print("initial point = 1, returns ",NewtonRaphson(f4, derivf4, np.array([[1]]), 100, 1e-9))
    
    

    print("\nWith backtracking")
    print("initial point = -1e5, returns ",NewtonRaphsonBT(f4, derivf4, np.array([[-1e5]]), 100, 1e-9))
    print("initial point = 1e5, returns ",NewtonRaphsonBT(f4, derivf4, np.array([[1e5]]), 100, 1e-9))
    print("initial point = 1, returns ",NewtonRaphsonBT(f4, derivf4, np.array([[1]]), 100, 1e-9))

    def f5(x):
        if x >= 0:
            return x**(1/3)
        return -(-x)**(1/3)
        
    def derivf5(x):
        if x >= 0:
            return (1/3) * x**(-2/3)
        return (1/3) * (-x)**(-2/3)

    print("\nFor f = x**(1/3) roots = [0]\n")
    print("Without backtracking")
    print("initial point = 1, returns ",NewtonRaphson(f5, derivf5, np.array([[1]]), 100, 1e-9))
    print("initial point = 5, returns ",NewtonRaphson(f5, derivf5, np.array([[5]]), 100, 1e-9))
    print("initial point = -4, returns ",NewtonRaphson(f5, derivf5, np.array([[-4]]), 100, 1e-9))
    
    
    

    print("\nWith backtracking")
    print("initial point = 1, returns ",NewtonRaphsonBT(f5, derivf5, np.array([[1]]), 100, 1e-9))
    print("initial point = 5, returns ",NewtonRaphsonBT(f5, derivf5, np.array([[5]]), 100, 1e-9))
    print("initial point = -4, returns ",NewtonRaphsonBT(f5, derivf5, np.array([[-4]]), 100, 1e-9))
    
    '''
    def f4(x):
        n = len(x)
        result = np.zeros([n,1])
        for i in range(n):
            result[i,0] = 10*math.cos(0.4*x[i,0]) + 20 * math.sin(0.4*x[i,0])
        return result


    def derivf4(x):
        n = len(x)
        result = np.zeros([n,1])
        for i in range(n):
            result[i,0] =  -4*math.sin(0.4*x[i,0]) + 8 * math.cos(0.4*x[i,0])
        return result


    print("\nf = 10*math.cos(0.4*x) + 20 * math.sin(0.4*x)\n")
    print("Without backtracking\n")
    print(NewtonRaphson(f4, derivf4, np.array([[6]]), 100, 1e-9))
    print(NewtonRaphson(f4, derivf4, np.array([[-4]]), 100, 1e-9))
    print(NewtonRaphson(f4, derivf4, np.array([[0]]), 100, 1e-9))

    print("With backtracking\n")
    print(NewtonRaphsonBT(f4, derivf4, np.array([[6]]), 100, 1e-9))
    print(NewtonRaphsonBT(f4, derivf4, np.array([[-4]]), 100, 1e-9))
    print(NewtonRaphsonBT(f4, derivf4, np.array([[0]]), 100, 1e-9))
    '''
