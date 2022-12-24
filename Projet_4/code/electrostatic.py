import numpy as np 
from newtonRaphson import *
import random as r
import matplotlib.pyplot as plt

#Returns the electrostatic energy of all charges which have their position in vector X
def E(X):
    N = len(X)
    sum1 = 0
    for i in range (N):
        sum2 = 0
        for j in range (N):
            if j != i:
                sum2 += np.log(np.abs(X[i] - X[j]))
        sum1 += np.log(np.abs(X[i] + 1)) + np.log(np.abs(X[i] -1)) + 1/2*sum2
    return sum1


#Returns the derivate of E in X with respect to X[i]
def iDerivE(X, i):
    N = len(X)
    sum = 0
    for j in range(N):
        if j != i:
            sum += 1/(X[i]-X[j])
    t1 = 1/(X[i] + 1)
    t2 = 1/(X[i] - 1)
    return t1 + t2 + sum
'''
#Returns the derivate of E in X with respect to X[i]
def iDerivE2(X, i):
    
    N = len(X)
    sum = 0
    for j in range(N):
        if j != i:
            sum += 1/(X[i]-X[j])
            

    return (1 / (X[i] + 1)) + (1 / (X[i] - 1)) #+ sum

'''
#Returns the vector "nabla"E(x1, ..., xn) depending on X = (x1, ..., xn)
def nablaE(X):
    N = len(X)
    Res = np.zeros(N)
    for i in range(N):
        Res[i] = iDerivE(X, i)
    return Res

'''
#Returns the vector "nabla"E(x1, ..., xn) depending on X = (x1, ..., xn)
def nablaE2(X):
    N = len(X)
    Res = np.zeros(N)
    for i in range(N):
        Res[i] = iDerivE2(X, i)
    return Res
'''
#Returns the jacobian of "nabla"E(x1, ..., xn) depending on X = (x1, ..., xn)
def JNablaE(X):
    N = len(X)
    J = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if(i == j):
                t1 = - 1 / (X[i] + 1)**2
                t2 = - 1 / (X[i] - 1)**2
                sum = 0 
                for m in range(N):
                    if(i != m):
                        sum += 1 / (X[i] - X[m])**2
                v = t1 + t2 - sum
            else:
                v = 1 / (X[i] - X[j])**2
            J[i][j] = v
    return J

#Returns a vector filled with N vertical axis coordinate for the N eletric charges
#a is the half length of the interval 
def randomCharges(N, a, seed):
    r.seed(seed)
    X = np.zeros(N)
    for i in range(N):
        sign = r.random() * 50
        if(sign < 25):
            v = r.random() * a 
        else:
            v = - r.random() * a
        X[i] = v
    return X

if __name__ == '__main__':
    #X = randomCharges(7, 1, r.random())
    #X = np.array([-0.8, -0.6, 0.2, 0.4, 0.6, 0.8, 0.3, 0.05])
    l = np.linspace(-1, 1, 200)
    X = np.array([i/4 for i in range(-3, 4)])
    print(X)
    print()
    R = NewtonRaphsonBT(nablaE, JNablaE, X, 1000, 0.01)
    print(R)
    plt.plot([-1, 1], [0, 0] , "r:o")
    plt.plot(R[0], np.zeros(len(R[0])), "b:o")
    P = np.polynomial.legendre.Legendre((0, 1, 2, 3, 4, 5, 6))
    plt.plot(l, P(l))
    plt.show()