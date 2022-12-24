import numpy as np
import sys
sys.path.append("../cholesky")
from facto_cholesky import *
from matrix_generator import *
import matplotlib.pyplot as plt

def conjGradPrecond_graph(A, b, x):
    M = make_preconditioner(A)
    xt = x.shape
    x = x.flatten()
    x = x.reshape(len(x),1)
    b = b.flatten()
    b = b.reshape(len(b),1)
    r = b - np.dot(A,x)
    z = solve_cholesky(M, r)
    p = z
    rsold = np.dot(np.transpose(r), z)
    L = []
    while(np.sqrt(np.dot(np.transpose(r), r)) > 1e-5):
        L.append(np.sqrt(np.dot(np.transpose(r), r))[0][0])
        Ap = np.dot(A,p)
        alpha = rsold/(np.dot(np.transpose(p), Ap))
        x += alpha*p
        r -= alpha * Ap
        z = solve_cholesky(M, r)
        rsnew = np.dot(np.transpose(r), z)
        p = z + (rsnew/rsold)*p
        rsold = rsnew
    L.append(np.sqrt(np.dot(np.transpose(r), r))[0][0])
    return x.reshape(xt), L

def conjGrad_graph(A, b, x):
    xt = x.shape
    x = x.flatten()
    x = x.reshape(len(x),1)
    b = b.flatten()
    b = b.reshape(len(b),1)
    r = b - np.dot(A,x)
    p = r
    rsold = np.dot(np.transpose(r), r)
    L = []
    for i in range(1, 10**6):
        L.append(np.sqrt(np.dot(np.transpose(r), r))[0][0])
        Ap = np.dot(A,p)
        alpha = rsold/(np.dot(np.transpose(p), Ap))
        x += alpha*p
        r -= alpha * Ap
        rsnew = np.dot(np.transpose(r), r)
        if np.sqrt(rsnew) < 1e-5:
            break
        p = r + rsnew/rsold*p
        rsold = rsnew
    return x.reshape(xt),L

def conjGradPrecond(A, b, x):
    print("GRADIENT PRECONDITIONNE : (||r|| needs to be lower than 0.001)")
    M = make_preconditioner(A)
    xt = x.shape
    x = x.flatten()
    x = x.reshape(len(x),1)
    b = b.flatten()
    b = b.reshape(len(b),1)
    r = b - np.dot(A,x)
    z = solve_cholesky(M, r)
    p = z
    rsold = np.dot(np.transpose(r), z)
    while(np.sqrt(np.dot(np.transpose(r), r)) > 1e-3):
        print("||r|| : ", np.sqrt(np.dot(np.transpose(r), r))[0][0])
        Ap = np.dot(A,p)
        alpha = rsold/(np.dot(np.transpose(p), Ap))
        x += alpha*p
        r -= alpha * Ap
        z = solve_cholesky(M, r)
        rsnew = np.dot(np.transpose(r), z)
        p = z + (rsnew/rsold)*p
        rsold = rsnew
    print("||r|| : ", np.sqrt(np.dot(np.transpose(r), r))[0][0])
    return x.reshape(xt)

def conjGrad(A, b, x):
    xt = x.shape
    x = x.flatten()
    x = x.reshape(len(x),1)
    b = b.flatten()
    b = b.reshape(len(b),1)
    r = b - np.dot(A,x)
    p = r
    rsold = np.dot(np.transpose(r), r)
    for i in range(1, 10**6):
        Ap = np.dot(A,p)
        alpha = rsold/(np.dot(np.transpose(p), Ap))
        x += alpha*p
        r -= alpha * Ap
        rsnew = np.dot(np.transpose(r), r)
        if np.sqrt(rsnew) < 1e-5:
            break
        p = r + rsnew/rsold*p
        rsold = rsnew
    return x.reshape(xt)

def make_preconditioner(A):
    T_inc = cholesky_incomplete(A)
    preconditionner = np.dot(T_inc, np.transpose(T_inc))
    return preconditionner

if __name__ == '__main__':

    x = np.random.rand(4,1)
    A = np.array([[1, 1, 1, 1],
                  [1, 5, 5, 5],
                  [1, 5, 14, 14],
                  [1, 5, 14, 15]])

    b = np.array([[5], [4], [2], [1]])

    b = np.array([[5], [4], [2], [1]])



    A = make_SSPD_matrix(10, 5)
    x = np.zeros([10,1])
    b = np.random.rand(1,10) * 1000
    b = b.reshape(10, 1)
    M = make_preconditioner(A)
    
    T1 = conjGradPrecond_graph(A, b, x)
    T2 = conjGrad_graph(A, b, x)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Norme du vecteur r")
    axes[0].title.set_text("Sans préconditionneur (en " + str(len(T2[1])) + " iterations)")
    
    plot1 = axes[0].plot(np.arange(1, len(T2[1]) + 1), T2[1], label="||r||")
    axes[0].legend()

    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Norme du vecteur r")
    axes[1].title.set_text("Avec préconditionneur (en " + str(len(T1[1])) + " iterations)")

    
    plot1 = axes[1].plot(np.arange(1, len(T1[1]) + 1), T1[1], label="||r||")
    axes[1].legend()

    plt.show()
