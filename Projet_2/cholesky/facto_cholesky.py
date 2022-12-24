import numpy as np
from math import sqrt
from matrix_generator import *

def cholesky_dense(A):
    T = np.zeros(A.shape)
    for j in range(len(A)):
        for i in range(len(A)):
            somme = 0
            for k in range(i):
                somme += T[i][k] * T[j][k]
            if (j == i):
                T[i][i] = np.sqrt(A[i][i] - somme)
            if (j > i):
                T[j][i] = (A[i][j] - somme) / T[i][i]
    return T

def cholesky_incomplete(A):
    T = np.zeros(A.shape)
    for j in range(len(A)):
        for i in range(len(A)):
            if (A[j][i] != 0):
                somme = 0
                for k in range(i):
                    somme += T[i][k] * T[j][k]
                if (j == i):
                    T[i][i] = np.sqrt(A[i][i] - somme)
                if (j > i):
                    T[j][i] = (A[i][j] - somme) / T[i][i]
    return T

def solve_cholesky(A,b):
    ct = b.shape
    b = b.flatten()
    T = cholesky_incomplete(A)
    Y = np.zeros(len(A))

    for i in range(len(A)):
        S = b[i]
        for j in range(i):
            S -= T[i][j] * Y[j]
        Y[i] = S / T[i][i]

    for i in range(len(A)-1, -1, -1):
        S = Y[i]
        for j in range(i+1,len(A)):
            S -= T[j][i] * Y[j]
        Y[i] = S / T[i][i]

    return Y.reshape(ct)
