import numpy as np 
import matplotlib.pyplot as plt
import random as r

# a function that creates square n*n matrix
def matrixGenerator(n):
    R = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
                R[i,j] = r.random()*100 + 1 # generate a random number between 1 and 100
    return R

# a function that creates square n*n bidiagonal matrix
def bidiagonalGenerator(n):
    R = np.zeros((n, n))
    for i in range(n):
        R[i,i] = r.random()*100 + 1 # generate a random number between 1 and 100
        if(i < n-1):
            R[i,i+1] = r.random()*100 + 1
    return R

# a function that returns the QR transformation by using the numpy function to 
# obtain the bidiagonal matrix.
def firstQR(A, NMax):
    (n, m) = np.shape(A)
    U = np.eye(m)
    V = np.eye(n)
    S = A
    #S = bd.bidiagonale(A)
    for i in range(NMax):
        (Q1, R1) = np.linalg.qr(np.matrix.transpose(S))
        (Q2, R2) = np.linalg.qr(np.matrix.transpose(R1))
        S = R2
        U = np.dot(U, Q2)
        V = np.dot(np.matrix.transpose(Q1), V)
        print(S)
        print(R1)
        print(R2)
    return (U, S, V)

# a function that returns the minimum value between a and b. 
def min(a ,b):
    if(a < b):
        return a
    return b

# a function that takes a matrix and an array, the function sum the absolute value of the off-diagonal 
# elements considered non-equal to 0. The result divided by the number of non-zero 
# off-diagonal elements and append to the t array (in order to show it as a graph). 
# valueThreshold is a threshold to approximate a value at 0 and returnThreshold 
# is a threshold to approximate the matrix as diagonal even if it has some off-diagonals non-zero terms.

def isDiagonal(A, t, valueThreshold, returnThreshold):
    (n,m) = np.shape(A)
    extraElmts = 0
    sumElmts = 0
    for i in range(n):
        for j in range(m):
            if(j != i):
                if(np.abs(A[i,j]) > valueThreshold):
                    sumElmts += np.abs(A[i,j])
                    extraElmts += 1
    
    result = sumElmts / extraElmts
    
    if(result <= returnThreshold):
        t.append(0) 
    else:
        t.append(result)
    return t

# a function that returns an array containing the different values taken by 
# diagonalPorcent(S) during the whole program and draw it.
def convergenceS(A, NMax):
    vT = 1e-14
    oDT = 4
    (n, m) = np.shape(A)
    t = []
    U = np.eye(m)
    V = np.eye(n)
    S = A
    #S = bidiagonale2.bidiagonale(A)
    t = isDiagonal(S, t, vT, oDT) 
    for i in range(NMax):
        (Q1, R1) = np.linalg.qr(np.matrix.transpose(S))
        (Q2, R2) = np.linalg.qr(np.matrix.transpose(R1))
        S = R2
        U = np.dot(U, Q2)
        V = np.dot(np.matrix.transpose(Q1), V)
        t = isDiagonal(S, t, vT, oDT) 
    plt.plot(t)
    plt.ylabel('sum of non-null off-diagonal elements divided by their number')
    plt.xlabel('iteration of QR factorization')
    #plt.show()
    return t

# a function that returns True if A = B and False otherwise. 
### not used because using the norm is a better method ###
def equalMatrix(A, B):
    (nA, mA) = np.shape(A)
    (nB, mB) = np.shape(B)
    if(nA != nB or mA != mB):
        return False
    else:
        for i in range(nA):
            for j in range(mA):
                if(A[i,j] != B[i,j]):
                    return False
        return True

# a function that returns True if the A matrix norm is equal to zero or about 
# (depends on the given threshold) and False otherwise
def isNormZero(A, threshold):
    (n,m) = np.shape(A)
    norm = 0
    for i in range(n):
        for j in range(m):
            norm += A[i,j] * A[i,j]
    norm = np.sqrt(norm) 
    if(norm <= threshold):
        return True
    return False

# a function that returns True if U x S x V = BD is an always verified loop 
# invariant and False otherwise
def USVBD(A, NMax, threshold):
    (n, m) = np.shape(A)
    U = np.eye(m)
    V = np.eye(n)
    S = A
    #S = bd.bidiagonale(A)
    BD = S
    product = np.dot(np.dot(U, S), V)
    invariant = isNormZero(product - BD, threshold) 
    for i in range(NMax):
        (Q1, R1) = np.linalg.qr(np.matrix.transpose(S))
        (Q2, R2) = np.linalg.qr(np.matrix.transpose(R1))
        S = R2
        U = np.dot(U, Q2)
        V = np.dot(np.matrix.transpose(Q1), V)
        product = np.dot(np.dot(U, S), V)
        invariant = isNormZero(product - BD, threshold) 
        if(invariant == False):
            return False
    return True

# a function that returns True if A is a bidiagonal matrix or about to (treshold) 
# and False otherwise.
def isBD(A, threshold):
    (n,m) = np.shape(A)
    for j in range(m):
        for i in range(n):
            # if you're not on the "bidiagonal" of the matrix
            if(i != j and j != i+1):
                if(np.abs(A[i,j]) > threshold):
                    return False
    return True

# a function that returns True if (S is bidiagonal) AND (R1 is bidiagonal) AND
# (R2 is bidiagonal) is an always verified loop invariant and False otherwise
def areBidiagonal(A, NMax, threshold):
    (n, m) = np.shape(A)
    U = np.eye(m)
    V = np.eye(n)
    S = A
    #S = bd.bidiagonale(A)
    for i in range(NMax):
        (Q1, R1) = np.linalg.qr(np.matrix.transpose(S))
        (Q2, R2) = np.linalg.qr(np.matrix.transpose(R1))
        S = R2
        U = np.dot(U, Q2)
        V = np.dot(np.matrix.transpose(Q1), V)
        invariant = isBD(S, threshold) and isBD(R1, threshold) and isBD(R2, threshold)
        if(invariant == False):
            return False 
    return True

### Tests ###

def test_convergence():
    for i in range(10):
        n = int(r.random() * 90 + 10) # integer between 10 and 100
        convergenceS(bidiagonalGenerator(n), 200)
    plt.show()   

def test_USVBD():
    for i in range(30):
        n = int(r.random() * 90 + 10) # integer between 10 and 100
        if(USVBD(bidiagonalGenerator(n), 200, 1e-11) == False): # 1e-11 minimum to PASS the test
            return False
    return True

def test_areBD():
    for i in range(30):
        n = int(r.random() * 90 + 10) # integer between 10 and 100
    if(areBidiagonal(bidiagonalGenerator(n), 200, 1e-120) == False): # threshold doesn't matter because the off-bidiagonal elements are never modified
            return False
    return True

if __name__ == "__main__":
    print("test the convergenceof S to a diagonal matrix ...")
    test_convergence()
    print("DONE")

    print("test the invariance of U*S*V = BD ...")
    if(test_USVBD()):
        print("PASSED")
    else:
        print("FAILED")

    print("test the invariance of S, R1 and R2 being bidiagonal matrix ...")
    if(test_areBD()):
        print("PASSED")
    else:
        print("FAILED")

