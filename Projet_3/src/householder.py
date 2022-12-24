import numpy as np

""" Renvoie le vecteur U tel que HX = Y = (Id - 2UtU)X = Y """
def getSymVect(X, Y):
    U = X - Y
    return U / np.linalg.norm(U)

""" Renvoie U * tU pour U un vecteur """
def getProdTrans(U):
    n = len(U)
    return np.array([[ U[x] * U[y] for x in range(n)] for y in range(n)])

""" Renvoie la matrice H de HouseHolder pour HX=Y """
def getHH(X, Y):
    U = getSymVect(X, Y)
    n = len(U)
    I = np.eye(n)
    return I - 2 * getProdTrans(U)  

""" Effectue un produit optimisé de H.V """
def prodHHVect(X, Y, V):
    U = getSymVect(X, Y)
    n = len(U)
    p = sum([U[i] * V[i] for i in range(n)]) ## Il s'agit du produit tU.V
    return V - 2 * p * U

""" Effectue un produit optimisé de H.M """
def prodHHMat(X, Y, M):
    U = getSymVect(X, Y)
    n = len(U)
    m = len(M[0])
    R = [[0 for i in range(m)] for l in range(n)]
    for c in range(m):
        p = sum([U[k] * M[k][c] for k in range(n)])
        V = [M[k][c] for k in range(n)] - 2 * p * U
        for l in range(n):
            R[l][c] = V[l] 
    return R

def applyHHVect(i, N, tN, Y):
    n = len(Y[0])
    inter = np.zeros((1,n))
    inter[0:1,i+1:n] = 2*np.dot(Y[0:1,i+1:len(Y[0])], np.dot(N, tN))
    return Y - inter


def applyHH(i, X ,Y, M):
    N = getSymVect(X, Y)
    tN = np.transpose(N)
    L = len(M)
    C = len(M[0])
    res = np.zeros((L,C))
    for j in range(L):
        res[j:j+1,0:C] = applyHHVect(i, N, tN, M[j:j+1,0:C])
    #print(res[0])
    return res

'''
X = np.array([3, 4, 0])
Y = np.array([0, 0, 5])

print(getHH(X, Y))

print(prodHHVect(X, Y, [1, 2, 3]))
print(np.dot(getHH(X, Y), [1, 2, 3]))

print(prodHHMat(X, Y, [[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
print(np.dot(getHH(X, Y), [[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
'''
