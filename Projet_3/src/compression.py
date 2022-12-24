import numpy as np
import matplotlib.pyplot as plt
from QR import *
from bidiagonale import *

'''Renvoie une liste contenant les éléments de la diagonale de  S positive et décroissante et U en conséquence'''
def apply_propriety(U, S):
    n, m = S.shape
    diag = []
    result = np.dot(U,S) 
    for i in range(min(n,m)): #Met dans  diag tous les éléments diagonaux de S
        diag.append(S[i][i])
    for i in range(len(diag)): #Met les éléments diagonaux positifs
        if (diag[i] < 0):
            diag[i] *= -1
    diag.sort(reverse=True) # Trie les éléments diagonaux dans l'ordre décroissant
    S_invTri = np.zeros(S.shape)
    for i in range(min(n,m)):
        S_invTri[i][i] = 1/diag[i]     
    new_U = np.dot(result, S_invTri)
    return new_U, diag

''' Retourne la décomposition SVD de la matrice M. S est renvoyé comme un vecteur contenant les éléments (mis positifs) de la diagonale et trié dans l'ordre décroissant '''
def svd_decomposition(M, NMax):
    Qleft, BD, Qright = bidiagonale(M)
    U, S, V = firstQR(BD, NMax)
    U, S = apply_propriety(U, S)
    return np.dot(Qleft, U), S, np.dot(V, Qright)


#Returns 3 matrix corresponding of color matrix of the picture given in parameter
def color_extraction(image):
    n, m = image.shape[0:2]
    R = np.zeros([n,m])
    G = np.zeros([n,m])
    B = np.zeros([n,m])
    for i in range(n):
        for j in range(m):
            R[i][j] = image[i][j][0]
            G[i][j] = image[i][j][1]
            B[i][j] = image[i][j][2]
    return R, G, B


#Returns a k-rank compressed picture of picture A
def compression_rank(k, A, NMax):
    n = A.shape[0]
    R, G, B = color_extraction(A)
    Ru, Rs, Rv = svd_decomposition(R,NMax)
    Gu, Gs, Gv = svd_decomposition(G,NMax)
    Bu, Bs, Bv = svd_decomposition(B,NMax)
    for i in range(k, n):
        Rs[i] = 0
        Gs[i] = 0
        Bs[i] = 0

    Rus = np.zeros(Ru.shape)
    Gus = np.zeros(Ru.shape)
    Bus = np.zeros(Ru.shape)
    for i in range(n):
        for j in range(k):
            Rus[i][j] = Ru[i][j]*Rs[j]
            Gus[i][j] = Gu[i][j]*Gs[j]
            Bus[i][j] = Bu[i][j]*Bs[j]
    R_compressed = np.dot(Rus, Rv)
    G_compressed = np.dot(Gus, Gv)
    B_compressed = np.dot(Bus, Bv)
    image = np.zeros(A.shape)
    image[:,:,0] = R_compressed
    image[:,:,1] = G_compressed
    image[:,:,2] = B_compressed
    return image

#Returns a k-rank compressed picture of picture A using numpy's svd
def compression_rank_with_numpy(k, A):
    n = A.shape[0]
    R, G, B = color_extraction(A)
    Ru, Rs, Rv = np.linalg.svd(R)
    Gu, Gs, Gv = np.linalg.svd(G)
    Bu, Bs, Bv = np.linalg.svd(B)
    for i in range(k, n):
        Rs[i] = 0
        Gs[i] = 0
        Bs[i] = 0

    Rus = np.zeros(Ru.shape)
    Gus = np.zeros(Ru.shape)
    Bus = np.zeros(Ru.shape)
    for i in range(n):
        for j in range(k):
            Rus[i][j] = Ru[i][j]*Rs[j]
            Gus[i][j] = Gu[i][j]*Gs[j]
            Bus[i][j] = Bu[i][j]*Bs[j]
    R_compressed = np.dot(Rus, Rv)
    G_compressed = np.dot(Gus, Gv)
    B_compressed = np.dot(Bus, Bv)
    image = np.zeros(A.shape)
    image[:,:,0] = R_compressed
    image[:,:,1] = G_compressed
    image[:,:,2] = B_compressed
    return image


def test_apply_propriety():

### TEST 1 ###
    U = np.array([[1,2,3],[5,4,6],[7,8,9]])
    S = np.array([[8,0,0],[0,1,0],[0,0,5]])
    new_U, diagS = apply_propriety(U,S)
    new_S = np.zeros(S.shape)
    n,m = S.shape
    for i in range(min(n,m)):
        new_S[i][i] = diagS[i]
    US = np.dot(U,S)
    new_US = np.dot(new_U,new_S)
    
    for i in range(len(diagS)):
        assert diagS[i] >= 0, "ELEMENTS OF DIAGONAL ARE ALL POSITIVE FAILED"      
        if (i < len(S)-1):
            assert diagS[i] >= diagS[i+1], "ELEMENTS SORTED IN DECREASING ORDER FAILED"
    for i in range(US.shape[0]):
            for j in range(US.shape[1]):
                 assert abs( US[i][j] - new_US[i][j] ) < 1e-12,  "TEST APPLY PROPRIETY ON U and S FAILED"

### TEST 2 ###
    U = np.array([[1,2,3],[5,4,6],[7,8,9]])
    S = np.array([[-8,0,0],[0,-1,0],[0,0,5]])
    new_U, diagS = apply_propriety(U,S)
    new_S = np.zeros(S.shape)
    n,m = S.shape
    for i in range(min(n,m)):
        new_S[i][i] = diagS[i]
    US = np.dot(U,S)
    new_US = np.dot(new_U,new_S)
    
    for i in range(len(diagS)):
        assert diagS[i] >= 0, "ELEMENTS OF DIAGONAL ARE ALL POSITIVE FAILED"      
        if (i < len(S)-1):
            assert diagS[i] >= diagS[i+1], "ELEMENTS SORTED IN DECREASING ORDER FAILED"
    for i in range(US.shape[0]):
            for j in range(US.shape[1]):
                 assert abs( US[i][j] - new_US[i][j] ) < 1e-12,  "TEST APPLY PROPRIETY ON U and S FAILED"

### TEST 3 ###
    U = np.array([[1,2,3],[5,4,6]])
    S = np.array([[-8,0,0],[0,-1,0],[0,0,5]])
    new_U, diagS = apply_propriety(U,S)
    new_S = np.zeros(S.shape)
    n,m = S.shape
    for i in range(min(n,m)):
        new_S[i][i] = diagS[i]
    US = np.dot(U,S)
    new_US = np.dot(new_U,new_S)
    
    for i in range(len(diagS)):
        assert diagS[i] >= 0, "ELEMENTS OF DIAGONAL ARE ALL POSITIVE FAILED"      
        if (i < len(S)-1):
            assert diagS[i] >= diagS[i+1], "ELEMENTS SORTED IN DECREASING ORDER FAILED"
    for i in range(US.shape[0]):
            for j in range(US.shape[1]):
                 assert abs( US[i][j] - new_US[i][j] ) < 1e-12,  "TEST APPLY PROPRIETY ON U and S FAILED"

### TEST 4 ###
    U = np.array([[1,2,3],[5,4,6]])
    S = np.array([[8,0,0],[0,1,0],[0,0,5]])
    new_U, diagS = apply_propriety(U,S)
    new_S = np.zeros(S.shape)
    n,m = S.shape
    for i in range(min(n,m)):
        new_S[i][i] = diagS[i]
    US = np.dot(U,S)
    new_US = np.dot(new_U,new_S)
    
    for i in range(len(diagS)):
        assert diagS[i] >= 0, "ELEMENTS OF DIAGONAL ARE ALL POSITIVE FAILED"      
        if (i < len(S)-1):
            assert diagS[i] >= diagS[i+1], "ELEMENTS SORTED IN DECREASING ORDER FAILED"
    for i in range(US.shape[0]):
            for j in range(US.shape[1]):
                 assert abs( US[i][j] - new_US[i][j] ) < 1e-12,  "TEST APPLY PROPRIETY ON U and S FAILED"

if __name__ == "__main__":
    
    test_apply_propriety()
    print("TEST APPLY PROPRIETY ON U and S SUCCES")

    img_full = plt.imread("fusee.png")
    rank = 10
    img_compressed = compression_rank(rank, img_full, 150)
    plt.title('Compression au rang ' +str(rank))
    plt.savefig('fusee_rang_' +str(rank))
    plt.show()

    

            

