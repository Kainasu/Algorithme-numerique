import numpy as np 
import copy
import householder
import QR

#A de dimension n*m et retourne une matrice bidiagonale


def scale(A, p): # for square matrix
    R = np.eye(p)
    (n, m) = np.shape(A)
    for i in range(n):
        for j in range(m):
            R[p-i-1, p-j-1] = A[n-i-1, m-j-1]
    return R


#A is a n*m matrix and the function returns 3 matrix such as A = Qleft*BD*Qright and BD is a bidiagonal matrix
def bidiagonale(A):
    n,m  = np.shape(A)
    Qleft = np.eye(n)
    Qright= np.eye(m)
    BD = copy.deepcopy(A)
    for i in range(n-1):
        X1 = np.array(BD[i:n, i])
        Y1 = np.zeros(n-i)#((n-i, 1))
        Y1[0] = np.linalg.norm(X1)
        Q1 = householder.getHH(X1, Y1)
        Q1 = scale(Q1, n)
        Qleft= np.dot(Qleft, Q1)
        BD = np.dot(Q1, BD)
        if(not(i == (m-2))):
            X2 = np.array(BD[i, (i+1):m])
            Y2 = np.zeros(m-i-1)#((m-i, 1))
            Y2[0] = np.linalg.norm(X2)
            Q2 = householder.getHH(X2, Y2)
            Q2 = scale(Q2, m)
            Qright = np.dot(Q2, Qright)
            BD = np.dot(BD, Q2) 
    return(Qleft, BD, Qright)


#a function which tests if A*B*C = D in term of isNormZero function
def test_prod(A, B, C, D, seuil): 
    return QR.isNormZero(np.dot(np.dot(A, B), C)-D, seuil)

#a function which returns true only if Qleft*BD*Qright = A for each iteration of the loop
def test_inv_boucle(A, seuil):
    n,m  = np.shape(A)
    Qleft = np.eye(n)
    Qright= np.eye(m)
    BD = copy.deepcopy(A)
    for i in range(n-1):
        X1 = np.array(BD[i:n, i])
        Y1 = np.zeros(n-i)#((n-i, 1))
        Y1[0] = np.linalg.norm(X1)
        Q1 = householder.getHH(X1, Y1)
        Q1 = scale(Q1, n)
        Qleft= np.dot(Qleft, Q1)
        BD = np.dot(Q1, BD)
        if(not(i == (m-2))):
            X2 = np.array(BD[i, (i+1):m])
            Y2 = np.zeros(m-i-1)#((m-i, 1))
            Y2[0] = np.linalg.norm(X2)
            Q2 = householder.getHH(X2, Y2)
            Q2 = scale(Q2, m)
            Qright = np.dot(Q2, Qright)
            BD = np.dot(BD, Q2)
        if(not(test_prod(Qleft, BD, Qright, A, seuil))):
            return False
    return True

#créer plein de matrices, et appeler avec isbidi (marche pas avec les rectangles)

#a function which returns a random n*n matrix
def matrix_generate(n):
    M = np.random.rand(n,n)
    M = 100*M
    return M

#a function which create N n² matrix, then bidiagonalize them and finally tests whether such matrix are bidiagonals
def test_bidiag(N,n,seuil):
    for i in range(N):
        A,B,C = bidiagonale(matrix_generate(n))
        if(not(QR.isBD(B, seuil))):
            return False
    return True

###TESTS

if __name__ == "__main__":
    M = np.array([[1,12,3,1],[44,5,6,1],[7,58,9,1],[10,111,12,1]])  

    print("test the invariance of Qleft * BD * Qright = A  ...")
    if(test_inv_boucle(M,1E-13)):
        print("PASSED")
    else:
        print("FAILED")

    print("test the return value is a bidiagonal matrix ...")
    if(test_bidiag(10, 2, 1E-13)):
        print("PASSED")
    else:
        print("FAILED")

#print(test_bidiag(10, 2, 1E-14))
#print(test_inv_boucle(M,1E-14))#marche pour 1E-13 mais plus pour 1E-14


#print(isBD(bidiagonale(M), 0.001))
#print(test_bidiag(10,3, 0.001))




#A,B,C = bidiagonale(M)
#print(A)
#print(B)
#print(C)
