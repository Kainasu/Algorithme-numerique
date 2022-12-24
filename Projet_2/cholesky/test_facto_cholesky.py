
import facto_cholesky as cho
import numpy as np
from math import sqrt
from matrix_generator import *  

def is_triangular_inf(A):
    for i in range(len(A)-1):
        for j in range(i+1, len(A)):
            if(A[i][j] != 0):
                return False
    return True

def test_cholesky_dense() :
    
    # A, B, C et D sont symétriques, définis positifs
    A = np.array([[1, 1, 1, 1],
                  [1, 5, 5, 5],
                  [1, 5, 14, 14],
                  [1, 5, 14, 15]])
                  
    T = cho.cholesky_dense(A)
    assert is_triangular_inf(T),  "TEST 1 CHOLESKY DENSE FAILED "
    A2 = np.dot(T,np.transpose(T))
    for j in range(len(A)):
            for i in range(len(A)):
                 assert abs( A[i][j] - A2[i][j] ) < 0.001,  "TEST 1 CHOLESKY DENSE FAILED "

    B = np.array([[4, 12, -16],
                [12, 37, -43],
                [-16, -43, 98 ]])
                
    T = cho.cholesky_dense(B)
    assert is_triangular_inf(T),  "TEST 2 CHOLESKY DENSE FAILED "
    B2 = np.dot(T,np.transpose(T))
    for j in range(len(B)):
        for i in range(len(B)):
                assert abs( B[i][j] - B2[i][j] ) < 0.001 ,  "TEST 2 CHOLESKY DENSE FAILED "

    C = np.array([[1.1, 1.2, 1.6, 1.7],
                  [1.2, 5.5, 5.5, 5.5],
                  [1.6, 5.5, 14, 14],
                  [1.7, 5.5, 14, 15]])
    
    T = cho.cholesky_dense(C)
    assert is_triangular_inf(T),  "TEST 3 CHOLESKY DENSE FAILED "
    C2 = np.dot(T,np.transpose(T))
    for j in range(len(C)): 
        for i in range(len(C)):
                assert abs( C[i][j] - C2[i][j] ) < 0.001,  "TEST 3 CHOLESKY DENSE FAILED "
    D = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
    
    T = cho.cholesky_dense(D)
    assert is_triangular_inf(T),  "TEST 4 CHOLESKY DENSE FAILED "
    D2 = np.dot(T,np.transpose(T))
    for j in range(len(D)): 
        for i in range(len(D)):
                assert abs( D[i][j] - D2[i][j] ) < 0.001,  "TEST 4 CHOLESKY DENSE FAILED "


def test_cholesky_incomplete(): 

    A = np.array([[1, 1, 1, 1],
                  [1, 5, 5, 5],
                  [1, 5, 14, 14],
                  [1, 5, 14, 15]])

    T = cho.cholesky_incomplete(A)
    assert is_triangular_inf(T),  "TEST 1 CHOLESKY INCOMPLETE FAILED "
    A2 = np.dot(T,np.transpose(T))
    for j in range(len(A)):
            for i in range(len(A)):
                 assert abs( A[i][j] - A2[i][j] ) < 0.001,  "TEST 1 CHOLESKY INCOMPLETE FAILED "

    B = np.array([[4, 12, -16],
                [12, 37, -43],
                [-16, -43, 98 ]])

    T = cho.cholesky_incomplete(B)
    assert is_triangular_inf(T),  "TEST 2 CHOLESKY INCOMPLETE FAILED "
    B2 = np.dot(T,np.transpose(T))
    for j in range(len(B)):
        for i in range(len(B)):
                assert abs( B[i][j] - B2[i][j] ) < 0.001,  "TEST 2 CHOLESKY INCOMPLETE FAILED "

    C = np.array([[1.1, 1.2, 1.6, 1.7],
                  [1.2, 5.5, 5.5, 5.5],
                  [1.6, 5.5, 14, 14],
                  [1.7, 5.5, 14, 15]])

    T = cho.cholesky_incomplete(C)
    assert is_triangular_inf(T),  "TEST 3 CHOLESKY INCOMPLETE FAILED "
    C2 = np.dot(T,np.transpose(T))
    for j in range(len(C)): 
        for i in range(len(C)):
                assert abs( C[i][j] - C2[i][j] ) < 0.001,  "TEST 3 CHOLESKY INCOMPLETE FAILED "
    D = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])

    T = cho.cholesky_incomplete(D)
    assert is_triangular_inf(T),  "TEST 4 CHOLESKY INCOMPLETE FAILED "
    D2 = np.dot(T,np.transpose(T))
    for j in range(len(D)): 
        for i in range(len(D)):
                assert abs( D[i][j] - D2[i][j] ) < 0.001,  "TEST 4 CHOLESKY INCOMPLETE FAILED "
    

def test_cholesky_solve(): 

    A = np.array([[1, 1, 1, 1],
                  [1, 5, 5, 5],
                  [1, 5, 14, 14],
                  [1, 5, 14, 15]])

    B = np.array([1, 2, 3, 4])

    C1 = cho.solve_cholesky(A,B)
    C2 = np.linalg.solve(A,B)

    for i in range(len(C1)):
        assert abs( C1[i] - C2[i] ) < 0.001,  "TEST 1 CHOLESKY SOLVE FAILED "

    
    A = np.array([[4, 12, -16],
                [12, 37, -43],
                [-16, -43, 98 ]])

    B = np.array([1, 2, 3])

    C1 = cho.solve_cholesky(A,B)
    C2 = np.linalg.solve(A,B)
    for i in range(len(C1)):
        assert abs( C1[i] - C2[i] ) < 0.001,  "TEST 2 CHOLESKY SOLVE FAILED "
    
    A = np.array([[1.1, 1.2, 1.6, 1.7],
                  [1.2, 5.5, 5.5, 5.5],
                  [1.6, 5.5, 14, 14],
                  [1.7, 5.5, 14, 15]])
    B = np.array([1, 2, 3, 4])

    C1 = cho.solve_cholesky(A,B)
    C2 = np.linalg.solve(A,B)
    for i in range(len(C1)):
        assert abs( C1[i] - C2[i] ) < 0.001,  "TEST 3 CHOLESKY SOLVE FAILED "
        
        
    A = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
    B = np.array([1, 2, 3, 4])

    C1 = cho.solve_cholesky(A,B)
    C2 = np.linalg.solve(A,B)
    for i in range(len(C1)):
        assert abs( C1[i] - C2[i] ) < 0.001,  "TEST 4 CHOLESKY SOLVE FAILED "
    


if __name__ == '__main__':
    test_cholesky_dense()
    print("TEST CHOLESKY DENSE SUCCESS")
    test_cholesky_incomplete()
    print("TEST CHOLESKY INCOMPLETE SUCCESS")
    test_cholesky_solve()
    print("TEST CHOLESKY SOLVE SUCCESS")