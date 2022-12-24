from facto_cholesky import *
from matrix_generator import *
import numpy as np

if __name__ == '__main__':

    print("\nRandom sparse symmetric positive definite Matrix :")
    n = int(input("Enter the size of the matrix : \n"))
    p = int(input("Enter the number of extra diagonal elements : \n"))
        
    A = make_SSPD_matrix(n, p)
    print("A =\n", A)

    print("\nCholesky_incomplete")
    
    T_inc = cholesky_incomplete(A)
    print("T_incomplete =\n", T_inc)
    T_incInverse = np.linalg.inv(T_inc)
    print("T_incomplete_inverse =\n", T_incInverse)
    calcul = np.dot(np.transpose(T_incInverse), T_incInverse)
    print("transposée(T_incomplete_inverse)*T_incomplete_inverse =\n", calcul)
   

    
    print("\nCholesky_dense")
    
    T_inc2 = cholesky_dense(A)
    print("T_incomplete =\n", T_inc2)
    T_incInverse2 = np.linalg.inv(T_inc2)
    print("T_incomplete_inverse =\n", T_incInverse2)
    calcul2 = np.dot(np.transpose(T_incInverse2), T_incInverse2)
    print("transposée(T_incomplete_inverse)*T_incomplete_inverse =\n", calcul2)
    
    print("préconditionnement avec Cholesky densee :", np.linalg.cond(np.dot(calcul2, A)))
    print("préconditionnement avec Cholesky incomplete :", np.linalg.cond(np.dot(calcul, A)))
    print("préconditionnement de A :", np.linalg.cond(A))
