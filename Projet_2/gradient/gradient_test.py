from gradient_conj import *

def test_gradient_conj():
    
    x = np.zeros([4,1])
    A = np.array([[1, 1, 1, 1],
                  [1, 5, 5, 5],
                  [1, 5, 14, 14],
                  [1, 5, 14, 15]])

    b = np.array([[8], [10], [4], [10]])
    
    Solve = np.linalg.solve(A, b)
    Grad = conjGrad(A, b, x)
    
    for i in range(len(A)):
        assert abs( Solve[i][0] - Grad[i][0] ) < 0.001,  "TEST 1 GRADIENT CONJUGUE FAILED "

    B = np.array([[1.1, 1.2, 1.6, 1.7],
                  [1.2, 5.5, 5.5, 5.5],
                  [1.6, 5.5, 14, 14],
                  [1.7, 5.5, 14, 15]])

    Solve2 = np.linalg.solve(B, b)
    Grad2 = conjGrad(B, b, x)
    
    
    for i in range(len(B)):
        assert abs( Solve2[i][0] - Grad2[i][0] ) < 0.001,  "TEST 2 GRADIENT CONJUGUE FAILED "
                
    C = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])

    Solve3 = np.linalg.solve(C, b)
    Grad3 = conjGrad(C, b, x)

    for i in range(len(C)):
        assert abs( Solve3[i][0] - Grad3[i][0] ) < 0.001,  "TEST 3 GRADIENT CONJUGUE FAILED "

def test_gradient_conj_with_precond():
    
    x = np.zeros([4,1])
    A = np.array([[1, 1, 1, 1],
                  [1, 5, 5, 5],
                  [1, 5, 14, 14],
                  [1, 5, 14, 15]])

    b = np.array([[8], [10], [4], [10]])
    
    Solve = np.linalg.solve(A, b)
    Grad = conjGradPrecond(A, b, x)

    for i in range(len(A)):
        assert abs( Solve[i][0] - Grad[i][0] ) < 0.001,  "TEST 1 GRADIENT CONJUGUE AVEC PRECONDITIONNEUR FAILED "

    B = np.array([[1.1, 1.2, 1.6, 1.7],
                  [1.2, 5.5, 5.5, 5.5],
                  [1.6, 5.5, 14, 14],
                  [1.7, 5.5, 14, 15]])

    Solve2 = np.linalg.solve(B, b)
    Grad2 = conjGradPrecond(B, b, x)
    
    
    for i in range(len(B)):
        assert abs( Solve2[i][0] - Grad2[i][0] ) < 0.001,  "TEST 2 GRADIENT CONJUGUE AVEC PRECONDITIONNEUR FAILED "
                
    C = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])

    Solve3 = np.linalg.solve(C, b)
    Grad3 = conjGradPrecond(C, b, x)
 
    for i in range(len(C)):
        assert abs( Solve3[i][0] - Grad3[i][0] ) < 0.001,  "TEST 3 GRADIENT CONJUGUE AVEC PRECONDITIONNEUR FAILED "

if __name__ == '__main__':
    test_gradient_conj()
    print("TEST GRADIENT CONJUGUE SUCCESS")
    test_gradient_conj_with_precond()
    print("TEST GRADIENT CONJUGUE AVEC PRECONDITIONNEUR SUCCESS")
