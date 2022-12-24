import numpy as np
from matplotlib import pyplot as plt
from newtonRaphson import *
#We are here in 2 dimensions

#This function returns the elastic force exercised by a spring on a object of coordinates U(x, y).
#The attach point of the pring is O(x0, y0), the spring konstant is k and its natural length is 0.
def fEl(O, k): 
    return lambda U : np.array([k*(O[0] - U[0]), k*(O[1] - U[1])])   

# Returns the Jacobian of the elastic force function with parameters (xO, yO), k
def jEl(O, k):
    return lambda U : np.array([
                        [-k, 0],
                        [0, -k]
                    ])
# Returns the centrifugal force function with parameters (xO, yO), k
def fC(O, k) : 
    return lambda U : np.array([k*(U[0] - O[0]), k*(U[1] - O[1])])  

# Returns the Jacobian of the centrifugal force function with parameters (xO, yO), k
def jC(O, k) : 
    return lambda U : np.array([
                    [k, 0],
                    [0, k]
                ])
        
# Returns the gravitational force function with parameters (xO, yO), k
def fG(O, k):
    return lambda U : np.array([-k*((U[0] - O[0])/(((U[0] - O[0])**2 + (U[1] - O[1])**2)**(3/2))),
                                 -k*((U[1] - O[1])/(((U[0] - O[0])**2 + (U[1] - O[1])**2)**(3/2)))])

# Returns the Jacobian of the gravitational force function with parameters (xO, yO), k
def jG(O, k):
    return lambda U : np.array([
        [-k*(((U[0] - O[0])**2 + (U[1] - O[1])**2)**(-3/2) - 3*(U[0] - O[0])**2*(((U[0] - O[0])**2 + (U[1] - O[1])**2)**(-5/2))),
        3*k*(U[0] - O[0])*(U[1] - O[1])*(((U[0] - O[0])**2 + (U[1] - O[1])**2)**(-5/2))],
        [3*k*(U[0] - O[0])*(U[1] - O[1])*(((U[0] - O[0])**2 + (U[1] - O[1])**2)**(-5/2)),
        -k*(((U[0] - O[0])**2 + (U[1] - O[1])**2)**(-3/2) - 3*(U[1] - O[1])**2*(((U[0] - O[0])**2 + (U[1] - O[1])**2)**(-5/2)))]
    ])

# Returns the sum of the force applied on point U according to parameters
def forces(U, kA, kB, kC, xA, xB, yA, yB):
    A = np.array([xA, yA])
    B = np.array([xB, yB])
    bary = np.array([(A[0] * 1 + B[0] * 0.01) / (1 + 0.01), 
                    (A[1] * 1 + B[1] * 0.01) / (1 + 0.01),])

    return np.array([
                    fG(A, kA)(U)[0] + fG(B, kB)(U)[0] + fC(bary, kC)(U)[0],
                    fG(A, kA)(U)[1] + fG(B, kB)(U)[1] + fC(bary, kC)(U)[1]
                    ])

# Returns a specialization of forces()
def make_forces(kA, kB, kC, xA, xB, yA, yB):
    return lambda U : forces(U, kA, kB, kC, xA, xB, yA, yB)

# Returns the Jacobian of the force sum applied on point U according to parameters
def jacob(U, kA, kB, kC, xA, xB, yA, yB):
    A = np.array([xA, yA])
    B = np.array([xB, yB])

    bary = np.array([(A[0] * 1 + B[0] * 0.01) / (1 + 0.01), 
                    (A[1] * 1 + B[1] * 0.01) / (1 + 0.01),])

    return np.array([
                    [jG(A, kA)(U)[0][0] + jG(B, kB)(U)[0][0] + jC(bary, kC)(U)[0][0], jG(A, kA)(U)[0][1] + jG(B, kB)(U)[0][1] + jC(bary, kC)(U)[0][1]],
                    [jG(A, kA)(U)[1][0] + jG(B, kB)(U)[1][0] + jC(bary, kC)(U)[1][0], jG(A, kA)(U)[1][1] + jG(B, kB)(U)[1][1] + jC(bary, kC)(U)[1][1]]  
                    ])

# Returns a specialization of jacob()
def make_jacob(kA, kB, kC, xA, xB, yA, yB):
    return lambda U : jacob(U, kA, kB, kC, xA, xB, yA, yB)

# find the Lagrangian points with two masses, one at (0, 0) and a k = 1 and
# the other ar (1, 0) an a k = 0.01. The coefficient of the centrifugal force
# applied to the point is k = 1. Try to find a solution on all points which are 
# in the rectangle (x0, y0) (top left corner) (x1, y1) (bottom right corner). The
# step represents the distance which separate each tested point. 
def example1(step, x0, x1, y0, y1, iter_count,eps):
    forces_p = make_forces(1, 0.01, 1, 0, 1, 0, 0)
    jacob_p = make_jacob(1, 0.01, 1, 0, 1, 0, 0)
    for i in [x0+k*step+0.01  for k in range(int((x1-x0)/step))]:
        for j in [y0+k*step+0.01 for k in range(int((y1-y0)/step))]:
            U = np.array([i, j])
            X = NewtonRaphsonBT(forces_p, jacob_p, U, iter_count, eps)
            #plt.plot(i, j, "o")
            if (X[1] == True):
                plt.plot(np.array([X[0][0]]), np.array(X[0][1]), "o")     
    plt.scatter(0, 0, c='orange', marker='*', s=300)
    plt.scatter(1, 0, c='blue', marker='*', s=150)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

#draws the tested points in example1(step, x0, x1, y0, y1, ...) 
def example1_testedpoints(step, x0, x1, y0, y1):
    for i in [x0+k*step+0.01  for k in range(int((x1-x0)/step))]:
        for j in [y0+k*step+0.01 for k in range(int((y1-y0)/step))]:
            U = np.array([i, j])
            plt.plot(np.array([U[0]]), np.array(U[1]), "o")  
    plt.scatter(0, 0, c='orange', marker='*', s=300)
    plt.scatter(1, 0, c='blue', marker='*', s=150)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

# find the Lagrangian points with two masses, one at (0, 0) and a k = 1 and
# the other ar (1, 0) an a k = 0.01. The coefficient of the centrifugal force
# applied to the point is k = 1. Try to find solution only testing some points at
# positions near to a Lagrangian point.  
def example1Opti(n, iter_count, eps):
    forces_p = make_forces(1, 0.01, 1, 0, 1, 0, 0)
    jacob_p = make_jacob(1, 0.01, 1, 0, 1, 0, 0)
    #The coordinates of the two masses
    A = np.array([0, 0])
    B = np.array([1, 0])
    #Two vectors
    u = B - A
    v = np.array([u[1], u[0]])
    bary = np.array([(A[0] * 1 + B[0] * 0.01) / (1 + 0.01), 
                    (A[1] * 1 + B[1] * 0.01) / (1 + 0.01),])
    # symetric of the barycenter according to the first object
    S1 = A - (bary - A)
    # symetric of the barycenter according to the second object
    S2 = B - (bary - B)
    #The coordinates of the tested points
    P = ([A - v, A - u - v, A - u, A - u + v, A + v, B + v, B + u + v, B + u, B + u - v, B - v, bary, S1, S2])
    for i in range(13):
        X = NewtonRaphsonBT(forces_p, jacob_p, P[i], iter_count, eps)
        if(X[1] == True):
            plt.plot(np.array([X[0][0]]), np.array(X[0][1]), "o")
    start = A - u
    norm_u = int(np.linalg.norm(u))
    for i in range(n):
        X = NewtonRaphsonBT(forces_p, jacob_p, start + i * (3 *u / n) , iter_count, eps)
        if(X[1] == True):
            plt.plot(np.array([X[0][0]]), np.array(X[0][1]), "o")
    plt.scatter(0, 0, c='orange', marker='*', s=300)
    plt.scatter(1, 0, c='blue', marker='*', s=150)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

#draws the tested points in example1Opti(n, ...) 
def example1Opti_testedpoints(n):
    plt.scatter(0, 0, c='orange', marker='*', s=300)
    plt.scatter(1, 0, c='blue', marker='*', s=150)
    A = np.array([0, 0])
    B = np.array([1, 0])
    u = B - A
    v = np.array([u[1], u[0]])
    start = A - u
    for i in range(n):
        plt.plot(np.array((start + i * (3 *u / n))[0]), np.array((start + i * (3 *u / n))[1]), "ro")

    bary = np.array([(A[0] * 1 + B[0] * 0.1) / (1 + 0.1), 
                    (A[1] * 1 + B[1] * 0.1) / (1 + 0.1),])

    P = ([A - v, A - u - v, A - u, A - u + v, A + v, B + v, B + u + v, B + u, B + u - v, B - v])

    for i in range(10):
        plt.plot(np.array(P[i][0]), np.array(P[i][1]), "bo")
    
    plt.plot(np.array(bary[0]), np.array(bary[1]), "go")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

if __name__ == '__main__':  
    print("\nVerification of the function correction \n")
    U = np.array([1.5, 0])
    print("f((1.5, 0)) = \n")
    forces_p = make_forces(1, 0.01, 1, 0, 1, 0, 0)
    print(forces_p(U))
    print("\nJ((1.5, 0)) = \n")
    jacob_p = make_jacob(1, 0.01, 1, 0, 1, 0, 0)
    print(jacob_p(U))
    print("\nOne of the Jacobian point\n")
    print(NewtonRaphsonBT(forces_p, jacob_p, U, 100, 0.01))
    print("\nExample non-optimized...")
    example1(0.1,  -1.5, 1.5, -1.5, 1.5, 1000, 0.0001)
    print("done\n")
    print("Tested points in the non-optimized example...")
    example1_testedpoints(0.1,  -1.5, 1.5, -1.5, 1.5)
    print("done\n")
    print("Example optimized...")
    example1Opti(80, 1000, 0.0001)
    print("done\n")
    print("Tested points in the optimized example...")
    example1Opti_testedpoints(20)
    print("done\n")




