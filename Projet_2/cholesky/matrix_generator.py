import numpy as np
import random

#make a sparse symmetric positive definite Matrix of size n and with p extra diagonal elements

def generate_combinaison(n):
    combi = []
    for i in range(1, n):
        for j in range(i):
            combi.append((i, j))
    return combi
    
def is_positive_definite(A):
    for i in range(len(A) + 1):
        if (np.linalg.det(A[0:i, 0:i]) <= 0):
            return False
    return True
        
def make_SSPD_matrix(n, p):
    p2 = p
    if (p > np.floor(n * (n - 1) / 2)):
        print("Try a value of p <= ", np.floor(n*(n-1)/2)) 
        return exit(1)
    reminder = p
    A = np.eye(n,n)
    combi = generate_combinaison(n)
    iter = 0
    for i in range(n):
        A[i][i] = random.randint(1,100)
    while (p > 0):
        x, y = combi.pop(random.randint(0, len(combi)-1))
        value = 0
        while (value == 0):
            value = random.randint(-50,50)
        A[x][y] = value
        A[y][x] = value
        if not is_positive_definite(A):
            A[x][y] = 0
            A[y][x] = 0
            combi.append((x,y))
        else:
            p = p-1 
        iter += 1
        if(iter > 10 * n):
            return  make_SSPD_matrix(n, p2);     
    return A
    
        
if __name__ == '__main__':
    n = int(input("Enter the length of the matrix : \n"))
    p = int(input("Enter the number of extra diagonal elements : \n"))
    A = make_SSPD_matrix(n, p)
    print(A)
    print("A est définie positive ? ", is_positive_definite(A))
