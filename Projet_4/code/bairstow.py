import numpy as np
np.random.seed(1234)

def bairstow(poly1,poly2,h):
    i = 10 
    w = 10
    a = poly2[1]
    b = poly2[2]
    n = len(poly1)-1 #polynomial's degree
    (q,r)=np.polydiv(poly1,poly2)
    while (abs(i) > h or abs(w) > h):
        if (len(r) == 1):
            r= np.append([[0]], r)
        newpoly1 = np.concatenate((q, r))
        (q1,r1)=np.polydiv(newpoly1,poly2)
        newnewpoly1 = np.concatenate((q1, r1))
        A = np.array([[newnewpoly1[n-2],newnewpoly1[n-3]], [(newnewpoly1[n-1]-newpoly1[n-1]),newnewpoly1[n-2]]]) 
        B = np.array([newpoly1[n-1],newpoly1[n]])
        x = np.linalg.solve(A, B) # solution of linear equation
        a += x[0] #new a's value with a correction
        b += x[1] #new b's value with a correction
        poly2 = [1,a,b] 
        (q,r)=np.polydiv(poly1,poly2) # Gb is the quotient of the euclidian division and Vb the rest
        i = r[0] 
        if (len(r) ==1 ):
            w = r[0]
        else :
            w = r[1]
    return (poly2,q)

#(poly,Q) = bairstow([1,0,1,0],[1,60,-9], 0.001) #(x−i)(x+i)x
#print(Qx)
#(poly3,qx) = bairstow(Qx,[1,60,-9], 0.001)
#poly3 = bairstow([1,-2,-13,14,24], [1,1,-2], 0.001)

def quadratic_formula(poly):
    a = poly[0]
    b = poly[1]
    c = poly[2]
    dis = b*b -4*a*c
    if (dis > 0):
        x1 = (-b - np.sqrt(dis))/(2*a)
        x2 = -b/a -x1
        return x1,x2
    else :
        if (dis == 0):
            return -b/(2*a)
        else :
            dis = -dis
            x1 = complex(-b/(2*a),-np.sqrt(dis)/(2*a))
            x2 = np.conj(x1)
            return x1,x2
            

def root(poly,h=1e-7):
    a = 1
    b = 2
    n = len(poly)-1
    tab = []
    if (n <= 0):
        return []
    else :
        while (n > 2):
            (poly2,R) = bairstow(poly,[1,a,b],h)
            tab.extend(quadratic_formula(poly2))
            n = n-2
            poly=R
        if (n ==1):
            tab.extend([-poly[1]/poly[0]])
        elif( n ==2):
            tab.extend(quadratic_formula(poly))
        return tab

#tab = root([1,-5,-1,19,-2,24,0],0.001) #(x²+1)x(3-x)(4-x)(x+2)
tab1 = root([16,-20,-208,452,-240])

#generate a polynome of degree 'degree' with only integer coefficients    
def polynome_generator_int(degree, min_coef=-100, max_coef=100):
    return np.random.randint(min_coef, max_coef, size=degree+1)

#test functions roots
def test_roots_int_coeff():
    for degree in range(11):
        x = polynome_generator_int(degree)
        roots = sorted(root(x), key = lambda x : abs(x) + x.imag)
        numpy_roots = sorted(np.roots(x), key = lambda x :abs(x) + x.imag)
        print("Polynome of degree:", degree, "\n",np.poly1d(x))
        print("roots :", roots)
        
        assert len(root(x)) == degree, "TEST RIGHT NUMBER OF ROOTS : FAILED"
        
        np.testing.assert_array_almost_equal(roots, numpy_roots, err_msg='TEST RIGHT VALUE OF ROOTS : FAILED', decimal=6)
    print("\n")    
    print("TEST RIGHT NUMBER OF ROOTS : SUCCESS")
    print("TEST RIGHT VALUE OF ROOTS : SUCCESS")
    


if __name__ == '__main__':

    test_roots_int_coeff()

   
    print("Test racine multiple\nPolynome : (x-1)^2*(x+1) ")
    poly = [1,-1,-1,1]
    roots = root(poly)
    numpy_roots = np.roots(poly)
    print("roots :", roots)
    #Racine multiple pas très fonctionnel
    """
    poly = [1,-2,-2,8,-7,2]#(x-1)^4*(x+2)
    roots = root(poly)
    numpy_roots = np.roots(poly)
    print("roots :", roots)
    """
   
   

