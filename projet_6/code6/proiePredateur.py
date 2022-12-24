import solving_method as sm
import matplotlib.pyplot as plt
import numpy as np

#Malthus Model
def simpleModel(gamma, meth, t0, tf, eps, y0):
    f = lambda t,N : gamma*N
    return sm.meth_epsilon(y0, t0, tf, eps, f, meth)

#Veruhlst Model
def difficultModel(gamma, kappa, meth, t0, tf, eps, y0):
    f = lambda t,N : gamma*N - (gamma*N*N)/kappa
    return sm.meth_epsilon(y0, t0, tf, eps, f, meth)

#Lotka-Volterra Model
def proiePredateur(a, b, c, d, meth, t0, tf, eps, y0):
    f = lambda t, y : np.array([y[0]*(a-b*y[1]), y[1]*(c*y[0]-d)]) 
    #return sm.meth_epsilon(y0, t0, tf, eps, f, meth) 
    T,Y,I = sm.meth_epsilon(y0, t0, tf, eps, f, meth)
    return T,Y,I

#Returns period -- to fix
def periodeProiePredateur(proie, preda, y0, t):
    t0 = t[0]
    k = 2
    eps = 0.01
    while np.absolute(proie[k] - y0[0]) > eps and np.absolute(preda[k] - y0[1]) > eps:
        k += 1
    return t[k] - t0

if __name__ == '__main__':

    #Parameters
    #gamma = -0.1
    gamma = 0.1
    kappa = 2
    meth = sm.step_RK4
    t0 = 0
    tf = 10
    eps = 0.01
    y0 = np.array(10)

    #Malthus Model
    t,y,index = simpleModel(gamma, meth, t0, tf, eps, y0)
    plt.plot(t, y, label='simpleMethod')
    plt.legend()
    plt.title("Malthus Model")
    plt.show()

    #Veruhlst Model
    t,y,index = difficultModel(gamma, kappa, meth, t0, tf, eps, y0)
    plt.plot(t, y, label='difficultMethod')
    plt.legend()
    plt.title("Veruhlst Model")
    plt.show()

    #Lotka-Volterra Model
    y0 = np.array([10,10])
    a,b,c,d = 10,2,3,5
    t,y,index = proiePredateur(a, b, c, d, meth, t0, tf, eps, y0)
    y1 = [x[0] for x in y]
    y2 = [x[1] for x in y]
    plt.plot(t, y1,label='N')
    plt.plot(t, y2, label='P')
    plt.title("Lotka-Volterra Model")
    plt.legend()
    plt.show()
    
    t,Y,index = proiePredateur(8,2,3,5, meth, t0, tf, eps, y0)
    y3 = [x[0] for x in Y]
    y4 = [x[1] for x in Y]
    """
    plt.plot(t, y3,label='N')
    plt.plot(t, y4, label='P')
    plt.legend()
    plt.show()
    """
    t,Y1,index = proiePredateur(8, 1, 4, 5, meth, t0, tf, eps, y0)
    y5 = [x[0] for x in Y1]
    y6 = [x[1] for x in Y1]
    """
    plt.plot(t, y5,label='N')
    plt.plot(t, y6, label='P')
    plt.legend()
    plt.show()
    """
    plt.plot(y1,y2,label='a=10,b=2,c=3,d=5')
    plt.plot(y3,y4,label='a=8,b=2,c=3,d=5')
    plt.plot(y5,y6, label='a=8,b=1,c=4,d=5')
    plt.title("Lotka-Volterra Model with differents values of a,b,c,d")
    plt.legend()
    plt.show()

    #Periods
    print(periodeProiePredateur(y1, y2, y0, t))
    print(periodeProiePredateur(y3, y4, y0, t))
    print(periodeProiePredateur(y5, y6, y0, t))
    
    #Local Behaviour
    for i in range(-10, 10,1):
        y0 = np.array([10+i*0.1, 10+i*0.1])
        t,y,index = proiePredateur(a, b, c, d, meth, t0, tf, eps, y0)
        y1 = [x[0] for x in y]
        y2 = [x[1] for x in y]
        plt.plot(y1,y2)
    plt.title("Local behaviour with an initial population (10,10)")
    plt.show()

    #Tangential_field
    f = lambda t, y : np.array([y[0]*(a-b*y[1]), y[1]*(c*y[0]-d)]) 
    sm.tangential_field_dim2(f, -10,20,1,-10,20,1, 'Tangential field of Lokta-Volterra equations' )
