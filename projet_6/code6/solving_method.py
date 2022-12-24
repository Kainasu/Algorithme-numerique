import numpy as np
import math
import matplotlib.pyplot as plt

#Euler method

def step_euler(y,t,h,f):
    return y + h*f(t,y)

#middle point method
def step_middle(y,t,h,f):
    y_mid = y + (h/2)*f(t, y)
    P = f(t+h/2, y_mid)
    return y + h*P
    
#Heun method
def step_heun(y,t,h,f):
    P1 = f(t, y)
    y2 = y + h*P1
    P2 = f(t+h, y2)
    P = (P1 + P2)/2 
    return y + h*P

#Runge-Kutta order 4 method
def step_RK4(y,t,h,f):
    P1 = f(t, y)
    y2 = y + (h/2)*P1
    P2 = f(t+h/2, y2)
    y3 = y + (h/2) * P2
    P3 = f(t+h/2, y3)
    y4 = y + h*P3
    P4 = f(t+h, y4)
    P = (P1 + 2*P2 + 2*P3 + P4)/6
    return y + h*P

def meth_n_step(y0,t0,N,h,f,meth): #treat everything with array? 
    y = np.zeros(N+1, dtype=object)
    t = np.zeros(N+1)
    y[0] = y0
    t[0] = t0
    for k in range(N):
        y[k+1] = meth(y[k], t[k], h, f)
        t[k+1] = t[k]+h
    return t, y

def cmp(y , y2 , eps): #normvalue: max of the norm value np.max(np.abs(y[k] - y2[2*k]))
    N = np.array([np.absolute(y[k] - y2[2*k]) for k in range(len(y))])
    for k in range(len(y)):
        if np.max(N) > eps:
            return False
    return True

def meth_epsilon(y0, t0, tf, eps, f, meth):
    N = 100
    h = (tf - t0)/N
    t, y = meth_n_step(y0,t0,N,h,f,meth)
    t2 ,y2 = meth_n_step(y0,t0,2*N,h/2 ,f,meth)
    index = 0
    while (not cmp(y,y2,eps)) :
        index +=1
        N=2*N
        h = (tf - t0)/N
        t, y = meth_n_step(y0,t0,N,h,f,meth)
        t2 ,y2 = meth_n_step(y0,t0,2*N,h/2 ,f,meth)
    return t, y, index

#Returns tangential field
def tangential_field_dim2(f,  xmin, xmax, dx, ymin, ymax, dy, *text):
    X, Y = np.meshgrid(np.arange(xmin, xmax, dx), np.arange(ymin, ymax, dy))
    t = 0
    coord = np.zeros(X.shape, dtype=object)
    Fx = np.zeros(X.shape)
    Fy = np.zeros(X.shape)
    for i in range(coord.shape[0]):
        for j in range(coord.shape[1]):
            coord[i, j] = f(t, np.array([X[i,j], Y[i,j]]))
            Fx[i,j] = coord[i,j][0]
            Fy[i,j] = coord[i,j][1]
    plt.quiver(X, Y, Fx, Fy)
    plt.title(text)
    plt.show()

#Returns the error between f(tf) computed from our method and the value really expected
def error_methods(y0,t0,N,f,meth,tf, yf):
    h = (tf - t0)/N
    t_meth, y_meth = meth_n_step(y0,t0,N,h,f,meth)
    yf_meth = y_meth[N]
    return np.absolute(yf - yf_meth)
    
####### TEST #######

if __name__ == '__main__':
    
    ###  meth_n_step
    
    ##Dimension 1
    a = lambda t : math.exp(np.arctan(t))
    f = lambda t, y : y/(1+t**2)
    y0 = 1
    t0 = 0
    h = 1
    N = 20
    t_euler, y_euler = meth_n_step(y0, t0, N, h, f, step_euler)
    t_middle, y_middle = meth_n_step(y0, t0, N, h, f, step_middle)
    t_heun, y_heun = meth_n_step(y0, t0, N, h, f, step_heun)
    t_RK4, y_RK4 = meth_n_step(y0, t0, N, h, f, step_RK4)

    
    t = np.arange(0,N*h,h)
    y = np.vectorize(a)
    plt.plot(t, y(t),'.', label='réel')
    
    plt.plot(t_euler, y_euler, label='euler')
    plt.plot(t_middle, y_middle, label='middle')
    plt.plot(t_heun, y_heun, label='heun')
    plt.plot(t_RK4, y_RK4, label='RK4')
    plt.title("dimension 1 with 20 step of h=1")
    plt.legend()
    plt.show()

        ## erreur dim 1 pour y' : y/(1+t**2)
    F = lambda t : math.exp(np.arctan(t))
    f = lambda t, y : y/(1+t**2)
    y0 = 1
    t0 = 0
    tf = 1
    yf = F(tf)
    t = np.linspace(1, 1e2, 10).astype(int)
    t = np.append(t,np.linspace(1e2, 1e3, 10).astype(int))
    #t = np.append(t,np.linspace(1e3, 1e4, 10).astype(int))
    #t = np.append(t,np.linspace(1e4, 1e5, 10).astype(int))
    #t = np.append(t,np.linspace(1e5, 1e6, 10).astype(int))
    error_euler = []
    error_middle = []
    error_heun = []
    error_RK4 = []
    for N in t:
        error_euler.append(error_methods(y0,t0,N,f,step_euler,tf, yf))
        error_middle.append(error_methods(y0,t0,N,f,step_middle,tf, yf))
        error_heun.append(error_methods(y0,t0,N,f,step_heun,tf, yf))
        error_RK4.append(error_methods(y0,t0,N,f,step_RK4,tf, yf))
    plt.plot(t, error_euler, label='euler')
    plt.plot(t, error_middle, label='middle')
    plt.plot(t, error_heun, label='heun')
    plt.plot(t, error_RK4, label='RK4')
    plt.title("erreur sur y(1) pour y'(t) = y/(1+t**2) avec y(0) = 1 en fonction de N")
    plt.grid(True, which='both', linestyle='--')
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.show()
    
    ##Sin & cos
    
    cosinus = lambda t : math.cos(t)
    sinus = lambda t : math.sin(t)
    sol = [cosinus, sinus]

    f2 = lambda t, y : np.array([-y[1], y[0]])
    y0 = np.array([1,0])
    t0 = 0
    h = 0.5
    N = 15
    t_euler, y_euler = meth_n_step(y0, t0, N, h, f2, step_euler)
    t_middle, y_middle = meth_n_step(y0, t0, N, h, f2, step_middle)
    t_heun, y_heun = meth_n_step(y0, t0, N, h, f2, step_heun)
    t_RK4, y_RK4 = meth_n_step(y0, t0, N, h, f2, step_RK4)
    
    t = np.arange(0,N*h,h)
    y = [[sol[0](t), sol[1](t)] for t in t]
    
    # Cos dimension 1
    plt.plot(t, [sol[0](t) for t in t],'.', label='réel')
    plt.plot(t_euler, [x[0] for x in y_euler], label='euler', c='blue')
    plt.plot(t_middle, [x[0] for x in y_middle], label='middle', c='red')
    plt.plot(t_heun, [x[0] for x in y_heun], label='heun', c='purple')
    plt.plot(t_RK4, [x[0] for x in y_RK4], label='RK4', c='orange')
    plt.title("cos with 15 step of h=0.5")
    plt.legend()
    plt.show()

    # Sin dimension 1
    plt.plot(t, [sol[1](t) for t in t],'.', label='réel')
    plt.plot(t_euler, [x[1] for x in y_euler], label='euler', c='blue')
    plt.plot(t_middle, [x[1] for x in y_middle], label='middle', c='red')
    plt.plot(t_heun, [x[1] for x in y_heun], label='heun', c='purple')
    plt.plot(t_RK4, [x[1] for x in y_RK4], label='RK4', c='orange')
    plt.title("sin with 15 step of h=0.5")
    plt.legend()
    plt.show()

    # dimension 2
    plt.plot([x[0] for x in y], [x[1] for x in y],'.', label='réel')
    plt.plot([x[0] for x in y_euler], [x[1] for x in y_euler], label='euler')
    plt.plot([x[0] for x in y_middle], [x[1] for x in y_middle],label='middle')
    plt.plot([x[0] for x in y_heun], [x[1] for x in y_heun],  label='heun' )
    plt.plot([x[0] for x in y_RK4], [x[1] for x in y_RK4], label='RK4' )
    plt.title("Dimension 2 : méthodes avec (y_1\'(t), y_2\'(t)) = (-y_2(t), y_1(t))")
    plt.legend()
    plt.show()

    
    ## erreur cos
    cosinus = lambda t : math.cos(t)
    sinus = lambda t : math.sin(t)
    f2 = lambda t, y : np.array([-y[1], y[0]])
    y0 = np.array([1,0])
    t0 = 0
    tf = np.pi/2
    yf = np.array([0,1])
    t = np.linspace(1, 1e2, 10).astype(int)
    t = np.append(t,np.linspace(1e2, 1e3, 10).astype(int))
    #t = np.append(t,np.linspace(1e3, 1e4, 10).astype(int))
    #t = np.append(t,np.linspace(1e4, 1e5, 10).astype(int))
    #t = np.append(t,np.linspace(1e5, 1e6, 10).astype(int))
    error_reel = []
    error_euler = []
    error_middle = []
    error_heun = []
    error_RK4 = []
    for N in t:
        error_reel.append(np.absolute(cosinus(tf)-yf[0]))
        error_euler.append(error_methods(y0,t0,N,f2,step_euler,tf, yf))
        error_middle.append(error_methods(y0,t0,N,f2,step_middle,tf, yf))
        error_heun.append(error_methods(y0,t0,N,f2,step_heun,tf, yf))
        error_RK4.append(error_methods(y0,t0,N,f2,step_RK4,tf, yf))
    plt.plot(t, error_reel, label='"reel"')
    plt.plot(t, [x[0] for x in error_euler], label='euler')
    plt.plot(t, [x[0] for x in error_middle], label='middle')
    plt.plot(t, [x[0] for x in error_heun], label='heun')
    plt.plot(t, [x[0] for x in error_RK4], label='RK4')
    plt.yscale('log')
    plt.xscale('log')
    plt.grid(True, which='both', linestyle='--')
    plt.title("erreur sur y_1(pi/2) pour (y_1'(t),y_2(t)) = (-y_2(t), y_1(t)) avec (y_1(0), y_2(0))=(1, 0) en fonction de N")
    plt.legend()
    plt.show()

    ### meth_eps

    a = lambda t : math.exp(np.arctan(t))
    f = lambda t, y : y/(1+t**2)
    y0 = 1
    t0 = 0
    tf = 20
    eps = 0.01
    t_euler, y_euler , index_euler = meth_epsilon(y0, t0, tf,eps , f, step_euler)
    t_middle, y_middle , index_middle = meth_epsilon(y0, t0, tf,eps , f, step_middle)
    t_heun, y_heun ,index_heun = meth_epsilon(y0, t0, tf,eps , f, step_heun)
    t_RK4, y_RK4 , index_RK4= meth_epsilon(y0, t0, tf,eps , f, step_RK4)

    
    t = np.arange(0,tf,1)
    y = np.vectorize(a)
    plt.plot(t, y(t),'.', label='réel')
    
    plt.plot(t_euler, y_euler, label='euler')
    plt.plot(t_middle, y_middle, label='middle')
    plt.plot(t_heun, y_heun, label='heun')
    plt.plot(t_RK4, y_RK4, label='RK4')
    plt.title("dimension 1 with eps = 0.01")
    plt.legend()
    plt.show()

    ##meth_eps Dimension 2

    cosinus = lambda t : math.cos(t)
    sinus = lambda t : math.sin(t)
    sol = [cosinus, sinus]

    f2 = lambda t, y : np.array([-y[1], y[0]])
    y0 = np.array([1,0])
    t0 = 0
    tf = 20
    eps = 0.01
    t_euler, y_euler, index_euler = meth_epsilon(y0, t0, tf, eps, f2, step_euler)
    t_middle, y_middle, index_middle = meth_epsilon(y0,t0, tf, eps, f2, step_middle)
    t_heun, y_heun ,index_heun = meth_epsilon(y0, t0, tf, eps, f2, step_heun)
    t_RK4, y_RK4,index_RK4 = meth_epsilon(y0, t0,tf, eps, f2, step_RK4)
    
    t = np.arange(0,tf,1)
    y = [[sol[0](t), sol[1](t)] for t in t]

    # Cos dimension 1
    plt.plot(t, [sol[0](t) for t in t],'.', label='réel')
    plt.plot(t_euler, [x[0] for x in y_euler], label='euler', c='blue')
    plt.plot(t_middle, [x[0] for x in y_middle], label='middle', c='red')
    plt.plot(t_heun, [x[0] for x in y_heun], label='heun', c='purple')
    plt.plot(t_RK4, [x[0] for x in y_RK4], label='RK4', c='orange')
    plt.title("cos with eps = 0.01")
    plt.legend()
    plt.show()

    # Sin dimension 1
    plt.plot(t, [sol[1](t) for t in t],'.', label='réel')
    plt.plot(t_euler, [x[1] for x in y_euler], label='euler', c='blue')
    plt.plot(t_middle, [x[1] for x in y_middle], label='middle', c='red')
    plt.plot(t_heun, [x[1] for x in y_heun], label='heun', c='purple')
    plt.plot(t_RK4, [x[1] for x in y_RK4], label='RK4', c='orange')
    plt.title("sin  with eps = 0.01")
    plt.legend()
    plt.show()

    # dimension 2
    plt.plot([x[0] for x in y], [x[1] for x in y],'.', label='réel')
    plt.plot([x[0] for x in y_euler], [x[1] for x in y_euler], label='euler')
    plt.plot([x[0] for x in y_middle], [x[1] for x in y_middle],label='middle')
    plt.plot([x[0] for x in y_heun], [x[1] for x in y_heun],  label='heun' )
    plt.plot([x[0] for x in y_RK4], [x[1] for x in y_RK4], label='RK4' )
    plt.title("Dimension 2 with eps = 0.01")
    plt.legend()
    plt.show()

    # Tangential field for cos/sin
    tangential_field_dim2(f2, -2, 2, 0.25, -2, 2, 0.25, 'champs de tangentes de (y_1\', y_2\') = (-y_2, y_1)')
    
    
