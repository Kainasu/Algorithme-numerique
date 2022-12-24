import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../gradient")
from gradient_conj import *
def Mc(n):
    m = np.zeros((n*n,n*n))
    for i in range(n*n):
        m[i][i] = 4
        if(i+1 < n*n):
            m[i][i+1] = -1
        if(i-1 > 0):
            m[i][i-1] = -1
        if(i+n < n*n):
            m[i][i+n] = -1
        if(i-n > 0):
            m[i][i-n] = -1
        
    return m


def f_rad(n , T):
    f = np.zeros(n*n)
    f[(n//2)*n + (n//2)] = T
    return f

def f_wa (n , T) :
    f = np.zeros(n*n)
    for i in range(n) :
        f[n*(n-1) + i] = T
    return f


def show_fig(X, Y , n, s) :
    xx = np.linspace(0 , 1 , n)
    yy = np.linspace(0 , 1 , n)
    T1 = np.reshape(X, (n, n))
    T2 = np.reshape(Y, (n, n))
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    plot1 = axes[0].pcolormesh(xx, yy, T1, cmap=plt.cm.hot, shading='auto')
    axes[0].set_title("Radiateur carré")
    plot2 = axes[1].pcolormesh(xx, yy, T2, cmap=plt.cm.hot, shading='auto')
    axes[1].set_title("Mur chaud")
    cbr = fig.colorbar(plot2)
    cbr.set_label(label='Temperature',weight='bold')
    fig.suptitle("Résolution de l'équation de la chaleur (avec " + s + ")")
    fig.tight_layout()

    plt.show()
    
if __name__ == '__main__':
    N = 50
    MTC = Mc(N)
    FRA = f_rad(N , 100)
    FWA = f_wa(N , 100)
    
    X = np.linalg.solve(MTC , FRA)
    XP = np.linalg.solve(MTC , FWA)
    show_fig(X ,XP, N, "linalg.solve")

    X = np.zeros(N * N)
    XP = np.zeros(N * N)
    X = conjGradPrecond(MTC, FRA, X)
    XP = conjGradPrecond(MTC, FWA, XP)
    show_fig(X ,XP, N, "conjGradPrecond")
