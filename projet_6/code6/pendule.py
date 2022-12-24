import solving_method as sm
import numpy as np
import scipy.integrate as sc
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import pylab as py
        

def one_link_pendulum(t, y):
    l = 2
    g = 9.81
    a = g/l
    b = 0
    theta, omega = y
    dydt = np.array([omega, -b * omega - a * np.sin(theta)])
    return dydt


def one_link_pendulum_bis(y, t):
    return one_link_pendulum(t, y)


def freq_one_link(sol, t):
    times = []
    n = len(sol)
    for i in range(n - 1):
        if (sol[i, 0] * sol[i + 1, 0] <= 0):
            times.append(i)
    ntime = len(times)
    time = (t[times[ntime - 1]] - t[times[0]]) / ((ntime - 1) / 2)
    freq = 1 / time
    return freq * np.pi * 2


def two_link_pendulum(y, t):
    theta1, theta2, omega1, omega2 = y
    m1, m2 = 0.2, 0.2
    l1, l2 = 2, 2
    g = 9.81
    omega1_p = (-g * (2 * m1 + m2) * np.sin(theta1) - m2 * g * np.sin(theta1 - 2 * theta2) - 2 * np.sin(theta1 - theta2) * m2 * (omega2 ** 2 * l2 + omega1 ** 2 * l1 * np.cos(theta1 - theta2))) / (l1 * (2 * m1 + m2 * (1 - np.cos(2 * (theta1 - theta2)))))
    omega2_p = (2 * np.sin(theta1 - theta2) * (omega1 ** 2 * l1 * (m1 + m2) + g * (m1 + m2) * np.cos(theta1) + omega2 ** 2 * l2 * m2 * np.cos(theta1 - theta2))) / (l2 * (2 * m1 + m2 * (1 - np.cos(2 * (theta1 - theta2)))))
    dydt = np.array([omega1, omega2, omega1_p, omega2_p])
    return dydt


def positions(theta, l):
    theta1, theta2 = theta
    l1, l2 = l
    n = len(theta1)
    x2 = np.zeros(n)
    y2 = np.zeros(n)
    for i in range(n):
        x2[i] = l1 * np.sin(theta1[i]) + l2 * np.sin(theta2[i])
        y2[i] = -l1 * np.cos(theta1[i]) - l2 * np.cos(theta2[i])
    return x2, y2


def first_turn_around(theta):
    i = 1
    while (i < len(theta) - 1 and (theta[i] - np.pi) * (theta[i + 1] - np.pi) > 0):
        i += 1
    if (i >= len(theta) - 1):
        return 0
    return i


def map_pendulum(n, max_time, max_iteration):
    g = 9.81
    m = [0.2, 0.2]
    l = [2, 2]
    t = np.linspace(0, max_time, max_iteration)
    mat = np.zeros((n, n))
    for i in range(n):
        print(str(int(i / n * 1000) / 10) + '%')
        for j in range(n):
            mat[j, i] = first_turn_around(sc.odeint(two_link_pendulum, [i * 2 * np.pi / n - np.pi, j * 2 * np.pi / n - np.pi, 0, 0], t)[:, 1]) / max_iteration * max_time
    print('100%')
    plt.imshow(mat, cmap='viridis', extent=[0, 2 * np.pi, 0, 2 * np.pi])
    plt.colorbar(label = 'time (s)')

    plt.title('Graph of the time for the pendulum to flip over')
    plt.xlabel(r'$\theta_1$ (rad)')
    plt.ylabel(r'$\theta_2$ (rad)')
    plt.show()

def map_freq(n, max_time, max_iteration):
    t = np.linspace(0, max_time, max_iteration)
    angle = np.linspace(np.pi / n, np.pi, n)
    freq = np.zeros(n)
    for i in range(n):
        y0 = np.array([angle[i], 0])
        sol = sc.odeint(one_link_pendulum_bis, y0, t)
        freq[i] = freq_one_link(sol, t)

    plt.figure()
    plt.plot(angle, freq, 'o')
    plt.axhline(y = np.sqrt(9.81 / 2))
    plt.xlabel('theta initial (rad)')
    plt.ylabel('frequency (rad/s)')
    plt.legend()
    plt.title('Frenquencies as a function of the initial angle')
    plt.show()

def init():
    global x
    global y
    min_x, max_x = min_max(x)
    min_y, max_y = min_max(y)
    ax.set_xlim(min_x - 0.2, max_x + 0.2)
    ax.set_ylim(min_y - 0.2, max_y + 0.2)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ln.set_data(xdata, ydata)
    return ln,

def update(frame):
    global y
    global x
    xi = x[int(frame)]
    yi = y[int(frame)]
    xdata.append(xi)
    ydata.append(yi)
    ln.set_data(xdata, ydata)
    return ln,

def min_max(a):
    min, max = a[0], a[0]
    for i in range(len(a)):
        if (a[i] < min):
            min = a[i]
        if (a[i] > max):
            max = a[i]
    return min, max


if __name__ == '__main__':

    n = 2500
    y0 = np.array([np.pi / 6, 0])
    t = np.linspace(0, 50, n)
    #sol = sc.odeint(one_link_pendulum, y0, t, args=(b, a))
    t, sol, index = sm.meth_epsilon(y0, t[0], t[n - 1], t[n - 1] / n, one_link_pendulum, sm.step_RK4)

    n = len(sol)
    x, y = np.zeros(n), np.zeros(n)
    for i in range(n):
        x[i] = sol[i][0]
        y[i] = sol[i][1] 

    plt.figure()
    plt.plot(t, x, 'b', label = 'theta(t)')
    plt.plot(t, y, 'g', label = 'omega(t)')
    plt.xlabel('t (s)')
    plt.legend()
    plt.title('Exemple of one link pendulum')
    plt.show()

    map_freq(50, 100, 1000)

    n = 1000
    y01 = np.array([np.pi, np.pi/2, 0, 0])
    y02 = np.array([np.pi, np.pi/2 + np.pi / 20, 0, 0])
    t = np.linspace(0, 20, n)
    sol1 = sc.odeint(two_link_pendulum, y01, t)
    sol2 = sc.odeint(two_link_pendulum, y02, t)
    #t, sol, index = sm.meth_epsilon(y0, t[0], t[n - 1], t[n - 1] / n, two_link_pendulum, sm.step_RK4)

    x11, y11, x21, y21 = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    x12, y12, x22, y22 = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    for i in range(n):
        x11[i] = sol1[i][0]
        y11[i] = sol1[i][2] 
        x21[i] = sol1[i][1]
        y21[i] = sol1[i][3]
        x12[i] = sol2[i][0]
        y12[i] = sol2[i][2] 
        x22[i] = sol2[i][1]
        y22[i] = sol2[i][3]

    plt.figure()
    plt.plot(t, x11, 'b', label = r'$\theta_1$(t)')
    plt.plot(t, y11, 'deepskyblue', linestyle='--', label = r'$\omega_1$(t)')
    plt.plot(t, x21, 'r', label = r'$\theta_2$(t)')
    plt.plot(t, y21, 'crimson', linestyle='--', label = r'$\omega_2$(t)')
    plt.xlabel('t (s)')
    plt.legend()
    plt.title('Exemple of two link pendulum')
    plt.show()

    x, y = positions([x11, x21], [2, 2])
    x2, y2 = positions([x12, x22], [2, 2])

    plt.figure()
    plt.subplot(121)
    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(r'm2 trajectory ($\theta_1 = \pi, \theta_2 = \pi / 2$)')
    plt.subplot(122)
    plt.plot(x2, y2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(r'm2 trajectory ($\theta_1 = \pi, \theta_2 = \pi / 2 + \pi / 20 $)')
    plt.show()

    i = np.linspace(0, len(x) - 1, len(x))

    fig, ax = plt.subplots()
    xdata, ydata = [], []
    ln, = plt.plot([], [], 'r', animated=True, label = 'm2 trajectory')
    ani = anim.FuncAnimation(fig, update, frames = i , init_func = init, blit = True, interval = 2.5, repeat = False)
    plt.legend()
    plt.show()

    map_pendulum(50, 10, 50)