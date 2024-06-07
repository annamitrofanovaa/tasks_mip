#Реализовать численное интегрирование уравнения m*ddy+p*dy+k*y=0 любым методом (m, p, k – заданные числовые параметры)
import numpy as np
import matplotlib.pyplot as plt

m = 1.0
p = 0.5
k = 2.0

def f(t, y, v):
    return v

def g(t, y, v):
    return - (p/m) * v - (k/m) * y

# Метод Рунге-Кутта
def runge_kutta_4(y0, v0, t0, tf, dt):
    n = int((tf - t0) / dt) + 1
    t = np.linspace(t0, tf, n)
    y = np.zeros(n)
    v = np.zeros(n)
    y[0] = y0
    v[0] = v0
    
    for i in range(1, n):
        k1_y = dt * f(t[i-1], y[i-1], v[i-1])
        k1_v = dt * g(t[i-1], y[i-1], v[i-1])
        
        k2_y = dt * f(t[i-1] + 0.5 * dt, y[i-1] + 0.5 * k1_y, v[i-1] + 0.5 * k1_v)
        k2_v = dt * g(t[i-1] + 0.5 * dt, y[i-1] + 0.5 * k1_y, v[i-1] + 0.5 * k1_v)
        
        k3_y = dt * f(t[i-1] + 0.5 * dt, y[i-1] + 0.5 * k2_y, v[i-1] + 0.5 * k2_v)
        k3_v = dt * g(t[i-1] + 0.5 * dt, y[i-1] + 0.5 * k2_y, v[i-1] + 0.5 * k2_v)
        
        k4_y = dt * f(t[i-1] + dt, y[i-1] + k3_y, v[i-1] + k3_v)
        k4_v = dt * g(t[i-1] + dt, y[i-1] + k3_y, v[i-1] + k3_v)
        
        y[i] = y[i-1] + (1.0 / 6.0) * (k1_y + 2*k2_y + 2*k3_y + k4_y)
        v[i] = v[i-1] + (1.0 / 6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
    
    return t, y, v

y0 = 1.0  
v0 = 0.0  
t0 = 0.0  
tf = 10.0 
dt = 0.01 

t, y, v = runge_kutta_4(y0, v0, t0, tf, dt)

plt.plot(t, y, label='y(t)')
plt.xlabel('Время')
plt.ylabel('Перемещение')
plt.title('Численное интегрирование методом Рунге-Кутты')
plt.legend()
plt.grid()
plt.show()
