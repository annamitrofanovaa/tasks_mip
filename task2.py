#Реализовать интерполяцию траектории полиномом пятого порядка (exercise 9.5), показать графики s(t), ds(t), dds(t)
import numpy as np
import matplotlib.pyplot as plt

T = 1

A = np.array([
    [0, 0, 0, 0, 0, 1],         
    [T**5, T**4, T**3, T**2, T, 1],
    [0, 0, 0, 0, 1, 0],        
    [5*T**4, 4*T**3, 3*T**2, 2*T, 1, 0],  
    [0, 0, 0, 2, 0, 0],         
    [20*T**3, 12*T**2, 6*T, 2, 0, 0],  
])

b = np.array([0, 1, 0, 0, 0, 0])

coefficients = np.linalg.solve(A, b)

def poly(t):
    return sum(coefficients[i] * t**(5-i) for i in range(6))

def dpoly(t):
    return sum((5-i) * coefficients[i] * t**(4-i) for i in range(5))

def ddpoly(t):
    return sum((5-i) * (4-i) * coefficients[i] * t**(3-i) for i in range(4))

def dddpoly(t):
    return sum((5-i) * (4-i) * (3-i) * coefficients[i] * t**(2-i) for i in range(3))

t_values = np.linspace(0, T, 1000)

s_values = np.array([poly(t) for t in t_values])
ds_values = np.array([dpoly(t) for t in t_values])
dds_values = np.array([ddpoly(t) for t in t_values])
ddds_values = np.array([dddpoly(t) for t in t_values])

plt.figure(figsize=(12, 8))

plt.subplot(4, 1, 1)
plt.plot(t_values, s_values, label='s(t)')
plt.xlabel('t')
plt.ylabel('s(t)')
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(t_values, ds_values, label='ds(t)')
plt.xlabel('t')
plt.ylabel('ds(t)')
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(t_values, dds_values, label='dds(t)')
plt.xlabel('t')
plt.ylabel('dds(t)')
plt.legend()

plt.subplot(4, 1, 4)
plt.plot(t_values, ddds_values, label='ddds(t)')
plt.xlabel('t')
plt.ylabel('ddds(t)')
plt.legend()

plt.tight_layout()
plt.show()
