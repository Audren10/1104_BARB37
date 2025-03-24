# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 10:17:35 2025

@author: Audren
"""

"""
Séance 8 : application numérique de Taylor et Runge-Kutta
exercices 51, 52, 57 (très similaire au 52!) 53, 54, 58, 59 
+ examens 91, 106, 119, 120 (par groupe)
"""
###############################################################################
# 51
from numpy import *
import numpy as np
import matplotlib.pyplot as plt

# Interval of integration and initial conditions
Xstart = 0; Xend = 1
Ustart = 0
# Exact solution at Xend (given by u(x)=0.5*(exp(x)-sin(x)-cos(x)))
Uend = 0.5 * (exp(Xend) - sin(Xend) - cos(Xend))

# Initialize arrays to store errors for different step sizes
Eexpl = zeros(8)  # Errors for explicit Euler
Eimpl = zeros(8)  # Errors for implicit Euler

# Loop over different step sizes (n = 2^(j+1) for j=0,...,7)
for j in range(8):
    n = int(pow(2, j+1))
    h = (Xend - Xstart) / n
    X = linspace(Xstart, Xend, n+1)
    
    # Allocate arrays for numerical solutions for each method
    Uexpl = zeros(n+1); Uexpl[0] = Ustart
    Uimpl = zeros(n+1); Uimpl[0] = Ustart
    
    # Explicit Euler method: u_{i+1} = u_i + h*(sin(x_i) + u_i)
    for i in range(n):
        Uexpl[i+1] = Uexpl[i] + h * (sin(X[i]) + Uexpl[i])
    
    # Implicit Euler method:
    # For u'(x) = sin(x) + u(x), the implicit Euler gives:
    # u_{i+1} = u_i + h*(sin(x_{i+1}) + u_{i+1})
    # Rearranging: u_{i+1} - h*u_{i+1} = u_i + h*sin(x_{i+1])
    # So, u_{i+1} = (u_i + h*sin(X[i+1])) / (1 - h)
    for i in range(n):
        Uimpl[i+1] = (Uimpl[i] + h * sin(X[i+1])) / (1 - h)
    
    # Compute the absolute errors at Xend for each method
    Eexpl[j] = abs(Uexpl[-1] - Uend)
    Eimpl[j] = abs(Uimpl[-1] - Uend)

# Estimate the order of convergence using successive errors:
# order = log2(E(j)/E(j+1))
Oexpl = log(abs(Eexpl[:-1] / Eexpl[1:])) / log(2)
Oimpl = log(abs(Eimpl[:-1] / Eimpl[1:])) / log(2)

print("Estimated Order of Convergence (Explicit Euler):", *['%.4f' % val for val in Oexpl])
print("Estimated Order of Convergence (Implicit Euler):", *['%.4f' % val for val in Oimpl])

# Optionally, plot the convergence errors versus step size on a log-log plot
hs = [(Xend - Xstart) / int(pow(2, j+1)) for j in range(8)]
plt.figure(figsize=(10,6))
plt.loglog(hs, Eexpl, 'o-', label='Explicit Euler Error')
plt.loglog(hs, Eimpl, 's-', label='Implicit Euler Error')
plt.xlabel('Step size (h)')
plt.ylabel('Absolute Error at x = 1')
plt.title('Convergence Errors for Explicit and Implicit Euler Methods')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()


###############################################################################
# 52
from numpy import *
from matplotlib import pyplot as plt

# Define the interval and initial condition
Xstart = 0; Xend = 4; 
Ustart = 1;

# Create a fine grid for the "exact" solution plot
x = linspace(Xstart, Xend, 100)
# Exact solution provided: u(x)= exp(-5*x) + x
u = exp(-5*x) + x

# Define the ODE function: f(x,u) = 5*(x-u) + 1
f = lambda x, u: 5*(x - u) + 1

# ----------------------%
# Explicit 1st-order Euler
# ----------------------%
plt.figure()
print('==================== Explicit 1st-order Euler ===============')
for j in range(1, 5):
    n = int(pow(2, j+1))
    X = linspace(Xstart, Xend, n+1)
    h = (Xend - Xstart)/n
    U = zeros(n+1)
    U[0] = Ustart
    for i in range(n):
        U[i+1] = U[i] + h * f(X[i], U[i])
    plt.subplot(2, 2, j)
    plt.xlim((-0.1, 4.1))
    plt.ylim((-2.0, 6.0))
    plt.yticks(arange(-2, 7, 2))
    plt.plot(x, u, '-k', X, U, '.-r', markersize=5.0)
    print(' u(4) = %21.14e (Explicit 1st-order Euler with %2d steps)' % (U[-1], n))

# ----------------------%
# Implicit 1st-order Euler
# ----------------------%
plt.figure()
print('==================== Implicit 1st-order Euler ===============')
for j in range(1, 5):
    n = int(pow(2, j+1))
    X = linspace(Xstart, Xend, n+1)
    h = (Xend - Xstart)/n
    U = zeros(n+1)
    U[0] = Ustart
    for i in range(n):
        # Implicit Euler: U[i+1] = U[i] + h * f(X[i+1], U[i+1])
        # Solve the linear equation: U[i+1] - h*5*U[i+1] = U[i] + h*(5*X[i+1] + 1)
        U[i+1] = (U[i] + h*(5*X[i+1] + 1)) / (1 + 5*h)
    plt.subplot(2, 2, j)
    plt.xlim((-0.1, 4.1))
    plt.ylim((-2.0, 6.0))
    plt.yticks(arange(-2, 7, 2))
    plt.plot(x, u, '-k', X, U, '.-g', markersize=5.0)
    print(' u(4) = %21.14e (Implicit 1st-order Euler with %2d steps)' % (U[-1], n))

# ----------------------%
# Explicit 4th-order Runge-Kutta
# ----------------------%
plt.figure()
print('==================== Explicit 4th-order Runge-Kutta ==========')
for j in range(1, 5):
    n = int(pow(2, j+1))
    X = linspace(Xstart, Xend, n+1)
    h = (Xend - Xstart)/n
    U = zeros(n+1)
    U[0] = Ustart
    for i in range(n):
        K1 = f(X[i], U[i])
        K2 = f(X[i] + h/2, U[i] + K1 * h/2)
        K3 = f(X[i] + h/2, U[i] + K2 * h/2)
        K4 = f(X[i] + h, U[i] + K3 * h)
        U[i+1] = U[i] + h*(K1 + 2*K2 + 2*K3 + K4)/6
    plt.subplot(2, 2, j)
    plt.xlim((-0.1, 4.1))
    plt.ylim((-2.0, 6.0))
    plt.yticks(arange(-2, 7, 2))
    plt.plot(x, u, '-k', X, U, '.-b', markersize=5.0)
    print(' u(4) = %21.14e (Explicit 4th-order Runge-Kutta with %2d steps)' % (U[-1], n))
plt.show()

###############################################################################
# 58 
import matplotlib.pyplot as plt
from numpy import linspace, meshgrid, arange, zeros, abs
import numpy as np

# Create a grid of x and y values in the interval [-3, 3]
x, y = meshgrid(linspace(-3, 3, 1000), linspace(-3, 3, 1000))

# Construct the complex grid: z = x + iy
z = x + 1j * y

# Compute the absolute value of the function f(z) = 1 + z + z^2/2
F = abs(1 + z + z*z/2)

# Plot filled contours of F with contour levels from 0 to 1 with a step of 0.1,
# using the reversed jet colormap.
plt.contourf(x, y, F, levels=arange(0, 1.1, 0.1), cmap=plt.cm.jet_r)
plt.colorbar(label='|1 + z + z^2/2|')
plt.xlabel('Real part')
plt.ylabel('Imaginary part')
plt.title('Contour Plot of |1 + z + z²/2|')
plt.show()
###############################################################################
# 59 
from numpy import *
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp

def f(x, u):
    L = 20.0
    k2 = 9.81 / L
    phi = pi / 2
    omega = 7.29e-05
    dudx = zeros(4)
    dudx[0] = u[2]
    dudx[1] = u[3]
    dudx[2] = 2 * omega * sin(phi) * u[3] - k2 * u[0]
    dudx[3] = -2 * omega * sin(phi) * u[2] - k2 * u[1]
    return dudx

# Integration settings
Xstart = 0
Xend = 300
Ustart = [1, 0, 0, 0]

# Using the explicit Euler method (for comparison)
n = 30000
X = linspace(Xstart, Xend, n+1)
U = zeros((n+1, 4))
U[0] = Ustart
h = (Xend - Xstart) / n

print("==== Using explicit Euler integrator :-(")
for i in range(n):
    U[i+1, :] = U[i, :] + h * f(X[i], U[i, :])
print("Number of steps used (explicit Euler): %d" % n)

plt.figure()
plt.xlim((-2.00, 2.00))
plt.ylim((-0.05, 0.05))
plt.plot(U[:, 0], U[:, 1], '-r', label='Explicit Euler')
plt.title("Phase plot (u[0] vs. u[1]) - Explicit Euler")
plt.xlabel("u[0]")
plt.ylabel("u[1]")
plt.legend()
plt.grid(True)
plt.show()

# Using RK23 integrator from scipy
print("==== Using RK23 integrator from scipy :-)")
sol = solve_ivp(f, [Xstart, Xend], Ustart, method='RK23')
print("Number of steps used (RK23): %d" % len(sol.t))

plt.figure()
plt.xlim((-2.00, 2.00))
plt.ylim((-0.05, 0.05))
plt.plot(sol.y[0], sol.y[1], '-b', label='RK23')
plt.title("Phase plot (u[0] vs. u[1]) - RK23")
plt.xlabel("u[0]")
plt.ylabel("u[1]")
plt.legend()
plt.grid(True)
plt.show()

"""
Séance 9 : méthodes prédicteur-correcteur 
ex 56, 60, 83 + examens 85, 99, 101, 126 (par groupe)
"""
###############################################################################
#56

import numpy as np
from matplotlib import pyplot as plt

# Define the function f(x, u) for the ODE system
def f(x, u):
    # Parameters
    C = 5.0  
    k = 6.0  
    # u[0] represents u(x) and u[1] represents its derivative, v(x) = u'(x)
    dudx = u[1]
    dvdx = - C * u[1] - k * u[0]
    return np.array([dudx, dvdx])

# Define the interval and initial conditions
Xstart = 0
Xend = 5
Ustart = [1, 0]  # u(0)=1, u'(0)=0

# Exact solution (analytical) for comparison:
# Given: u(x) = 3*exp(-2x) - 2*exp(-3x)
#        v(x) = u'(x) = -6*exp(-2x) + 6*exp(-3x)
x_exact = np.linspace(Xstart, Xend, 100)
u_exact = 3.0 * np.exp(-2 * x_exact) - 2.0 * np.exp(-3 * x_exact)
v_exact = -6.0 * np.exp(-2 * x_exact) + 6.0 * np.exp(-3 * x_exact)

# Plot the exact solution in black
plt.figure(figsize=(12, 6))
plt.plot(x_exact, u_exact, '-k', label='Exact u(x)')
plt.plot(x_exact, v_exact, '-k', label='Exact v(x)')

# Solve the ODE using the predictor-corrector method for two different step sizes
for n in [20, 40]:
    # Create the grid
    X = np.linspace(Xstart, Xend, n+1)
    U = np.zeros((n+1, 2))  # U will store the numerical solution [u, v]
    U[0] = Ustart
    h = (Xend - Xstart) / n  # Step size
    
    # Predictor-Corrector Loop (Heun's method)
    for i in range(n):
        # Predictor step: Euler step to estimate U at the next time step
        P = U[i, :] + h * f(X[i], U[i, :])
        # Corrector step: Average the slopes at the current and predicted points
        U[i+1, :] = U[i, :] + h * (f(X[i], U[i, :]) + f(X[i+1], P)) / 2
    
    # Plot the numerical solution for u and v in red
    # (For clarity, only label one of the numerical plots)
    label_u = f'Numerical u(x), n={n}' if n == 20 else None
    label_v = f'Numerical v(x), n={n}' if n == 20 else None
    plt.plot(X, U[:, 0], '.-r', markersize=5, label=label_u)
    plt.plot(X, U[:, 1], '.-r', markersize=5, label=label_v)

plt.xlabel('x')
plt.ylabel('Solution')
plt.title('Exact and Numerical Solutions of the ODE System')
plt.legend()
plt.grid(True)
plt.show()
###############################################################################
# 60 

import numpy as np
import matplotlib.pyplot as plt

# Create a grid over the complex plane: x (real) and y (imaginary) range from -3 to 3
x, y = np.meshgrid(np.linspace(-3, 3, 1000), np.linspace(-3, 3, 1000))
z = x + 1j * y

# Define the coefficients for the stability polynomial:
# b = 0.5 + z/2 + (3/8)*z^2 and c = z^2/4
b = 0.5 + z/2 + 3 * z**2 / 8
c = z**2 / 4.0

# Compute the two roots of the characteristic polynomial:
# a = b ± sqrt(b^2 - c)
f1 = np.abs(b - np.sqrt(b*b - c))
f2 = np.abs(b + np.sqrt(b*b - c))

# The 'gain' is the maximum of the moduli of the two roots
gain = np.maximum(f1, f2)

# Plot the stability region: We are interested in the region where gain <= 1.
plt.contourf(x, y, gain, levels=np.arange(0, 1.1, 0.1), cmap=plt.cm.jet_r)
plt.contour(x, y, gain, levels=np.arange(0, 1.1, 0.1), colors='black')

# Enhance the plot with grid lines and axis lines
ax = plt.gca()
ax.axhline(y=0, color='k')
ax.axvline(x=0, color='k')
ax.yaxis.grid(color='gray', linestyle='dashed')
ax.xaxis.grid(color='gray', linestyle='dashed')
plt.xticks(np.arange(-3, 4, 1))
plt.yticks(np.arange(-3, 4, 1))
plt.xlabel('Real Axis')
plt.ylabel('Imaginary Axis')
plt.title('Stability Region (|a| ≤ 1)')
plt.colorbar(label='Gain (max modulus of roots)')
plt.show()

"""
Séance 10 : Equations non linéaires

exercices 102, 116 et 136 
Méthode du point fixe : ex 133, 64, 63

Gauss-Seidel : exercice 72 
"""
###############################################################################£
# 63
# la convergence est de plus en plus lente car non lipschitz ! 
from cmath import sqrt

# Define the function for fixed-point iteration
f = lambda x: 2 * sqrt(x - 1)

def iter(x, tol, nmax):
    n = 0
    delta = float("inf")
    while abs(delta) > tol and n < nmax:
        x_old = x
        x = f(x)
        delta = x - x_old
        n += 1
        print(f"x = {x.real:+.6f}{x.imag:+.6f}i (Estimated error {abs(delta):.7e} at iteration {n})")
    return x
# Perform fixed-point iteration starting from x = 1.5
print("Starting iteration from x = 1.5")
result = iter(1.5, 1e-2, 50)
print(f"Found x = {result}")
###############################################################################
# 64 
# deux solutions x= 2 et x= 4
# en partant de 1.9, lipschitz non respecté ! 
g = lambda x : 4*x- x*x/2- 4

def iter(x, tol, nmax):
    n = 0
    delta = float("inf")
    while abs(delta) > tol and n < nmax:
        x_old = x
        x = g(x)
        delta = x - x_old
        n += 1
        print(f"x = {x.real:+.6f}{x.imag:+.6f}i (Estimated error {abs(delta):.7e} at iteration {n})")
    return x
# Perform fixed-point iteration starting from x = 1.9
print("Starting iteration from x = 1.5")
result = iter(1.9, 1e-5, 50)
print(f"Found x = {result}")

# Perform fixed-point iteration starting from x = 3.8
print("\nStarting iteration from x = 2.5")
result = iter(3.8, 1e-5, 50)
print(f"Found x = {result}")

# Perform fixed-point iteration starting from x = 6.0
print("\nStarting iteration from x = 2.5")
result = iter(6.0, 1e-5, 50)
print(f"Found x = {result}")

###############################################################################
# 72 
# changer la deuxieme equation par -6 x1 -8 x2 = -4
import numpy as np
from scipy.linalg import norm

def g(x):
    # Update rules based on the system of equations
    x_new = np.zeros_like(x)
    x_new[0] = (-3*x[1] + 6) / 5
    x_new[1] = -(6*x_new[0] - 4) / 8
    return x_new

def gauss(x, tol, nmax):
    n = 0
    delta = tol + 1
    x = np.array(x, dtype=float)
    while norm(delta) > tol and n < nmax:
        n += 1
        x_old = x.copy()
        x = g(x)
        delta = x - x_old
        print(f"Estimated error {norm(delta):.7e} at iteration {n}")
    return x

# Initial guess
initial_guess = [0, 0]
tolerance = 1e-6
max_iterations = 50

solution = gauss(initial_guess, tolerance, max_iterations)
print("Solution:", solution)

"""
Séance 11 : Newton-Raphson

139, 125, 122, 115, 62, 61 

approximation par la sécante : ex 138, 111, 98
application pouit trouver un pt stationnaire: ex 128, 73, 74 
application à Euler implicite : 132 et 104 
"""
###############################################################################
# 62 
from math import exp

# Define the function and its derivative
f = lambda x: x * exp(x)
dfdx = lambda x: (x + 1.0) * exp(x)

def newton(x, tol, nmax):
    n = 0
    delta = float("inf")
    while (abs(delta) > tol) and (n < nmax):
        delta = -f(x) / dfdx(x)
        x = x + delta
        n = n + 1
        print(f"x = {x:.7e} (Estimated error {abs(delta):.7e} at iteration {n})")
    return x

# Test the function
print(f"Found x = {newton(0.2, 1e-13, 50):.7e}") 
print(f"Found x = {newton(20.0, 1e-13, 50):.7e}") 

###############################################################################
# 61 

import numpy as np

def f1(x):
    return np.log(x) - 5 + x

def df1(x):
    return 1/x + 1

def newton_method(f, df, x0, tol=1e-10, max_iter=1000):
    x = x0
    for _ in range(max_iter):
        fx = f(x)
        dfx = df(x)
        if dfx == 0:
            raise ValueError("Derivative is zero; no convergence.")
        x_new = x - fx / dfx
        if abs(x_new - x) < tol:
            return x_new
        x = x_new
    raise ValueError("Maximum iterations exceeded; no convergence.")

initial_guess = 1.0  # Choose a positive initial guess
root1 = newton_method(f1, df1, initial_guess)
print(f"Root of ln(x) - 5 + x = 0 is approximately: {root1}")

import numpy as np

def f2(x):
    return (x - 2)**2 - np.log(x)

def df2(x):
    return 2*(x - 2) - 1/x

initial_guess = 2.5  # Choose an initial guess greater than 2
root2 = newton_method(f2, df2, initial_guess)
root2bis = newton_method(f2, df2, initial_guess - 2)
print(f"Root of (x - 2)^2 = ln(x) is approximately: {root2}")
print(f"Root of (x - 2)^2 = ln(x) is approximately: {root2bis}")
import numpy as np

def f3(x):
    return np.exp(-x) - x

def df3(x):
    return -np.exp(-x) - 1

initial_guess = 0.5  # Choose an initial guess
root3 = newton_method(f3, df3, initial_guess)
print(f"Root of e^(-x) = x is approximately: {root3}")

def f4(x):
    return x**3 + 4*x**2 - 10

def df4(x):
    return 3*x**2 + 8*x

initial_guess = 1.5  # Choose an initial guess
root4 = newton_method(f4, df4, initial_guess)
print(f"Root of x^3 + 4x^2 - 10 = 0 is approximately: {root4}")


"""
Séance 12
ex 75, 140, 107, 103, 96, 94, 88
"""
###############################################################################
# 75 
from numpy import *
from numpy.linalg import solve
def poissonSolve(nx,ny):
    n = nx*ny; h = 2/(ny-1)
    A = zeros((n,n))
    B = zeros(n)
    for i in range(n):
        A[i,i] = 1.0
    for i in range(1,nx-1):
        for j in range(1,ny-1):
            index = i + j*nx
            A[index,index] = 4.0
            A[index,index-1] =-1.0
            A[index,index+1] =-1.0
            A[index,index-nx] =-1.0
            A[index,index+nx] =-1.0
            B[index] = 1
    A = A / (h*h)
    return solve(A,B).reshape(ny,nx)