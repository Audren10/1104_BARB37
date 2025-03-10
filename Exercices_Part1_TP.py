# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 17:03:18 2025

@author: audre
"""

### Exercice 8 

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# Définition des points du quart de cercle (0° à 90°)
X_points = np.array([np.sin(k * np.pi / 6) for k in range(4)])
Y_points = np.array([np.cos(k * np.pi / 6) for k in range(4)])

# Création de la spline cubique
spline = CubicSpline(X_points, Y_points)

# Génération des points pour tracer la spline
X_fine = np.linspace(X_points.min(), X_points.max(), 500)
Y_spline = spline(X_fine)

# Génération des points pour l'arc de cercle réel (référence)
theta = np.linspace(0, np.pi/2, 100)
X_arc = np.sin(theta)
Y_arc = np.cos(theta)

# Tracé
plt.figure(figsize=(6,6))
plt.plot(X_arc, Y_arc, '--', label="Arc de cercle réel", color='gray')  # Référence
plt.plot(X_fine, Y_spline, label="Spline Cubique", color='blue')  # Interpolation spline
plt.scatter(X_points, Y_points, color='red', zorder=3, label="Points d'origine")  # Points de contrôle

# Mise en forme
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.axis("equal")  # Pour conserver la forme circulaire
plt.title("Approximation d'un arc de cercle avec une spline cubique")

plt.show()


### Exercice 10 

x = np.linspace(1975, 2025, 100)
X = [1975, 1980, 1985, 1990]
U = [70.2, 70.2, 70.3, 71.2]

# polynomes de Lagrange 

def lagrange_interpolation(x, xi, yi):
    n = len(xi)
    result = 0.0
    for j in range(n):
        term = yi[j]
        for i in range(n):
            if i != j:
                term *= (x - xi[i]) / (xi[j] - xi[i])
        result += term  
    return result

y_lagrange = lagrange_interpolation(x, X, U)
# Spline cubique
spline = CubicSpline(X, U, extrapolate=True)
y_spline = spline(x)

# Tracé des interpolations et extrapolations
plt.figure(figsize=(8, 5))
plt.plot(x, y_lagrange, label="Lagrange", color='red', linestyle='dashed')
plt.plot(x, y_spline, label="Spline Cubique", color='blue')
plt.scatter(X, U, color='black', zorder=3, label="Données initiales")

# Mise en forme
plt.xlabel("Année")
plt.ylabel("Valeur interpolée")
plt.legend()
plt.grid(True)
plt.title("Comparaison de l'extrapolation : Lagrange vs Spline cubique")
plt.show()

### Exercice 11

X = np.linspace(-1, 1, 21)
U = np.sin(2 * np.pi * X)

# Création du tableau U avec une perturbation alternée
Uperb = np.array([np.sin(2 * np.pi * x) + (-1) ** (i + 1)  for i, x in enumerate(X)])

x = np.linspace(-1, 1, 1000)

# Approximation polynomiale de degré n (voir la convergence en augmentant n)
a_perb = np.polyfit(X, Uperb, 5)    # Coefficients du polynôme
a = np.polyfit(X, U, 5)

y_poly_perb = np.polyval(a_perb, x)  # Évaluation du polynôme
y_poly = np.polyval(a, x)

# Interpolation spline cubique
spline = CubicSpline(X, U)
y_spline = spline(x)

# Tracé des interpolations
plt.figure(figsize=(8, 5))
plt.plot(x, y_poly, label="Polynôme de Lagrange", color='red', linestyle='dashed')
plt.plot(x, y_poly_perb, label= "Polynôme de Lagrange", color = 'pink', linestyle = 'dashed')
plt.plot(x, y_spline, label="Spline Cubique", color='blue')
plt.scatter(X, U, color='black', zorder=3, label="Données initiales")

# Mise en forme
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(True)
plt.title("Comparaison : Polynôme de degré 3 vs Spline cubique")

plt.show()


### Exercice 12

def interp(T,x,y):
    phi = np.array([ (1.0-x)*(1.0-y)/4.0,
                 (1.0-x)*(1.0+y)/4.0,
                 (1.0+x)*(1.0+y)/4.0,
                 (1.0+x)*(1.0-y)/4.0 ])
    
    return T @ phi

# Températures aux sommets du carré
# T = [T_bas_gauche, T_haut_gauche, T_haut_droite, T_bas_droite]
T = np.array([30.0, 30.0, 30.0, 30.0])

# Grille de points pour l'évaluation
x = np.linspace(-1, 1, 10)
y = np.linspace(-1, 1, 10)
X, Y = np.meshgrid(x, y)

# Calcul des températures interpolées
Z = np.array([[interp(T, xi, yi) for xi in x] for yi in y])

# Ajout d'un foyer de chaleur au centre
Z += np.exp(-5 * (X**2 + Y**2))

# Tracé de la distribution de température
plt.figure(figsize=(8, 6))
contour = plt.contourf(X, Y, Z, 20, cmap='hot')
plt.colorbar(contour)
plt.title('Distribution de température avec un foyer au centre')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

### Exercice 13

### voir programme CubicSpline 

### Exercice 18 
from scipy.sparse import csr_matrix
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve
from scipy.integrate import quad
u = lambda x : (1/((x-0.3)**2 + 0.01) + 1/((x-0.9)**2 + 0.04)- 6)   
def alphonse(X):
                         
    n = len(X)
    e = n-1
    h = np.diff(X)
    hlow = np.array([*h,0])/6
    hup = np.array([0,*h])/6
    A = spdiags([hlow,2*(hup+hlow),hup],[-1,0,1],n,n)
    B = np.zeros((n,1))
    for elem in range(e):
        Xleft,Xright = X[elem],X[elem+1]
        B[elem]+= quad(lambda x: u(x)*(Xright-x)/(Xright-Xleft),Xleft,Xright)[0]
        B[elem+1] += quad(lambda x: u(x)*(x- Xleft)/(Xright-Xleft),Xleft,Xright)[0]

    return spsolve(csr_matrix(A),B)


# Séance 4 sur les NURBS
# Vincent Legat 
#
# -1- Définition récursive des B-splines (t est un tableau à entrer en input)

def b(t,T,i,p):
  if p == 0:
    return (T[i] <= t)*(t < T[i+1])
  else:
    u  = 0.0 if T[i+p ]  == T[i]   else (t-T[i])/(T[i+p]- T[i]) * b(t,T,i,p-1)
    u += 0.0 if T[i+p+1] == T[i+1] else (T[i+p+1]-t)/(T[i+p+1]-T[i+1]) * b(t,T,i+1,p-1)
    return u
 
# -2- Tracer une courbe de manière vectorielle.

T = [0,0,0,0,1,2,2,2,2]
X = [0,1,2,3,4]
Y = [0,3,0,3,0]
p = 3
n = len(T)-1
 
t = np.arange(T[p],T[n-p],0.001)
 
B = np.zeros((n-p,len(t)))
for i in range(0,n-p):
  B[i,:] = b(t,T,i,p)
  
  
x = X @ B
y = Y @ B
plt.plot(X,Y,'--r',X,Y,'or',x,y,'-r')
 
# On change un point : seulement UNE PARTIE de la courbe est modifiée....
# Par contre avec des splines usuelles, TOUTE la courbe est modifiée

 
X = [5,1,2,3,4]
Y = [5,3,0,3,0]
x = X @ B
y = Y @ B
plt.plot(X,Y,'--b',X,Y,'ob',x,y,'-b')
plt.show()

#Exercice  26
X = [1,0,0,1,0,0,1,0]
Y = [0,1,0,0,1,0,0,1]
T = [0,1,2,3,4,5,6,7]
t = np.linspace(2,5,100)
plt.plot(CubicSpline(T,X)(t),CubicSpline(T,Y)(t))
plt.show()

#La vraie solution est celle du devoir02. Ici on montre que contrairement aux Bsplines (devoir 3), il ne suffit pas d'ajouter des pts d'interpolation
# pour obtenir une version périodique... La solution du devoir 2 était donc un peu plus complexe à coder que celle du devoir 3 :-) 