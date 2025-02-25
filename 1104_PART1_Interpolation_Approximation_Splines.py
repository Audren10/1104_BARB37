# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 15:59:06 2025

@author: Audren Balon 
"""


# Approximation, interpolation 
#
# Based on Vincent Legat
# Ecole Polytechnique de Louvain

import numpy as np
from numpy import *
from scipy.integrate import quad
from numpy.linalg import solve
import matplotlib.pyplot as plt
import time 
### 1
### Il est toujours utile d'utiliser time.time() pour comparer les performances 
### entre des méthodes 
### Attention, l'horloge dépend de la machine : le type de machine que l'on utilise est toujours précisé dans les publications scientifiques

# =========================================================================
 
def tic():
  global startTime
  startTime = time.time()
 
# =========================================================================
 
def toc(message = ''):
  global startTime
  stopTime = time.time()
  if message:
    message = ' (' + message + ')' ;
  print("Elapsed time is %.6f seconds %s" % ((stopTime - startTime),message) )
  startTime = 0
  
# =========================================================================

### 2 
### Définir une fonction analytique en python via lambda 
"""
Définir une fonction lambda permet d'éviter d'écrire une fct dédiée 
On évite: 
def u(x): 
    return 1/(1.1-x)

def phi(id,x): 
    if id == 0: 
        return x*(1+x)/2
    elif id == 1 : 
        return -x*(1-x)/2
    else:
        return -x*(1-x)/2
        
"""
 
u = lambda x : 1/(1.1-x)  # u(x)

phi = lambda id,x : { 
    0 : lambda x: x*(1+x)/2,
    1 : lambda x: -x*(1-x)/2,
    2 : lambda x: (1-x)*(1+x)  
  }[id](x)
 
"""
soit on définit les fonctions de Lagrange sur base du dictionnaire lambda défini plus haut
uniquement pour les polynomes de base de Lagrange de degré 2 
sinon, on doit changer le dictionnaire avec les fonctions de base de degré plus élevés... moins robuste
"""
def lagrange_deg2(x,U):
  u = zeros(size(x))
  for i in range(3):
    u += U[i]*phi(i,x)
  return u

"""
soit on effectue les boucles (cas plus général pour tous les degrés)
Essayez d'optimiser le code avec le moins de boucles possibles -> la vectorisation du calcul est toujours une bonne idée
"""

def min(A, B): 
    if(A > B) :
        return B
    else : 
        return A

# implémentation naive du code du prof 
def lagrange_naive(x,X,U):
 
  n = min(len(X),len(U))
  m = len(x)  
  uh = zeros(m)
  phi = zeros(n)  
  for j in range(m):
    uh[j] = 0
    for i in range(n):
      phi[i] = 1.0
      for k in range(n):
        if i != k:
          phi[i] = phi[i]*(x[j]-X[k])/(X[i]-X[k])
      uh[j] = uh[j] + U[i] * phi[i]
  return uh
# implémentation optimale du prof 
def lagrange(x, X, U):
    n = min(len(X), len(U))          
    phi = ones((n, len(x)))  # Correctement défini
    for i in range(n):
        for j in range(n):
            if i != j:
                phi[i] *= (x - X[j]) / (X[i] - X[j])
    return dot(U, phi)
 
# mon implémentation 
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

### On compare les techniques  (Benchmarking)
for n in [2000,20000]:
  x = linspace(-1,1,n)
  tic()
  lagrange(x,[0,1,2],[0,1,2])
  toc("Lagrange n = %d" % n)
  tic()
  lagrange_naive(x,[0,1,2],[0,1,2])
  toc("LagrangeNaive n = %d" % n)
  tic()
  lagrange_interpolation(x,[0,1,2],[0,1,2])
  toc("LagrangeTuteur n = %d" % n)
  
n = 15
h = 1/n
x = linspace(-1,1,2000) 
Xunif = arange(-1+h/2,1+h/2,h)
Xcheb = cos( pi*(2*arange(0,2*n)+1) / (4*n) )
 
functions = [lambda x : 1/(x**2 + 0.09), 
             lambda x : abs(x - 0.1),
             lambda x : 1/(1+ 250*x**2), 
             lambda x : cos(x) ]
 
"""
On impose la valeur de la fonction aux points d'interpolation, mais pas la dérivée
lorsque la dérivéé peut exploser, cela crée des instabilités aux bords de l'intervalle -> effet de Runge 


Pour des fonctions régulières, avec une dérivée bornée (comme cos(x), dérivée bornée par 1) : points d'inteprolations équidistants suffisent !
Pour des fonctions régulières, avec une dérivée non bornée, on peut observer des effets de Runge (croissance exponentielle de l'erreur)
Solution : 
    - mettre + de points aux bornes de l'intervalle -> absicce de Chebychev avec erreur équioscillante entre les points d'interpolation
Pour des fonctions irrégulières, on peut observer des effets semblables à Runge (oscillations avec croissance exponentielle)
Solution : 
    - découper l'intervalle en sous-intervalles et imposer la dérivée à la jonction entre tous les intervalles  -> splines (dont les splines cubiques et les NURBS)
Dans tous les cas, si le système est mal conditionné (epsilon-machine ne permet pas de discnerner un déterminant très très petit ou nul), système d'interpolation non solvable. 
En réalité, il est inévitable pour des systèmes de taille croissante de devenir mal-conditionnés. polyfit et polyval deviennent alors inutiles. Il existe des solutions pour essayer d'éviter les systèmes mal conditionnés, 
non vus dans le cadre de ce cours.

Pour rappel, le conditionnement d'une matrice normale est kappa = + grande valeur propre / + petite valeur propre 
Si la différence entre les valeurs propres est trop importante, le conditionnement est grand et le système est dit mal-conditionné. 
De manière générale, le conditionnement est kappa = ||A|| ||inv(A)|| (où l'on choisit la norme matricielle induite : par ex norme de Frobenius induit par tr(AB^T) ou norme-2 induite par un vecteur dans l'espace)

"""

# Observation de l'effet de Runge 

for u in functions:
  Uunif = u(Xunif)
  Ucheb = u(Xcheb)
  plt.figure('Polynomial interpolation')
  plt.plot(x,lagrange(x,Xunif,Uunif),'-b',label='Abscisses équidistantes')
  plt.plot(x,lagrange(x,Xcheb,Ucheb),'-r',label='Abscisses de Tchebychev')
  plt.plot(Xunif,Uunif,'ob',Xcheb,Ucheb,'or')
  plt.xlim((-1,1))
  plt.title(f"Observation : Effet de Runge pour l'interpolation avec abscisses équidistantes {u}")
  plt.ylim((0,max(Uunif)*2))
  plt.title('Abscisses : n = %d (uniforme) %d (Tchebychev)' 
             % (len(Xunif),len(Xcheb)))
  plt.legend(loc='upper right')
  plt.show()
  
# Interpolation 
   
X = array([1,-1,0])
U = u(X)
plt.plot(X,U,'.r',markersize=20)
x = linspace(-1,1,200)
plt.plot(x,u(x),'-r')   # en rouge la vraie fonction qu'on connait
plt.title("Graphe Interpolation et Approximation intégrale")
uInterpolation = lagrange_deg2(x,U)
plt.plot(x,uInterpolation,'-k') # en noir l'interpolation 

# Approximation intégrale 

""" 
Si on connait la fonction u(x) sur tout le domaine, pourquoi se contenter d'utiliser sa valeur aux points d'interpolation pour calculer l'approximation? 
Pour etre le plus precis possible, utilisons toute l'information que nous avons !
La somme intérieure devient une intégrale dans les équations normales !
L'intégrale est approximée par la somme lorsque nous connaissons uniquement les valeur saux points d'inteprolation !
"""
b = zeros(3)
A = zeros((3,3))
for i in range(3):
  b[i] = quad(lambda x: u(x)*phi(i,x),-1,1)[0]
  for j in range(3):
    A[i,j] = quad(lambda x: phi(i,x)*phi(j,x),-1,1)[0]
 
Uapp = solve(A,b)
uApproximation = lagrange_deg2(x,Uapp)
plt.plot(x,uApproximation,'-b') # en bleu l'approximation intégrale 
 
plt.show()

# Approximation 

from matplotlib import pyplot as plt
 
plt.figure("Linear approximation")
 
X = [ -55, -25,   5,  35,  65]
U = [3.25,3.20,3.02,3.32,3.10]

a = polyfit(X,U,1)
x = linspace(X[0],X[-1],100)
uh = polyval(a,x)
 
plt.plot(x,uh,'-b',X,U,'or')
plt.title("Approximation Linéaire")
plt.show()

# Spline Cubique 

# Avec la fonction du package 
from scipy.interpolate import CubicSpline as spline
 
 
X = arange(-55,70,10)
U = [3.25, 3.37, 3.35, 3.20, 3.12, 3.02, 3.02,
           3.07, 3.17, 3.32, 3.30, 3.20, 3.10]
          
x = linspace(X[0],X[-1],100)
 
uhLag = polyval(polyfit(X,U,len(X)-1),x)
uhSpl = spline(X,U)
plt.title("Spline cubique")
plt.plot(x,uhLag,'--r',x,uhSpl(x),'-b')
plt.plot(X,U,'or') 
plt.xlim([-60,80])
plt.ylim([3.0,3.5])
 
plt.show()

###############################################################################

n = 21
L = 1
X = linspace(-L,L,n)      # ; print(X)  # Pour debugger step by step                 
U = sin(2*pi*X)           # ; print(U)  # Pour debugger step by step         
 
x = linspace(-L,L,10*n)

# Accéder aux données d'un tableau
print(X)
print(X[:])      # on prend tous les elements 
print(X[0:10])   # de l'élément 0 à 10 par pas de 1 ( par défaut)
print(X[0:21:2]) # de l'élément 0 à 21 par saut de 2 
 

Y = X[0:21:2]   # ici, on copie le pointeur (deux pointeurs vers une meme data)

Z = copy(X)     # ici, on copie les données (deux pointeurs vers deux datas distinctes, l'une copiée de l'autre )
Z[0] = 69;    print('X[0] = %3d - Y[0] = %3d - Z[0] = %3d' % (X[0],Y[0],Z[0]))
X[0] = 456;   print('X[0] = %3d - Y[0] = %3d - Z[0] = %3d' % (X[0],Y[0],Z[0]))
X[0] = -1;    print('X[0] = %3d - Y[0] = %3d - Z[0] = %3d' % (X[0],Y[0],Z[0]))
 
 
uSpline1 = spline(X[0:10],U[0:10])(x)
uSpline2 = spline(X[0:21:2],U[0:21:2])(x)
 
plt.plot(x,uSpline1,'-b',label='spline sur les 10 premiers point')
plt.plot(x,uSpline2,'-r',label='spline sur un point sur deux')
plt.plot(X[0:10],U[0:10],'.r',markersize=20,label='10 premiers points')
plt.plot(X[0:21:2],U[0:21:2],'.b',markersize=10,label='1 point sur 2')
plt.legend(loc='upper right')
plt.show()
 
# Maintenant, sauriez-vous implémenter la Spline Cubique vous-même : Question 13 (Séance d'exercice 2)
# Inspectez l'implémentation de CubicSpline dans la documentation ! 
# Proposez votre version 

from scipy.interpolate import CubicSpline as spline
import inspect
lines = inspect.getsource(spline)
print(lines)
"""
def SplineCubique(X, U): 
    
        ... A faire 
    
    
    return Spline 
"""