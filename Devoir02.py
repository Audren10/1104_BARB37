# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 17:34:33 2025

@author: audre
"""
import numpy as np

def spline(x, h, U):
    """
    Interpolation par splines cubiques périodiques avec la formule en t1 et t2.
    
    Paramètres:
      x : array_like
          Points où évaluer la spline.
      h : float
          Pas entre les nœuds (abscisses équidistantes).
      U : array_like
          Valeurs aux nœuds, avec U[0] identique à U[n] pour la périodicité.
    
    Retourne:
      S : array
          Valeurs interpolées en x.
    """
    U = np.asarray(U)
    n = U.size
    # Construction des nœuds (n+1 points, dernier point identique au premier)
    X = np.arange(n+1) * h
    # Ramener x dans [0, n*h)
    x_mod = np.mod(x, n * h)
    # Indice de l'intervalle
    index = (x_mod // h).astype(int)
    index = np.clip(index, 0, n-1)
    
    # Définition de t1 et t2
    # t1 = distance entre x et le nœud supérieur : X[i+1]-x_mod
    # t2 = distance entre x et le nœud inférieur : x_mod-X[i]
    t1 = X[(index+1) % (n+1)] - x_mod  # Note : ici X[n] = n*h, qui correspond à U[0] par périodicité.
    t2 = x_mod - X[index]
    
    # Construction du système pour U2 (dérivées secondes)
    A = np.zeros((n, n))
    d = np.zeros(n)
    for i in range(n):
        A[i, i] = 4.0
        A[i, (i-1) % n] = 1.0
        A[i, (i+1) % n] = 1.0
        
        d[i] = (6.0 / h**2) * ( U[(i+1) % n] - 2*U[i] + U[(i-1) % n] )
    U2 = np.linalg.solve(A, d)
    
    # Formule d'évaluation de la spline
    S = ( U2[index] / (6*h) * (t1**3) +
          U2[(index+1) % n] / (6*h) * (t2**3) +
          (U[index] / h - U2[index] * h / 6) * t1 +
          (U[(index+1) % n] / h - U2[(index+1) % n] * h / 6) * t2 )
    
    return S

def main() :

    from matplotlib import pyplot as plt
    plt.rcParams['toolbar'] = 'None'
    plt.rcParams['figure.facecolor'] = 'lavender'

    n = 4;
    h = 3*np.pi/(2*(n+1));
    T = np.arange(0,3*np.pi/2,h)
    X = np.cos(T); Y = np.sin(T)

    fig = plt.figure("Splines cubiques et cercle :-)")
    plt.plot(X,Y,'.r',markersize=10)
    t = np.linspace(0,2*np.pi,100)
    plt.plot(np.cos(t),np.sin(t),'--r')

    t = np.linspace(0,3*np.pi/2,100)
    plt.plot(spline(t,h,X),spline(t,h,Y),'-b')
    plt.axis("equal"); plt.axis("off")
    plt.show()
 
if __name__ == '__main__':
  main()
  
"""
import matplotlib
from matplotlib import pyplot as plt
from numpy import *

 
def mouse(event):
  global X,Y,n
  if (event.dblclick):
    t  = arange(0,n+0.001,0.001)
    x  = spline(t,1.0,X)
    y  = spline(t,1.0,Y)
    plt.plot(x,y,'-b')
    X,Y = [],[]; n = 0
  else :    
    x = event.xdata 
    y = event.ydata
    if (x != None and y != None) :
      n = n + 1
      X = append(X,[x])
      Y = append(Y,[y])
      print("New data : " + str(x) + "," + str(y))
      plt.plot([x],[y],'.r',markersize=10)
  fig.canvas.draw()
 
 
matplotlib.rcParams['toolbar'] = 'None'
matplotlib.rcParams['lines.linewidth'] = 1
plt.rcParams['figure.facecolor'] = 'lavender'
 
X,Y = [],[]; n = 0   
fig = plt.figure("Cubic spline interpolation")
fig.canvas.mpl_connect('button_press_event',mouse)
plt.ylim((0,1)); plt.xlim((0,1.3)); plt.axis("off")
 
plt.show()
"""