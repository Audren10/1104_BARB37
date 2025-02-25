from numpy import *
from numpy.linalg import solve


def cubicSolve(U,h) :
  n = len(U)-1
  A = zeros((n+1,n+1))
  b = zeros(n+1)
  #créer la matrice tri-diagonale   
  A[0,0] = A[n,n] = 1
  for i in range(1,n):
    A[i,i-1:i+2] = [1,4,1]
  A *= (h*h)/6
  #créer le vecteur b 
  b[1:-1] = U[:-2] - 2*U[1:-1] + U[2:]
  #résoudre le système (ne jamais inverser le système !!!)
  # on pourrait aussi utiliser un solveur pour matrices creuses (sparse solver) -> très utile ici et bcp plus efficace quand
  # la taille de la matrice croit car bcp d'éléments nuls (cfr Analyse Numérique pour les mathaps)
  ddU = solve(A,b)
  return ddU
     
def cubicSpline(X,h,U,ddU,x) :

  n = len(U)-1
  index = zeros_like(x,dtype='int')
  #méthode un peu obscure à comprendre, mais efficace. La fonction clip contraint les valeurs de index entre 1 et n 
  for i in range(n):
    index += (x >= X[i]).astype(int)
  index = clip(index,1,n)

  return ( ddU[index-1]/(6*h)*(X[index]   - x)**3
           + ddU[index]/(6*h)*(x - X[index-1])**3
           + (U[index-1]/h - ddU[index-1]*h/6)*(X[index]   - x)
           + (U[index]/h   - ddU[index]  *h/6)*(x - X[index-1]))
   
#
# -1- Test du splines cubiques


u = lambda x : 1/(x*x + 1)  
#u = lambda x : 2*cos(x)  

Xstart = -5.0
Xend   =  5.0                                         
n =10                                                      
X,h = linspace(Xstart,Xend,n+1,retstep=True)                   
                                
#
# -2- Calcul des splines cubiques sur un intervalle quelconque
#
#

Xstart = -4.2
Xend   =  9.7     
U = u(X) 
ddU = cubicSolve(U,h)
x = linspace(-4.2,9.7,200)
uh  = cubicSpline(X,h,U,ddU,x)
print("==== Computing the cubic splines curve") 
print("ddU = ",end=''); print(ddU)

#
# - Obtenir la bonne courbe avec scipy pour vérifier
#

from scipy.interpolate import CubicSpline as spline
uhSpl = spline(X,U,bc_type='natural')



from matplotlib import pyplot as plt
plt.rcParams['toolbar'] = 'None'
plt.rcParams['figure.facecolor'] = 'white'
plt.figure("Cubic splines :-)")
plt.plot(X,U,'or',markersize='5',label='Data points')
plt.plot(x,uh,'-b',label='Cubic splines interpolation')
plt.plot(x,uhSpl(x),'-g',label='Cubic splines from scipy !')
plt.legend(loc='lower left')
plt.savefig('picture.png')
plt.show()

