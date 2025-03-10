# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 13:46:47 2025

@author: Audren Balon
"""

import numpy as np
from scipy.spatial import Delaunay

# ------------------------------------------------------------------------------------
# Groupe 11.76 LEPL 1502 – Intégration du flux magnétique
# ------------------------------------------------------------------------------------

def magnetComputeInduction(Xmagnet, Ymagnet, Zmagnet, Xcoil, Ycoil, triangles, Xshift, mu0, mu):
    """
    Compute the magnetic flux intercepted by a coil.
    """
    nElem = len(triangles)
    m = len(Xshift)
    
    surface = np.zeros(nElem)
    for iElem in range(nElem):
        x = Xcoil[triangles[iElem, :]]
        y = Ycoil[triangles[iElem, :]]
        surface[iElem] = ((x[1] - x[0]) * (y[2] - y[0]) - (y[1] - y[0]) * (x[2] - x[0])) / 2.0
    
    Sbobine = np.sum(surface)
    phi = np.zeros(m)
    
    for i in range(m):
        for iElem in range(nElem):
            Xspire = np.mean(Xcoil[triangles[iElem, :]])
            Yspire = np.mean(Ycoil[triangles[iElem, :]])
            Wspire = surface[iElem]
            
            for jElem in range(nElem):
                Xp = Xspire - np.mean(Xmagnet[triangles[jElem, :]]) - Xshift[i]
                Yp = Yspire - np.mean(Ymagnet[triangles[jElem, :]])
                Zp = -Zmagnet
                
                r = np.sqrt(Xp**2 + Yp**2 + Zp**2)
                coeff = -(mu0 * mu) / (4 * np.pi * r**5)
                phi[i] += coeff * (3 * Zp**2 - r**2) * Wspire * surface[jElem]
        
        phi[i] = phi[i] / Sbobine
        print(f"Iteration {i:2d} : shift = {Xshift[i]:6.3f} [cm] : phi = {phi[i] * 200:.3f}")
    
    return phi

# ------------------------------------------------------------------------------------
# Script de test
# ------------------------------------------------------------------------------------

# Matérial parameters
mu0 = 4e-7 * np.pi * 1e-2  # Perméabilité du vide [H/cm]
Rmagnet = 1.27  # Rayon de l'aimant [cm]
Hmagnet = 0.635  # Épaisseur de l'aimant [cm]
Zmagnet = 0.5  # Position verticale de l'aimant [cm]
Br = 0.267  # Magnétisation résiduelle [T]
mu = Rmagnet**2 * Hmagnet * np.pi * Br / mu0  # Moment magnétique [A cm^2]
Rcoil = 1.2  # Rayon de la bobine [cm]
nSpires = 200  # Nombre de spires

# ------------------------------------------------------------------------------------
# Mesh generation
# ------------------------------------------------------------------------------------
nR = 6
nTheta = 6
nNode = 1 + np.sum(np.arange(1, nR)) * nTheta

R = np.zeros(nNode)
Theta = np.zeros(nNode)
index = 1
dR = 1.0 / (nR - 1)

for i in range(1, nR):
    dTheta = 2 * np.pi / (i * nTheta)
    for j in range(0, i * nTheta):
        R[index] = i * dR
        Theta[index] = j * dTheta
        index += 1

X = R * np.cos(Theta)
Y = R * np.sin(Theta)
triangles = Delaunay(np.stack((X, Y), axis=1)).simplices
nElem = len(triangles)

print(f"Number of triangles: {nElem}")
print(f"Number of nodes: {nNode}")

# ------------------------------------------------------------------------------------
# Compute flux and induced voltage
# ------------------------------------------------------------------------------------
m = 41
Xstart, Xstop = -5, 5  # cm
Xshift = np.linspace(Xstart, Xstop, m)

Tstart, Tstop = 0, 0.5  # s
T, delta = np.linspace(Tstart, Tstop, m, retstep=True)

Xmagnet = Rmagnet * R * np.cos(Theta)
Ymagnet = Rmagnet * R * np.sin(Theta)
Xcoil = Rcoil * R * np.cos(Theta)
Ycoil = Rcoil * R * np.sin(Theta)

phi = magnetComputeInduction(Xmagnet, Ymagnet, Zmagnet, Xcoil, Ycoil, triangles, Xshift, mu0, mu)
phi *= nSpires

voltage = np.diff(phi) / (delta * 10)
