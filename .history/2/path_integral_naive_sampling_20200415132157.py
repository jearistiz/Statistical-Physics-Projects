# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from time import time
import pandas as pd

# Author: Juan Esteban Aristizabal-Zuluaga
# date: 202004151200

def rho_free(x,xp,beta):
    """Uso: devuelve elemento de matriz dsnsidad para el caso de una partícula libre en un toro infinito."""
    return (2.*np.pi*beta)**(-0.5) * np.exp(-(x-xp)**2 / (2 * beta) )

def harmonic_potential(x):
    """Devuelve valor del potencial armónico para una posición x dada"""
    return 0.5*x**2

def anharmonic_potential(x):
    """Devuelve valor de potencial anarmónico para una posición x dada"""
    # return np.abs(x)*(1+np.cos(x)) #el resultado de este potencial es interesante
    return 0.5*x**2 - x**3 + x**4

def QHO_canonical_ensemble(x,beta):
    """
    Uso:    calcula probabilidad teórica cuántica de encontrar al oscilador armónico 
            (inmerso en un baño térmico a temperatura inversa beta) en la posición x.
    
    Recibe:
        x: float            -> posición
        beta: float         -> inverso de temperatura en unidades reducidas beta = 1/T.
    
    Devuelve:
        probabilidad teórica cuántica en posición x para temperatura inversa beta. 
    """
    return (np.tanh(beta/2.)/np.pi)**0.5 * np.exp(- x**2 * np.tanh(beta/2.))

beta = 4.
N = 10
dtau = beta/N
n_steps = int(1e5)
delta = 1.
path_x = [0.] * N
pathss_x = [path_x[:]]
potential = harmonic_potential
append_every = 10
for step in range(n_steps):
    k = np.random.randint(0,N-1)
    knext, kprev = (k+1) % N, (k-1) 
    x_new = path_x[k] + np.random.uniform(-delta,delta)
    old_weight = (  rho_free(path_x[kprev],path_x[k],dtau) * 
                    np.exp(- dtau * potential(path_x[k]))  *
                    rho_free(path_x[k],path_x[knext],dtau)  )
    new_weight = (  rho_free(path_x[kprev],path_x[k],dtau) * 
                    np.exp(- dtau * potential(x_new))  *
                    rho_free(path_x[k],path_x[knext],dtau)  )
    if np.random.uniform() < new_weight/old_weight:
        path_x[k] = x_new
    if step%append_every == 0:
        pathss_x.append(path_x)

pathss_x = np.array(pathss_x)
N_plot = 201
x_max = 3
x_plot = np.linspace(-x_max,x_max,N_plot)
plt.plot()
plt.hist(pathss_x[:,0],bins=int(np.sqrt(N/append_every)),normed=True)
plt.()
plt.show()