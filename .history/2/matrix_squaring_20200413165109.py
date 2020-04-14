# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from time import time
import pandas as pd

def rho_free(x,xp,beta):
    """
    Uso: devuelve elemento de matriz dsnsidad para el caso de una partícula libre en un toro infinito.
    """
    return (2.*np.pi*beta)**(-0.5) * np.exp(-(x-xp)**2 / (2 * beta) )

def harmonic_potential(x):
    """Devuelve valor del potencial harmónico para una posición x dada"""
    return 0.5*x**2

def anharmonic_potential(x):
    """Devuelve valor de potencial anharmónico para una posición x dada"""
    return 0.5*x**2 - x**3 + x**4

def QHO_canonical_ensemble(x,beta):
    """
    Uso:    calcula probabilidad teórica cuántica de encontrar al osciladoe armónico 
            (presente en un baño térmico) en la posición x.
    
    Recibe:
        x: float            -> posición
        beta: float         -> inverso de temperatura en unidades reducidas beta = 1/T.
    
    Devuelve:
        probabilidad teórica cuántica en posición dada para temperatura T dada. 
    """
    return (np.tanh(beta/2.)/np.pi)**0.5 * np.exp(- x**2 * np.tanh(beta/2.))

def rho_trotter(grid, beta, potential=harmonic_potential):
    """
    Uso:    devuelve matriz densidad en aproximación de Trotter para altas temperaturas y un potencial dado

    Recibe:
        grid: list      -> lista de dimensión N
        beta: float     -> inverso de temperatura en unidades reducidas
        potential: func -> potencial de interacción

    Devuelve:
        matrix          -> matriz densidad de dimension NxN
    """
    return np.array([ [ rho_free(x , xp,beta) * np.exp(-0.5*beta*(potential(x)+potential(xp))) for x in grid] for xp in grid])

x_max = 5.
nx = 101
dx = 2. * x_max / (nx - 1)
grid_x = np.array([i*dx for i in range(-int((nx-1)/2), int(nx/2 + 1))])
N_beta = 4
beta_fin = 4
beta_ini = beta_fin/N_beta
potential, potential_string = harmonic_potential, 'harmonic_potential'
rho = rho_trotter(grid_x,beta_ini/N_beta,potential=harmonic_potential)
for i in range(N_beta):
    rho = np.dot(rho,rho)
    rho *= dx
    beta_ini *= 2.
    print('%d) beta: %.2E -> %.2E'%(i, beta_ini/2,beta_ini))

# checkpoint: trace(rho)=0 when N_beta>16 and nx~1000 or nx~100 
# parece que la diferencia entre los picos es siempre constante
# cuando N_beta=4 el resultado es más óptimo
print (dx)
print(np.trace(rho))

rho_normalized = rho/(np.trace(rho)*dx)     #rho normalizado 
weights = np.diag(rho_normalized)           #densidad de probabilidad dada por los elementos de la diagonal
# Guardamos datos en archivo .csv usando pandas
rho_data = {'Position x': grid_x,
            'Prob. density': weights]}
rho_data = pd.DataFrame(data=rho_data)
rho_csv = open('rho-%s-x_max_%.3f-nx_%d-N_beta_%d-beta_fin_%.3f.csv'%(potential_string,x_max,nx,N_beta,beta_fin),'a')
rho_csv.write('# %s   x_max = %.3f   nx = %d   N_beta = %d   beta_fin = %.3f'%(potential_string,x_max,nx,N_beta,beta_fin))
rho_data.to_csv(rho_csv)
rho_csv.close()

# Figura preliminar
plt.figure()
plt.plot(grid_x, weights, label = 'Matrix Convolution +\nTrotter formula')
plt.plot(grid_x, QHO_canonical_ensemble(grid_x,beta_fin), label=u'$\pi^{(Q)}(x;beta)$' )
plt.legend(title=u'$\\beta=%.2E$'%beta_fin)
plt.tight_layout()
plt.show()
plt.close()
