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

def rho_trotter(x_max = 5., nx = 101, beta=1, potential=harmonic_potential):
    """
    Uso:    devuelve matriz densidad en aproximación de Trotter para altas temperaturas y un potencial dado

    Recibe:
        grid: list      -> lista de dimensión N
        beta: float     -> inverso de temperatura en unidades reducidas
        potential: func -> potencial de interacción

    Devuelve:
        matrix          -> matriz densidad de dimension NxN
    """
    dx = 2. * x_max / (nx - 1)
    grid_x = np.array([i*dx for i in range(-int((nx-1)/2), int(nx/2 + 1))])
    rho = np.array([ [ rho_free(x , xp, beta) * np.exp(-0.5*beta*(potential(x)+potential(xp))) for x in grid_x] for xp in grid_x])
    return rho, grid_x, dx

def density_matrix_convolution_trotter(rho, grid_x, N_beta = 4, beta_ini = 1, print_steps=True):
    """
    Uso:    entrega (N_beta-1) convoluciones de la matriz densidad rho(beta_ini), es decir,
            entrega aproximación de rho(N_beta*beta_ini).
    """
    dx = grid_x[1] - grid_x[0]
    N_conv = N_beta-1
    beta_fin = beta_ini * N_beta
    for i in range(N_conv):
        rho = dx * np.dot(rho,rho)
        #rho *= dx
        beta_step = (i+2)*beta_ini
        if print_steps==True:
            print('step %d) beta: %.2E -> %.2E'%(i, (i+1)*beta_ini, beta_step))
    trace_rho = np.trace(rho)*dx
    return rho, trace_rho, beta_fin

def save_pi_x_csv(grid_x, x_weights, file_name, relevant_info, print_data=True):
    # Guardamos datos de pi(x;beta) en archivo .csv usando pandas
    pi_x_data = {'Position x': grid_x,
                'Prob. density': x_weights}
    pi_x_data = pd.DataFrame(data=pi_x_data)
    with open(file_name,mode='w') as rho_csv:
        rho_csv.write(relevant_info+'\n')
    rho_csv.close()
    with open(file_name,mode='a') as rho_csv:
        pi_x_data.to_csv(rho_csv)
    rho_csv.close()
    if print_data==True:
        print(pi_x_data)
    return pi_x_data

x_max = 5.
nx = 101
N_beta = 14
beta_fin = 6
beta_ini = beta_fin/N_beta
potential, potential_string = harmonic_potential, 'harmonic_potential'
rho, grid_x, dx = rho_trotter(x_max = x_max, nx = nx, beta = beta_ini, potential = potential)
rho, trace_rho, beta_fin_2 = density_matrix_convolution_trotter(rho, grid_x, N_beta = N_beta, beta_ini = beta_ini, print_steps=True)
# checkpoint: trace(rho)=0 when N_beta>16 and nx~1000 or nx~100 
# parece que la diferencia entre los picos es siempre constante
# cuando N_beta=4 el resultado es más óptimo
print(trace_rho, beta_fin_2)
rho_normalized = rho/trace_rho          #rho normalizado 
x_weights = np.diag(rho_normalized)     #densidad de probabilidad dada por los elementos de la diagonal
file_name = 'pi_x-%s-x_max_%.3f-nx_%d-N_beta_%d-beta_fin_%.3f.csv'%(potential_string,x_max,nx,N_beta,beta_fin)
relevant_info = u'# %s   x_max = %.3f   nx = %d   N_beta = %d   beta_fin = %.3f'%(potential_string,x_max,nx,N_beta,beta_fin)
save_pi_x_csv(grid_x, x_weights, file_name, relevant_info, print_data=0)

# Figura preliminar
plt.figure()
plt.plot(grid_x, x_weights, label = 'Matrix Convolution +\nTrotter formula')
plt.plot(grid_x, QHO_canonical_ensemble(grid_x,beta_fin), label=u'$\pi^{(Q)}(x;beta)$' )
plt.legend(title=u'$\\beta=%.2E$'%beta_fin)
plt.tight_layout()
plt.show()
plt.close()
