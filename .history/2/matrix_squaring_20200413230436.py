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

def density_matrix_squaring(rho, grid_x, N_iter = 1, beta_ini = 1, print_steps=True):
    """
    Uso:    
    """
    dx = grid_x[1] - grid_x[0]
    beta_fin = beta_ini * 2 **  N_iter
    print('\nbeta_ini = %.3f'%beta_ini,
            '\n----------------------------------------------------------------')
    for i in range(N_iter):
        rho = dx * np.dot(rho,rho)
        if print_steps==True:
            print(u'Iteration %d)  2^%d * beta_ini --> 2^%d * beta_ini'%(i, i, i+1))
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


def run_pi_x_squaring(x_max=5., nx=200, N_iter=7, beta_fin=4, potential=harmonic_potential,
                         potential_string =  'harmonic_potential', print_steps=True,
                         save_data=True, plot=True, save_plot=True, show_plot=True):
    beta_ini = beta_fin * 2**(-N_iter)
    # Cálculo de rho con aproximación de Trotter
    rho, grid_x, dx = rho_trotter(x_max, nx, beta_ini, potential)
    # Aproximación de rho con matrix squaring iterado N_iter veces.
    rho, trace_rho, beta_fin_2 = density_matrix_squaring(rho, grid_x, N_iter, 
                                                            beta_ini, print_steps)
    print('----------------------------------------------------------------\n',
           u'beta_fin = %.3f   Z(beta_fin) = Tr(rho(beta_fin)) ≈ %.3E \n'%(beta_fin_2,trace_rho))
    # Normalización de rho y cálculo de densidades de probabilidad para valores en grid_x
    rho_normalized = rho/trace_rho
    x_weights = np.diag(rho_normalized)
    if save_data==True:
        # Nombre del archivo csv en el que guardamos valores de pi(x;beta_fin)
        file_name = u'pi_x-%s-x_max_%.3f-nx_%d-N_iter_%d-beta_fin_%.3f.csv'\
                                            %(potential_string,x_max,nx,N_iter,beta_fin)
        # Información relevante para agregar como comentario al archivo csv
        relevant_info = u'# %s   x_max = %.3f   nx = %d   '%(potential_string,x_max,nx) + \
                        u'N_iter = %d   beta_ini = %.3f   '%(N_iter,beta_ini,) + \
                        u'beta_fin = %.3f'%beta_fin
        # Guardamos valores  de pi(x;beta_fin) en archivo csv
        save_pi_x_csv(grid_x, x_weights, file_name, relevant_info, print_data=0)
    # Gráfica y comparación con teoría
    if plot == True:
        plt.figure(figsize=(8,5))
        plt.plot(grid_x, x_weights, label = 'Matrix squaring +\nfórmula de Trotter.\n$N=%d$ iteraciones\n$dx=%.3E$'%(N_iter,dx))
        plt.plot(grid_x, QHO_canonical_ensemble(grid_x,beta_fin), label=u'Valor teórico QHO')
        plt.xlabel(u'x')
        plt.ylabel(u'$\pi^{(Q)}(x;\\beta)$')
        plt.legend(loc='best',title=u'$\\beta=%.2f$'%beta_fin)
        plt.tight_layout()
        if save_plot==True:
            plot_name = u'pi_x-plot-%s-x_max_%.3f-nx_%d-N_iter_%d-beta_fin_%.3f.eps'\
                                            %(potential_string,x_max,nx,N_iter,beta_fin)
            plt.savefig(plot_name)
        if show_plot==True:
            plt.show()
        plt.close()
    return 0

plt.rcParams.update({'font.size':15})
run_pi_x_squaring(x_max=7., nx=200, N_iter=7, beta_fin=4, potential=harmonic_potential,
                         potential_string =  'harmonic_potential', print_steps=True,
                         save_data=True, plot=True, save_plot=False, show_plot=False)
