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
    # return np.abs(x)*(1+np.cos(x)) #el resultado de este potencial es interesante
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
    Uso:    devuelve matriz densidad en aproximación de Trotter para altas temperaturas
            y bajo el potencial "potential".

    Recibe:
        xmax: float     -> los valores de x estarán en el intervalo (-xmax,xmax).
        nx: int         -> número de valores de x considerados.
        beta: float     -> inverso de temperatura en unidades reducidas.
        potential: func -> potencial de interacción, debe ser función de x.

    Devuelve:
        rho: numpy array, shape=(nx,nx)     ->  matriz densidad en aproximación de Trotter para
                                                altas temperaturas y  potencial dado.
        grid_x: numpy array, shape=(nx,)    ->  valores de x en los que está evaluada rho.
        dx: float                           ->  separación entre valores contiguos de grid_x
    """
    dx = 2. * x_max / (nx - 1)
    grid_x = np.array([i*dx for i in range(-int((nx-1)/2), int(nx/2 + 1))])
    rho = np.array([ [ rho_free(x , xp, beta) * np.exp(-0.5*beta*(potential(x)+potential(xp))) for x in grid_x] for xp in grid_x])
    return rho, grid_x, dx

def density_matrix_squaring(rho, grid_x, N_iter = 1, beta_ini = 1, print_steps=True):
    """
    Uso:    devuelve matriz densidad luego de aplicarle algoritmo matrix squaring N_iter veces.
            El sistema asociado a la matriz densidad obtenida (al final de aplicar el algoritmo)
            está a temperatura inversa beta_fin = beta_ini * 2**(N_iter).

    Recibe:
        rho: numpy array, shape=(nx,nx)     ->  matriz densidad en aproximación de Trotter para
                                                altas temperaturas y  potencial dado.
        grid_x: numpy array, shape=(nx,)    ->  valores de x en los que está evaluada rho.
        N_iter: int                         ->  número de iteraciones del algoritmo.
        beta_ini: float                     ->  valor de inverso de temperatura asociado a la
                                                matriz densidad rho.
        print_steps: bool                   ->  muestra valores de beta en cada iteración
    
    Devuelve:
        rho: numpy array, shape=(nx,nx)     ->  matriz densidad de estado rho a temperatura 
                                                inversa igual a beta_fin.
        trace_rho: int                      ->  traza de la matriz densidad a temperatura inversa
                                                igual a beta_fin. Por la definición que tomamos
                                                de "rho", ésta es equivalente a la función 
                                                partición en dicha temperatura. 
        beta_fin: float                     ->  temperatura inversa del sistema asociado a rho.
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
    """
    Uso: guarda datos de la distribución pi(x;beta) en un archivo .csv

    Recibe:
        grid_x: numpy array, shape=(nx,)    ->  valores de x en los que está evaluada pi(x;beta).
        x_weights: numpy array, shape=(nx,) ->  valores de pi(x;beta) para cada x en grid_x
        file_name: str                      ->  nombre del archivo en el que se guardarán datos.
        relevant_info: str                  ->  información que se agrega como comentario en 
                                                primera línea.
        print_data: bool                    ->  decide si imprime datos guardados, en pantalla.
    
    Devuelve:
        pi_x_data: pd.DataFrame             ->  valores de pi(x;beta) para x en grid_x en formato
                                                "pandas".
    """
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

def run_pi_x_squaring(x_max=5., nx=201, N_iter=7, beta_fin=4, potential=harmonic_potential,
                         potential_string =  'harmonic_potential', print_steps=True,
                         save_data=True, plot=True, save_plot=True, show_plot=True):
    """
    Uso:    corre algoritmo matrix squaring, guarda datos en archivo de texto y grafica 
            pi(x;beta) comparándolo con teoría para el oscilador armónico cuántico.

    Recibe:
        xmax: float         ->  los valores de x estarán en el intervalo (-xmax,xmax).
        nx: int             ->  número de valores de x considerados.
        N_iter: int         ->  número de iteraciones del algoritmo matrix squaring.
        beta_ini: float     ->  valor de inverso de temperatura que queremos tener al final de
                                aplicar el algoritmo matrix squaring. 
        potential: func     ->  potencial de interacción para aproximación de trotter, debe ser 
                                función de x.
        potential_string: str   ->  nombre del potencial (indexamos los archivos que se generan).
        print_steps: bool   ->  decide si imprime los pasos del algoritmo matrix squaring.
        save_data: bool     ->  decide si guarda los datos en archivo .csv.
        plot: bool          ->  decide si grafica.
        save_plot: bool     ->  decide si guarda la figura.
        show_plot: bool     ->  decide si muestra la figura en pantalla. 
    """
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
        # Nombre del archivo .csv en el que guardamos valores de pi(x;beta_fin)
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
run_pi_x_squaring(potential = harmonic_potential, potential_string =  'harmonic_potential',
                                            save_data=True, save_plot=True, show_plot=True)
