# -*- coding: utf-8 -*-
from __future__ import division
import os
from time import time
import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pandas as pd



def ising_microstates(L=2):
    # Tamaño del sistema
    N = L * L
    
    t_0 = time()

    # Lista en la que se guardan explícitamente todos los microestados (2**N en total)
    # cada fila del array representará un microestado diferente
    microstates = np.array([[0] * N] * 2**N)
    
    # La primera mitad de los microestados
    # El procedimiento consiste en que en cada paso, para el espín i-esimo se llena 
    # estratégicamente en la mitad los microestados con 1 y en la otra mitad con -1,
    # así se obtienen todas las configuraciones posibles. 
    for i in range(N):
        index_factor = int(2**N  / 2**(i+1))
        for j in range(2**i):
            microstates[j*index_factor : (j+1)*index_factor, i] = (-1)**j
    
    # La segunda mitad de los microestados son los estados opuestos a los de la primera
    # mitad
    microstates[int((2 ** N) / 2):,:] = - np.copy(microstates[:int((2 ** N) / 2), :])
    
    t_1 = time()
    comp_time = t_1 - t_0
    print('Explicit microstates:  L = %d --> computation time = %.3f'%(L,comp_time))

    return microstates

def ising_neighbours(L=2):
    """
    vecinos del espin i en el formato {i: (derecha, izquierda, abajo, arriba)}

    derecha = número de 'fila' anterior * número de espines por fila 
              + ubicación relativa en fila de espín de la dereha teniendo en cuenta
                condiciones de frontera períodicas (módulo L)
    
    abajo = posición de espín + número de espines por fila, teniendo en cuenta
                                condiciones de frontera periódicas (módulo N) 
    
    izquierda = número de 'fila' anterior * número de espines por fila 
                + ubicación relativa en fila de espín de la izquierda, teniendo en 
                  cuenta condiciones de frontera períodicas (módulo L)

    arriba = posición de espín - número de espines por fila, teniendo en cuenta
                                 condiciones de frontera periódicas (módulo N) 
    """
    N  = L * L
    ngbrs = {i: ((i//L)*L + (i+1)%L, (i+L) % N,
               (i//L)*L + (i-1)%L, (i-L) % N) for i in range(N)}
    return ngbrs

def ising_energy(microstates, ngbrs, J=1):
    t_0 = time()

    energies = []
    N = len(ngbrs)
    for microstate_j in microstates:
        energy_j = 0
        for i in range(N):
            for ngbr in ngbrs[i]:
                energy_j -=  microstate_j[i]*microstate_j[ngbr]
        energies.append(energy_j)
    
    # En el algoritmo hemos contado cada contribución de energía 2 veces, por tanto se
    # debe hacer la corrección. Además se agrega el factor de la integral de intercambio.
    energies = 0.5 * J * np.array(energies)

    t_1 = time()
    comp_time = t_1-t_0

    print('Explicit energies:  L = %d --> computation time = %.3f'%(L,comp_time))

    return energies

def ising_config_plot(config, show_plot=True, save_plot=False, plot_file_Name=None):
    L = int(len(config)**0.5)
    bw_cmap = colors.ListedColormap(['black', 'white'])
    fig, ax = plt.subplots(1, 1)
    ax.xaxis.set_ticks_position('top')
    ax.imshow(arrayy.reshape(L,L), cmap=bw_cmap, extent=(0,L,L,0))
    if save_plot:
        if not plot_file_Name:
            now = datetime.datetime.now()
            plot_file_Name =  ('ising-plot-L_%d-date_%.4d%.2d%.2d%.2d%.2d%.2d.eps'
                                        %(L,now.year,now.month,now.day,now.hour,
                                          now.minute,now.second))
        script_dir = os.path.dirname(os.path.abspath(__file__))
        plot_file_Name = script_dir + '/' + plot_file_Name
        plt.savefig(plot_file_Name)
    if show_plot:
        plt.show()
    return

def ising_energy_plot(energies, L, show_plot=True, save_plot=False):
    x_lim = [0, 0, 10, 18, 0, 30, 80]
    plt.xlim(-1*x_lim[L],x_lim[L])
    plt.hist(energies, bins=L**3+1)
    plt.show()
    plt.close()
    return


############
############ test ising plot
############

L = 4
arrayy = ising_microstates(L)
neighbours = ising_neighbours(L)
print(neighbours)
energies = ising_energy(arrayy,neighbours)
plt.hist(energies, bins=L**3+1)
plt.show()
plt.close()
print(pd.DataFrame(arrayy))
microstate_plot = 2 ** (L*L) - np.random.randint(1, 2 ** (L*L))
arrayy = arrayy[microstate_plot,:]
print(pd.DataFrame(arrayy))
ising_config_plot(arrayy, save_plot=False)