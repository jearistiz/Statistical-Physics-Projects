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
    config = np.array([[0] * N] * 2**N)
    
    # La primera mitad de los microestados
    # El procedimiento consiste en que en cada paso, para el espín i-esimo se llena 
    # estratégicamente en la mitad los microestados con 1 y en la otra mitad con -1,
    # así se obtienen todas las configuraciones posibles. 
    for i in range(N):
        index_factor = int(2**N  / 2**(i+1))
        for j in range(2**i):
            config[j*index_factor : (j+1)*index_factor, i] = (-1)**j
    
    # La segunda mitad de los microestados son los estados opuestos a los de la primera
    # mitad
    config[int((2 ** N) / 2):,:] = - np.copy(config[:int((2 ** N) / 2), :])
    
    t_1 = time()
    comp_time = t_1 - t_0
    print('L = %d --> computation time = %.3f'%(L,comp_time))

    return config

def ising_neighbours(L=2):
    """
    vecinos del espin i en el formato {i: (derecha, izquierda, abajo, arriba)}

    derecha = número de 'fila' anterior * número de espines por fila 
              + ubicación relativa en fila de espín de la dereha teniendo en cuenta
                condiciones de frontera períodicas (módulo L)
    
    izquierda = número de 'fila' anterior * número de espines por fila 
                + ubicación relativa en fila de espín de la izquierda, teniendo en 
                  cuenta condiciones de frontera períodicas (módulo L)
    
    abajo = posición de espín + número de espines por fila, teniendo en cuenta
                                condiciones de frontera periódicas (módulo N) 
    
    arriba = posición de espín - número de espines por fila, teniendo en cuenta
                                 condiciones de frontera periódicas (módulo N) 
    """
    N  = L * L
    nbr = {i: ((i//L)*L + (i+1)%L, (i+L) % N,
               (i//L)*L + (i-1)%L, (i-L) % N) for i in range(N)}
    return nbr

############
############ test ising microstates
############
print(ising_microstates())


def ising_plot(config, show_plot=True, save_plot=False, plot_file_Name=None):
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

############
############ test ising plot
############

L = 3
arrayy = ising_microstates(L)
print(pd.DataFrame(arrayy))
microstate_plot = 2 ** (L*L) - np.random.randint(1, 2 ** (L*L))
arrayy = arrayy[,:]
print(pd.DataFrame(arrayy))
ising_plot(arrayy, save_plot=True)