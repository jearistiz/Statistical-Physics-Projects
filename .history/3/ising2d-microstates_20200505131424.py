# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from time import time



def ising_microstates(L=2):
    # Tamaño del sistema
    N = L * L
    
    # Lista en la que se guardan explícitamente todos los microestados (2**N en total)
    config = np.array([[0] * N] * 2**N)
    
    # La primera mitad de los microestados
    for i in range(N):
        index_factor = int(2**N  / 2**(i+1))
        for j in range(2**i):
            config[j*index_factor : (j+1)*index_factor, i] = (-1)**j
    
    # La segunda mitad de los microestados son los estados opuestos a los de la primera
    # mitad
    config[int((2 ** N) / 2):,:] = - np.copy(config[:int((2 ** N) / 2), :])

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

print(ising_microstates())


def ising_plot():
    L = 4
    arrayy = np.ones(L,L)
    arrayy[0:2,:] *= -1 
    bw_cmap = colors.ListedColormap(['black', 'white'])
    plt.imshow(arrayy)
    plt.show
    return


ising_plot()