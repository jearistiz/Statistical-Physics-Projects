# -*- coding: utf-8 -*-
from __future__ import division
import os
import datetime
import collections
from time import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pandas as pd
from numba import njit
from numba.typed import List

from ising2d_microstates import (script_dir, ising_microstate_plot, save_csv,
                                 ising_energy_plot)


@njit
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
    ngbrs = []
    for i in range(N):
        ngbrs.append(np.array([int((i//L)*L + (i+1)%L), int((i+L) % N), int((i//L)*L + (i-1)%L), int((i-L) % N)]))
    return ngbrs


@njit
def ising_energy(microstates, ngbrs, J=1, save_data=False, data_file_name=None,
                 print_log=True):

    energies = []
    N = len(ngbrs)
    L = int(N**0.5)
    for microstate_j in microstates:
        energy_j = 0
        for i in range(N):
            energy_j -= microstate_j[i] * np.sum(np.array([microstate_j[ngbr] for ngbr in ngbrs[i]]))
        energies.append(energy_j)
    # En el algoritmo hemos contado cada contribución de energía 2 veces, por tanto se
    # debe hacer corrección. Además se agrega el factor de la integral de intercambio.
    energies = 0.5 * J * np.array(energies)
    return energies

@njit
def ising_metropolis_energies(microstate=np.ones(36,dtype=np.int64), 
                              read_ini_microstate_data=False, L=6, beta=1., J=1,
                              N_steps=10000, N_transient=100):
    
    N = L * L
    ngbrs = ising_neighbours(L)
    
    if read_ini_microstate_data:
        pass
    else: 
        microstate = np.random.choice(np.array([1,-1]), N)
    
    energy = ising_energy([microstate], ngbrs, J=J, print_log=False)[0]
    energies = []
    # Transiente
    for i in range(N_transient):
        k = np.random.randint(N)
        delta_E = 2. * J  * microstate[k] * np.sum(np.array([microstate[ngbr_i] for ngbr_i in ngbrs[k]]))
        if  np.random.uniform(0,1) < np.exp(-beta * delta_E):
            microstate[k] *= -1
            energy += delta_E
    # Se comienzan a guardar las energías
    for i in range(N_steps):
        k = np.random.randint(N)
        delta_E = 2. * J * microstate[k] * np.sum(np.array([microstate[ngbr_i] for ngbr_i in ngbrs[k]]))
        if np.random.uniform(0,1) < np.exp(-beta * delta_E):
            microstate[k] *= -1
            energy += delta_E
        energies.append(energy)
    N_steps2 = np.array(len(energies),dtype=np.int64)
    avg_energy_per_spin = np.float(np.sum(np.array(energies))/(N_steps2 * N * 1.))

    return energies, microstate, avg_energy_per_spin


def ising_metropolis_microstate_plot(config, L, beta, J=1, N_steps=10000, N_transient=100,
                                     show_plot=True, save_plot=False, plot_file_name=None):
    
    if save_plot:
        if not plot_file_name:
            plot_file_name = ('ising-metropolis-config-plot-L_%d-temp_%.3f'%(L, 1./beta)
                              + '-N_steps_%d-N_transient_%d.pdf'%(N_steps, N_transient))
    
    ising_microstate_plot(np.array(config), show_plot, save_plot, plot_file_name)

@njit
def thermalization_demo(microstates_ini=np.ones((3,36), dtype=np.int64), 
                        L=6, beta=np.array([1/10.,1/2.5,1/0.1]),
                        J=1, N_steps=10000, N_transient=100):
    energies_array = []
    microstate_array = []
    avg_energy_per_spin_array = []

    for microstate in microstates_ini:
        ising_args = (microstate, False, L, beta, J, N_steps, N_transient)
        energies, microstate, avg_energy_per_spin = ising_metropolis_energies(*ising_args)
        energies_array.append(energies)
        microstate_array.append(microstate)
        avg_energy_per_spin.append(microstate)

    for energies in energies_array:
        avg_energy_per_spin.append(np.array([np.sum(energies[:])]))


    return

