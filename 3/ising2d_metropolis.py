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
from numba import njit, prange
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
                        read_ini_microstate_data=False, L=6,
                        beta=np.array([1/10.,1/2.5,1/0.1]),
                        J=1, N_steps=10000, N_transient=0):
    
    N = L * L
    energies_array = []
    microstate_array = []
    avg_energy_per_spin_array = []

    for i, microstate in enumerate(microstates_ini):
        ising_args = (microstate, read_ini_microstate_data, L, beta[i], J, N_steps, N_transient)
        energies, microstate, avg_E_per_spin_final = ising_metropolis_energies(*ising_args)
        energies_array.append(np.array(energies))
        microstate_array.append(microstate)

    for energies in energies_array:
        avg_energy_per_spin = []
        E_cumulative = 0.
        for i, E in enumerate(energies):
            E_cumulative += E
            E_avg_per_spin = E_cumulative / ((i+1) * N)
            avg_energy_per_spin.append(E_avg_per_spin)
        avg_energy_per_spin_array.append(np.array(avg_energy_per_spin))

    return avg_energy_per_spin_array, beta, energies_array, microstate_array


def plot_thermalization_demo(avg_energy_per_spin_array, beta=np.array([1/10.,1/2.5,1/0.1]),
                             L=6, J=1, N_steps=10000, N_transient=0,
                             thermaization_data_file_name=None, show_plot=True, save_plot=False,
                             plot_file_Name=None, **kwargs):

    plt.figure()
    N_steps = len(avg_energy_per_spin_array[0])
    steps = range(1, N_steps+1)
    for i, E_per_spin in enumerate(avg_energy_per_spin_array):
        plt.plot(steps, E_per_spin, label = '$ T = %.3f$'%(1./beta[i]))
    plt.xlabel('Número de iteraciones (Metrópolis)')
    plt.ylabel('$\langle E \\rangle / N $')
    plt.legend(loc='best',fancybox=True, framealpha=0.65,
               title='$N = L \\times L = %d \\times %d$'%(L, L))
    plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    plt.tight_layout()
    if save_plot:
        if not plot_file_Name:
            temps = 1/np.array(beta)
            T_string = '_'.join([str(T) for T in temps])
            plot_file_Name = ('ising-metropolis-thermalization-plot-L_%d-T_'%L + T_string
                              + '-N_steps_%d-N_transient_%d.pdf'%(N_steps, N_transient))
        plot_file_Name = script_dir + '/' + plot_file_Name
        plt.savefig(plot_file_Name)
    if show_plot:
        plt.show()
    plt.close()
    
    return

@njit(parallel=True)
def energies_momenta_montecarlo(energies, n, N_steps):
    E_n = 0.
    for i in prange(len(energies)):
        E_n += energies[i]**n
    return E_n/N_steps

@njit
def specific_heat_montecarlo(L=6, N_steps=int(1e5), N_transient=int(2.5e4), J=1,
                             T_min = 0.1, T_max=5., N_temp=30):
    T_array = np.linspace(T_min, T_max, N_temp)
    N = L * L
    cv_array = []
    for i in prange(N_temp):
        metropolis_args = (np.ones(N, dtype=np.int64), False, L,
                           1./T_array[i], J, N_steps, N_transient)
        energies, microstate, avg_energy_per_spin = ising_metropolis_energies(*metropolis_args)
        E_1 = energies_momenta_montecarlo(energies, 1, N_steps)
        E_2 = energies_momenta_montecarlo(energies, 2, N_steps)
        cv = (E_2 - E_1**2) / (T_array[i]**2 * N)
        cv_array.append(cv)
    return cv_array, T_array

@njit
def several_specific_heats(L_array=np.array([2, 3, 4, 8]), N_steps_factor=int(5e3),
                           J=1, T_min = 0.1, T_max=5., N_temp=30):
    
    cv_arrays = []
    T_arrays = []
    for i in prange(L_array.shape[0]):
        L = L_array[i]
        N = L * L
        N_steps = int(N * N_steps_factor)
        N_transient = int(N_steps * 0.4)
        cv_args = (L, N_steps, N_transient, J, T_min, T_max, N_temp)
        cv_list, T_list = specific_heat_montecarlo(*cv_args)
        cv_arrays.append(np.array(cv_list))
        T_arrays.append(T_list)
    

    return cv_arrays, T_arrays, L_array, N_steps_factor


def specific_heat_plot(cv_arrays, T_arrays, L_array, N_steps_factor, J=1, 
                       cv_data_file_name=None, show_plot=True,
                       save_plot=False, plot_file_Name=None, **kwargs):
    
    plt.figure()
    for i, cv_T_i in enumerate(cv_arrays):
        plt.plot(T_arrays[i], cv_T_i, '--',
                 label = '$ %d \\times %d $'%(L_array[i], L_array[i]))
    plt.xlabel('$T$')
    plt.ylabel('$c_v$')
    plt.legend(loc='best',fancybox=True, framealpha=0.65, title='$N = L \\times L$')
    plt.ylim(0)
    plt.tight_layout()
    if save_plot:
        if not plot_file_Name:
            L_string = '_'.join([str(L) for L in L_array])
            plot_file_Name = ('ising-metropolis-specific_heat-plot-L_' + L_string
                              + '-N_steps_factor_%d.pdf'%(N_steps_factor))
        plot_file_Name = script_dir + '/' + plot_file_Name
        plt.savefig(plot_file_Name)
    if show_plot:
        plt.show()
    plt.close()
    return



