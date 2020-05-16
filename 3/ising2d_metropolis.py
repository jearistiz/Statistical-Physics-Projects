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
                                 ising_energy_plot, energies_momenta)


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
        ngbrs.append(np.array([int((i // L) * L + (i + 1) % L),     # derecha
                               int((i + L) % N),                    # abajo
                               int((i // L) * L + (i - 1) % L),     # izquierda
                               int((i - L) % N)]))                  # arriba
    return ngbrs

#Calcula las energías de los microestados que se le entregan.
@njit
def ising_energy(microstates, ngbrs, J=1, save_data=False, data_file_name=None,
                 print_log=True):

    energies = []
    N = len(ngbrs)
    L = int(N**0.5)
    for microstate_j in microstates:
        energy_j = 0
        for i in range(N):
            energy_j -= (microstate_j[i] 
                         * np.sum(np.array([microstate_j[ngbr] for ngbr in ngbrs[i]])))
        energies.append(energy_j)
    # En el algoritmo hemos contado cada contribución de energía 2 veces, por tanto se
    # debe hacer corrección. Además se agrega el factor de la integral de intercambio.
    energies = 0.5 * J * np.array(energies)
    return energies

# Algoritmo metrópolis: muestrea energías del modelo de Ising en ensamble canónico
# a partir de configuración dada o de configuración aleatoria
@njit
def ising_metropolis_energies(microstate=np.ones(36,dtype=np.int64), 
                              read_ini_microstate_data=False, L=6, beta=1., J=1,
                              N_steps=10000, N_transient=100):
    N = L * L
    # Calcula vecinos
    ngbrs = ising_neighbours(L)

    # Si los datos se no se leyeron, genera microestado inicial aleatoriamente
    if read_ini_microstate_data:
        pass
    else: 
        microstate = np.random.choice(np.array([1,-1]), N)
    
    # Calcula energía inicial
    energy = ising_energy([microstate], ngbrs, J=J, print_log=False)[0]
    # Arreglo donde se guardarán energías de los microestados muestreados
    energies = []

    # En el transiente no se guardan las energías,
    # se espera a que el sistema se termalice.
    for i in range(N_transient):
        k = np.random.randint(N)
        delta_E = (2. * J * microstate[k]
                   * np.sum(np.array([microstate[ngbr_i] for ngbr_i in ngbrs[k]])))
        if  np.random.uniform(0,1) < np.exp(-beta * delta_E):
            microstate[k] *= -1
            energy += delta_E
    # Pasado el transiente, se comienzan a guardar las energías
    for i in range(N_steps):
        k = np.random.randint(N)
        delta_E = (2. * J * microstate[k]
                   * np.sum(np.array([microstate[ngbr_i] for ngbr_i in ngbrs[k]])))
        if np.random.uniform(0,1) < np.exp(-beta * delta_E):
            microstate[k] *= -1
            energy += delta_E
        energies.append(energy)
    
    # Se calcula la energía media por espín del microestado final
    N_steps2 = np.array(len(energies),dtype=np.int64)
    avg_energy_per_spin = np.float(np.sum(np.array(energies))/(N_steps2 * N * 1.))

    # Se devuelven las energías, el microestado final y la energía media
    # por espín del microestado final. 
    return energies, microstate, avg_energy_per_spin

# Grafica microestado dado
def ising_metropolis_microstate_plot(config, L, beta, J=1, N_steps=10000, N_transient=100,
                                     show_plot=True, save_plot=False, plot_file_name=None):
    
    if save_plot:
        if not plot_file_name:
            plot_file_name = ('ising-metropolis-config-plot-L_%d-temp_%.3f'%(L, 1./beta)
                              + '-N_steps_%d-N_transient_%d.pdf'%(N_steps, N_transient))
    
    ising_microstate_plot(np.array(config), show_plot, save_plot, plot_file_name)

# Algoritmo que calcula la termalización de los sistemas en ensamble canónico para
# algoritmo metrópolis
@njit
def thermalization_demo(microstates_ini=np.ones((3,36), dtype=np.int64), 
                        read_ini_microstate_data=False, L=6,
                        beta=np.array([1/10.,1/2.5,1/0.1]),
                        J=1, N_steps=10000, N_transient=0):
    
    N = L * L
    energies_array = []
    microstate_array = []
    avg_energy_per_spin_array = []

    # Corre algoritmo metrópolis para cada beta escogido
    for i, microstate in enumerate(microstates_ini):
        ising_args = (microstate, read_ini_microstate_data, L, beta[i], J, N_steps, N_transient)
        energies, microstate, avg_E_per_spin_final = ising_metropolis_energies(*ising_args)
        energies_array.append(np.array(energies))
        microstate_array.append(microstate)

    # Calcula energía promedio por espín para cada uno de los betas y para cada
    # paso N_step por el que pasa el algoritmo. 
    for energies in energies_array:
        avg_energy_per_spin = []
        E_cumulative = 0.
        for i, E in enumerate(energies):
            E_cumulative += E
            E_avg_per_spin = E_cumulative / ((i+1) * N)
            avg_energy_per_spin.append(E_avg_per_spin)
        avg_energy_per_spin_array.append(np.array(avg_energy_per_spin))

    return avg_energy_per_spin_array, beta, energies_array, microstate_array

# Grafica la termalización calculada en la función anterior
def plot_thermalization_demo(avg_energy_per_spin_array, beta=np.array([1/10.,1/2.5,1/0.1]),
                             L=6, J=1, N_steps=10000, N_transient=0,
                             thermaization_data_file_name=None, show_plot=True, save_plot=False,
                             plot_file_Name=None, **kwargs):

    plt.figure()
    # Si L=6 calcula por enumeración exacta <E>/N para los valores de beta deseados
    if L==6:
        Es = np.array([72, 68, 64, 60, 56, 52, 48, 44, 40,
                       36, 32, 28, 24, 20, 16, 12, 8, 4, 0])
        Omegas = np.array([2, 0, 72, 144, 1620, 6048, 35148, 159840, 804078,
                           3846576, 17569080, 71789328, 260434986, 808871328,
                           2122173684, 4616013408, 8196905106, 11674988208,
                           13172279424])
        nn = len(Omegas)-1
        Es_array = np.concatenate((Es[:nn], np.array([0]), -Es[:nn]))
        Omegas_array = np.concatenate((Omegas[:nn], np.array([Omegas[-1]]), Omegas[:nn]))
        E_over_N_array = []
        for beta_i in beta:            
            Z_contrib = Omegas_array * np.exp(-beta_i * Es_array)
            Z = np.sum(Z_contrib)
            p_E = Z_contrib/Z
            E_over_N = np.sum(Es_array * p_E)/(L*L)
            E_over_N_array.append(E_over_N)
            plt.plot([0, N_steps],[E_over_N, E_over_N], '--', c='k')
            print('T = %.4f     <E>/N = %.4f'%(1/beta_i, E_over_N) )
    N_steps = len(avg_energy_per_spin_array[0])
    steps = range(1, N_steps+1)
    # Grafica termalización para cada beta deseado
    for i, E_per_spin in enumerate(avg_energy_per_spin_array):
        if L==6:
            plt.plot(steps, E_per_spin,
                    label = '$ T = %.1f\,\,\\frac{\langle E \\rangle}{N}=%.2f$'
                    %(1./beta[i],E_over_N_array[i]))
        else:
            plt.plot(steps, E_per_spin, label = '$ T = %.1f$'%(1./beta[i]))
    plt.xscale('log')
    plt.xlabel('Número de iteraciones')
    plt.ylabel('$\langle E \\rangle / N $')
    plt.legend(loc=3, fancybox=True, framealpha=0.9,
               title='$N = L \\times L = %d \\times %d$'%(L, L))
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

# Calcula el momento <E**n> deseado.
@njit(parallel=True)
def energies_momenta_montecarlo(energies, n, N_steps):
    E_n = 0.
    for i in prange(len(energies)):
        E_n += energies[i]**n
    return E_n/N_steps

# Calcula calor específico para un L e intervalo para T dado, 
# usando algoritmo metrópolis
@njit
def specific_heat_montecarlo(L=6, N_steps=int(1e5), N_transient=int(2.5e4), J=1,
                             T_min = 0.1, T_max=5., N_temp=30):
    # Temperaturas para las cuales se va a calcular cv
    T_array = np.linspace(T_min, T_max, N_temp)
    N = L * L
    cv_array = []
    # Cálculo de cv para cada temperatura
    for i in prange(N_temp):
        # Corre algoritmo metrópolis
        metropolis_args = (np.ones(N, dtype=np.int64), False, L,
                           1./T_array[i], J, N_steps, N_transient)
        energies, microstate, avg_energy_per_spin = ising_metropolis_energies(*metropolis_args)
        # Calcula promedio de E
        E_1 = energies_momenta_montecarlo(energies, 1, N_steps)
        # Calcula promedio de E**2
        E_2 = energies_momenta_montecarlo(energies, 2, N_steps)
        # calcula cv explícitamente cv = beta**2 * (<E**2> - <E>**2)/N
        cv = (E_2 - E_1**2) / (T_array[i]**2 * N)
        cv_array.append(cv)
    return cv_array, T_array

# Calcula calor específico para varios valores de L y un intervalo de T deseado
@njit
def several_specific_heats(L_array=np.array([2, 3, 4, 8]), N_steps_factor=int(5e3),
                           N_transient_factor=0.7, J=1, T_min = 0.1, T_max=5., N_temp=30):
    
    cv_arrays = []
    T_arrays = []
    for i in prange(L_array.shape[0]):
        L = L_array[i]
        N = L * L
        N_steps = int(N * N_steps_factor)
        N_transient = int(N_steps * N_transient_factor)
        cv_args = (L, N_steps, N_transient, J, T_min, T_max, N_temp)
        cv_list, T_list = specific_heat_montecarlo(*cv_args)
        cv_arrays.append(np.array(cv_list))
        T_arrays.append(T_list)
    

    return cv_arrays, T_arrays, L_array, N_steps_factor

# Grafica calores específicos para diferentes L en función de T.
def specific_heat_plot(cv_arrays, T_arrays, L_array, N_steps_factor,
                       N_transient_factor, T_min, T_max, N_temp, J=1, 
                       read_cv_data_part_1=True, read_cv_data=False,
                       cv_data_file_name=None, show_plot=True,
                       save_plot=False, plot_file_Name=None, **kwargs):
    
    plt.figure()
    if read_cv_data:
        if not cv_data_file_name:
            L_string = '_'.join([str(L) for L in L_array])
            cv_data_file_name = ('ising-metropolis-specific_heat-plot-L_' + L_string
                + '-N_steps_factor_%d-N_transient_factor_%d-T_min_%.3f-T_max_%.3f-N_temp_%d.csv'
                % (N_steps_factor, N_transient_factor, T_min, T_max, N_temp))
        cv_data_file_name = script_dir + '/' + cv_data_file_name
        cv_data = pd.read_csv(cv_data_file_name, index_col=0, comment='#')
        cv_data = cv_data.to_numpy(dtype=np.float).transpose()
        T_arrays = []
        cv_arrays = []
        for i in range(int(len(cv_data)/2)):
            T_arrays.append(cv_data[i * 2])
            cv_arrays.append(cv_data[i * 2 + 1])
    for i, cv_T_i in enumerate(cv_arrays):
        plt.plot(T_arrays[i], cv_T_i, '-*', ms=4,
                 label = '$ %d \\times %d $'%(L_array[i], L_array[i]))
    if read_cv_data_part_1:
        cv_data_file_name = 'ising-specific-heat-parte-1.csv'
        cv_data_file_name = script_dir + '/' + cv_data_file_name
        cv_data = pd.read_csv(cv_data_file_name, index_col=0, comment='#')
        cv_data = cv_data.to_numpy(dtype=np.float).transpose()
        for i in range(int(len(cv_data)/2)):
            plt.plot(cv_data[i*2], cv_data[i * 2 + 1], '--', c='k', lw=1.5)
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
