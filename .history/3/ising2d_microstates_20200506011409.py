# -*- coding: utf-8 -*-
from __future__ import division
import os
from time import time
import datetime
import collections

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pandas as pd


script_dir = os.path.dirname(os.path.abspath(__file__))

def save_csv(data, data_headers=None, data_index=None, file_name=None,
             relevant_info=None, print_data=True):
    """
    Uso:    data debe contener listas que serán las columnas de un archivo CSV que se guardará
            con nombre file_name. relevant_info agrega comentarios en primeras líneas del
            archivo.

    Recibe:
        data: array of arrays, shape=(nx,ny)    ->  cada columna es una columna del archivo.
        data_headers:  numpy array, shape=(ny,) ->  nombres de las columnas
        data_index:  numpy array, shape=(nx,)   ->  nombres de las filas
        file_name: str                      ->  nombre del archivo en el que se guardarán datos.
        relevant_info: list of str          ->  información que se agrega como comentario en
                                                primeras líneas. Cada elemento de esta lista
                                                se agrega como una nueva línea.
        print_data: bool                    ->  decide si imprime datos guardados, en pantalla.
    
    Devuelve:
        data_pdDF: pd.DataFrame             ->  archivo con datos formato "pandas data frame".
        guarda archivo con datos e inforamación relevante en primera línea.
    """
    
    data_pdDF = pd.DataFrame(data, columns=data_headers, index=data_index)

    # Asigna nombre al archivo para que se guarde en el folder en el que está
    # guardado el script que lo usa
    if file_name==None:
        now = datetime.datetime.now()
        #path completa para este script
        file_name = (script_dir + '/' + 'csv-file-%.4d%.2d%.2d%.2d%.2d%.2d.csv'
                                        %(now.year,now.month,now.day,now.hour,
                                          now.minute,now.second))

    # Crea archivo CSV y agrega comentarios relevantes dados como input
    if relevant_info:
        
        # Agregamos información relevante en primeras líneas
        with open(file_name,mode='w') as file_csv:
            for info in list(relevant_info):
                file_csv.write('# '+info+'\n')
        file_csv.close()

        # Usamos pandas para escribir en archivo formato csv.
        with open(file_name,mode='a') as file_csv:
            data_pdDF.to_csv(file_csv)
        file_csv.close()

    else:
        with open(file_name,mode='w') as file_csv:
            data_pdDF.to_csv(file_csv)
        file_csv.close()
    
    # Imprime datos en pantalla.
    if print_data==True:
        print(data_pdDF)

    return data_pdDF


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
    print('\n----------------------------------------------------------\n'
          + 'Explicit microstates:  L = %d --> computation time = %.3f \n'%(L,comp_time)
          + '----------------------------------------------------------\n')

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


def ising_energy(microstates, ngbrs, J=1, save_data=False, data_file_name=None):
    t_0 = time()

    energies = []
    N = len(ngbrs)
    L = int(N**0.5)
    for microstate_j in microstates:
        energy_j = 0
        for i in range(N):
            for ngbr in ngbrs[i]:
                energy_j -= microstate_j[i] * microstate_j[ngbr]
        energies.append(energy_j)
    
    # En el algoritmo hemos contado cada contribución de energía 2 veces, por tanto se
    # debe hacer corrección. Además se agrega el factor de la integral de intercambio.
    energies = 0.5 * J * np.array(energies)

    # Guardamos datos de energías
    if save_data:
        if not data_file_name:
            data_file_name = 'ising-energy-data-L_%d.csv'%(L)
        data_file_name = script_dir + '/' + data_file_name
        relevant_info = ['2D Ising energies: all microstates. L=%d.'%L]
        headers = ['i-th microstate energy']
        save_csv(energies, data_headers=headers, file_name=data_file_name, relevant_info=relevant_info)

    t_1 = time()
    comp_time = t_1-t_0
    print('\n--------------------------------------------------------\n'
          + 'Explicit energies:  L = %d --> computation time = %.3f \n'%(L,comp_time)
          + '--------------------------------------------------------\n')

    return energies


def read_energy_data(energy_data_file_name):
    energy_data_file_name = script_dir + '/' + energy_data_file_name
    energies = pd.read_csv(energy_data_file_name, index_col=0, comment='#')
    energies = energies.to_numpy(dtype=int)
    energies = energies.transpose()
    energies = energies.tolist()[0]
    return energies


def energies_to_frequencies(energies):
    energy_omegas = dict(collections.Counter(energies))
    energy_omegas = sorted(energy_omegas.items(), key=lambda kv: kv[0])
    energy_omegas = np.array([list(item) for item in energy_omegas])
    energy_omegas = energy_omegas.transpose()
    energy_omegas = energy_omegas.tolist()
    energies, omegas = np.array(energy_omegas[0]), np.array(energy_omegas[1])
    return energies, omegas


def ising_microstate_plot(config, show_plot=True, save_plot=False, plot_file_name=None):
    
    L = int(len(config)**0.5)
    bw_cmap = colors.ListedColormap(['black', 'white'])
    
    fig, ax = plt.subplots(1, 1)
    ax.imshow(config.reshape(L,L), cmap=bw_cmap, extent=(0,L,L,0), aspect='equal')
    ax.xaxis.set_ticks_position('top')
    ax.set_xticks(range(0,L+1))
    ax.set_yticks(range(0,L+1))
    plt.tight_layout()
    if save_plot:
        if not plot_file_name:
            now = datetime.datetime.now()
            plot_file_name =  'ising-config-plot-L_%d.pdf'%(L)
        plot_file_name = script_dir + '/' + plot_file_name
        plt.savefig(plot_file_name)
    if show_plot:
        plt.show()
    return


def ising_energy_plot(energies, L, read_data=False, energy_data_file_name=None, show_plot=True, save_plot=False, plot_file_Name=None):
    
    if read_data:
        energies = read_energy_data(energy_data_file_name)
    
    energies, omegas = energies_to_frequencies(energies)

    x_lim = [0, 0, 10, 20, 30, 55, 80]

    plt.xlim(-1*x_lim[L],x_lim[L])
    plt.bar(energies, omegas, width=1, label='Histograma energías\nIsing $L\\times L=%d$'%(L*L))
    plt.xlabel('$E$')
    plt.ylabel('Frecuencia')
    plt.legend(loc='best', fancybox=True, framealpha=0.5)
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    plt.tight_layout()
    if save_plot:
        if not plot_file_Name:
            plot_file_Name = 'ising-energy-plot-L_%d.pdf'%(L)
        plot_file_Name = script_dir + '/' + plot_file_Name
        plt.savefig(plot_file_Name)
    if show_plot:
        plt.show()
    plt.close()

    return


def partition_func(energies, L, beta=4, beta_max=None, N_values=None, read_data=False,
                   energy_data_file_name=None, plot_histogram=False, show_plot=True,
                   save_plot=False, plot_file_Name=None):
    
    # Lee datos
    if read_data:
        if not energy_data_file_name:
            energy_data_file_name = script_dir + '/ising-energy-data-L_%d.csv'%(L)
        energies = read_energy_data(energy_data_file_name)
    
    energies, omegas = energies_to_frequencies(energies)

    # Para calcular Z en solo un valor de beta
    if not beta_max:
        Z_contributions = omegas * np.exp(- beta * energies)
        Z_value = sum(Z_contributions)
        if plot_histogram:
            x_lim = [0, 0, 10, 20, 30, 55, 80]
            plt.xlim(-1*x_lim[L],x_lim[L])
            plt.bar(energies, Z_contributions, width=1, label='Contribuciones a $Z(\\beta)$\nIsing $L\\times L=%d$'%(L*L))
            plt.xlabel('$E$')
            plt.ylabel('$\Omega(E)e^{-\\beta E }$')
            plt.legend(loc='best', fancybox=True, framealpha=0.5, title='$\\beta=%.3f$'%beta)
            plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
            plt.tight_layout()
            if save_plot:
                if not plot_file_Name:
                    plot_file_Name = 'ising-Z_contributions-plot-L_%d.pdf'%(L)
                plot_file_Name = script_dir + '/' + plot_file_Name
                plt.savefig(plot_file_Name)
            if show_plot:
                plt.show()
            plt.close()
        return Z_value, beta
    # Para calcular Z en varios valores de beta
    else:
        beta_array = np.linspace(beta, beta_max, N_values)
        Z_array = []
        for beta in beta_array:
            Z_array.append(sum(omegas * np.exp(- beta * energies)))
        return Z_array, beta_array

def partition_array(energies, L, beta_min=0.1, beta_max=50, N_values=100, read_data=False, energy_data_file_name=None):
    if read_data:
        energies = read_energy_data(energy_data_file_name)
    energies, omegas = energies_to_frequencies(energies)