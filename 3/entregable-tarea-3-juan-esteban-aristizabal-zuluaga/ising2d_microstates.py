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
from scipy.interpolate import interp1d
from scipy.optimize import fmin, curve_fit

# Obtiene path del directorio en que está ubicado este script
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

# Cálculo de microestados
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
    # Éste método es equivalente a generar números binarios remplazando ceros por -1
    # pero es más rápido que hacerlo explícitamente en el algoritmo, ya que los números
    # binarios habría que separarlos y convertirlos de str a int.
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
          + 'Explicit microstates:  L = %d --> computation time = %.7f \n'%(L,comp_time)
          + '----------------------------------------------------------\n')

    return microstates

# Cálculo de vecinos: condiciones de frontera periódicas
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
    ngbrs = {i: [(i//L)*L + (i+1)%L, (i+L) % N,
               (i//L)*L + (i-1)%L, (i-L) % N] for i in range(N)}
    return ngbrs

# Cálculo de vecinos: condiciones de frontera libres
def ising_neighbours_free(L):
    N = L * L
    ngbrs = {}
    for i in range(N):
        ngbrs_i = []
        #nbr up
        if i // L == 0:
            pass
        else:
            ngbrs_i.append(i-L)
        #nbr right
        if (i+1)%L == 0:
            pass
        else:
            ngbrs_i.append(i+1)
        #nbr down:
        if i//L == L-1:
            pass
        else:
            ngbrs_i.append(i+L)
        #ngbr left:
        if (i+1)%L == 1:
            pass
        else:
            ngbrs_i.append(i-1)
        ngbrs[i] = ngbrs_i
    return ngbrs

# Cálculo de energías para cada microestado. Los microestados deben estar 
# contenidos en un arreglo y cada lista dentro del arreglo representa un 
# microestado diferente
def ising_energy(microstates, ngbrs, J=1, save_data=False, data_file_name=None,
                 print_log=True):

    
    energies = []       # lista en que se almacenan energías de cada microestado
    N = len(ngbrs)
    L = int(N**0.5)
    # Se calcula energía para cada microestado
    for microstate_j in microstates:
        energy_j = 0
        for i in range(N):
            energy_j -= microstate_j[i] * np.sum([microstate_j[ngbr] for ngbr in ngbrs[i]])
        energies.append(energy_j)
    
    # En el algoritmo hemos contado cada contribución de energía 2 veces, por tanto se
    # debe hacer corrección. Además se agrega el factor de la fuerza de la interacción.
    energies = 0.5 * J * np.array(energies)

    # Guardamos datos de energías en archivo CSV
    if save_data:
        if not data_file_name:
            data_file_name = 'ising-energy-data-L_%d.csv'%(L)
        data_file_name = script_dir + '/' + data_file_name
        relevant_info = ['2D Ising energies: all microstates. L=%d.'%L]
        headers = ['i-th microstate\'s energy']
        save_csv(energies, data_headers=headers, file_name=data_file_name, 
                 relevant_info=relevant_info, print_data=False)

    return energies

# Lee archivo donde están guardadas las energías de cada microestado.
def read_energy_data(energy_data_file_name):
    energy_data_file_name = script_dir + '/' + energy_data_file_name
    energies = pd.read_csv(energy_data_file_name, index_col=0, comment='#')
    energies = energies.to_numpy(dtype=int)
    energies = energies.transpose()
    energies = energies.tolist()[0]
    return energies

# Convierte la lista de energías de cada microestado en un histograma
# representado por las listas Omega(E) (omegas) y E (energies)
def microstate_energies_to_frequencies(microstate_energies):
    # Se calculan los omegas y las energías usando la librería collections
    energy_omegas = dict(collections.Counter(microstate_energies))
    # Se organizan las energías ascendentemente
    energy_omegas = sorted(energy_omegas.items(), key=lambda kv: kv[0])
    energy_omegas = np.array([list(item) for item in energy_omegas])
    energy_omegas = energy_omegas.transpose()
    energy_omegas = energy_omegas.tolist()
    # Todas las energías diferentes --> energies.
    # Número de veces que se repite cada energía) --> omegas.
    energies, omegas = np.array(energy_omegas[0]), np.array(energy_omegas[1])
    print('--------------------')
    print('Energies and omegas:')
    print('--------------------')
    print(pd.DataFrame({'E': energies, 'Omega(E)': np.array(omegas, dtype=int)}),'\n')
    return energies, omegas

# Grafica un microestado dado
def ising_microstate_plot(config ,show_plot=True, save_plot=False, plot_file_name=None):
    
    L = int(len(config)**0.5)
    bw_cmap = colors.ListedColormap(['black', 'white'])
    
    fig, ax = plt.subplots(1, 1)
    ax.imshow(config.reshape(L,L), cmap=bw_cmap, extent=(0,L,L,0), aspect='equal')
    ax.xaxis.set_ticks_position('top')
    if L<10:
        ax.set_xticks(range(0,L+1))
        ax.set_yticks(range(0,L+1))
    else:
        ax.set_xticks(np.linspace(0, L, 10, dtype=int))
        ax.set_yticks(np.linspace(0, L, 10, dtype=int))
    plt.tight_layout()
    if save_plot:
        if not plot_file_name:
            plot_file_name =  'ising-config-plot-L_%d.pdf'%(L)
        plot_file_name = script_dir + '/' + plot_file_name
        plt.savefig(plot_file_name)
    if show_plot:
        plt.show()
    return

# Grafica histograma de energías Omega(E) vs E
def ising_energy_plot(microstate_energies, L, read_data=False, energy_data_file_name=None,
                      interpolate_energies=True, show_plot=True, save_plot=False,
                      plot_file_Name=None, normed=False, x_lim=[0, 0, 10, 20, 35, 55, 80],
                      y_label='$\Omega(E)$', legend_title=None):
    
    # Lee datos de energías de todos los microestados de archivo de texto
    if read_data:
        if not energy_data_file_name:
            energy_data_file_name = 'ising-energy-data-L_%d.csv'%(L)
        microstate_energies = read_energy_data(energy_data_file_name)
    
    # Pasa energías de microestados a histograma usando función definida en
    # este módulo
    energies, omegas = microstate_energies_to_frequencies(microstate_energies)
    
    # Si se quiere el histograma normado
    if normed:
        omegas = np.array(omegas)/len(microstate_energies)

    E_min = min(energies)
    E_max = max(energies)
    E_plot = np.linspace(E_min, E_max, 100)
    
    # Se grafica el histograma y se guardan si se desea
    plt.plot()
    plt.bar(energies, omegas, width=1,
            label='Histograma energías\nIsing $L\\times L = %d \\times %d$'%(L, L))
    if interpolate_energies:
        omega_interp = interp1d(energies, omegas, kind='cubic')
        plt.plot([0],[0])
        plt.plot(E_plot, omega_interp(E_plot), label='Interpolación splines')
    if not x_lim:
        pass
    else:
        plt.xlim(-1*x_lim[L],x_lim[L])
    plt.xlabel('$E$')
    plt.ylabel(y_label)
    plt.legend(loc='best', fancybox=True, framealpha=0.5, title=legend_title)
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

# Grafica las contribuciones a la función partición Omega(E)e^{-beta E} vs E
def partition_func_stat_weights(microstate_energies, L, beta=4, beta_max=None, N_beta=None,
                                read_data=False, energy_data_file_name=None,
                                plot_histogram=False, show_plot=True, save_plot=False,
                                plot_file_Name=None, **kwargs):
    
    # Lee datos
    if read_data:
        if not energy_data_file_name:
            energy_data_file_name = 'ising-energy-data-L_%d.csv'%(L)
        microstate_energies = read_energy_data(energy_data_file_name)
    
    # Energías y número de microestados asociados a cada una de dichas energías.
    energies, omegas = microstate_energies_to_frequencies(microstate_energies)

    # Para calcular Z en solo un valor de beta (y graficar contribuciones).
    if not beta_max:
        Z_contributions = omegas * np.exp(- beta * energies)
        Z_value = sum(Z_contributions)
        statistical_weights = Z_contributions / Z_value
        if plot_histogram:
            x_lim = [0, 0, 10, 20, 35, 55, 80]
            plt.xlim(-1*x_lim[L],x_lim[L])
            plt.bar(energies, Z_contributions, width=1, 
                    label='Contribuciones a $Z(\\beta)$\nIsing $L\\times L=%d$'%(L*L))
            plt.xlabel('$E$')
            plt.ylabel('$\Omega(E)e^{-\\beta E }$')
            plt.legend(loc='best', fancybox=True, framealpha=0.5, title='$\\beta=%.4f$'%beta)
            plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
            plt.tight_layout()
            if save_plot:
                if not plot_file_Name:
                    plot_file_Name = 'ising-Z_contributions-plot-L_%d-beta_%.4f.pdf'%(L,beta)
                plot_file_Name = script_dir + '/' + plot_file_Name
                plt.savefig(plot_file_Name)
            if show_plot:
                plt.show()
            plt.close()
        return Z_value, statistical_weights, beta, energies, omegas
    # Para calcular Z en varios valores de beta
    else:
        beta_array = np.linspace(beta, beta_max, N_beta)
        Z_array = []
        statistical_weights_array = []
        for beta in beta_array:
            Z_contributions = omegas * np.exp(- beta * energies)
            Z = sum(Z_contributions)
            statistical_weights = Z_contributions / Z
            Z_array.append(Z)
            statistical_weights_array.append(statistical_weights)
        return Z_array, statistical_weights_array, beta_array, energies, omegas


def energies_momenta(microstate_energies, L, n=1, beta_min=0.5, beta_max=None, N_beta=50,
                     read_data=False, energy_data_file_name=None):
    """
    Calcula el enésimo momento de la energía <E^n> en el ensamble canónico
    
    Se puede calcular para varios valores de beta especificando meta max y N_beta
    Se puede calcular también para varios valores de n entregando una lista con 
    los valores de n deseados.
    """

    Z_array, statistical_weights_array, beta_array, energies, omegas = \
        partition_func_stat_weights(microstate_energies, L, beta_min, beta_max,
                                    N_beta, read_data=read_data,
                                    energy_data_file_name=energy_data_file_name)

    if type(n)==int or type(n)==float: n = [n]

    E_n_array = []

    for n_i in n:
        E_n_i = []
        if beta_max:
            for statistical_weights in statistical_weights_array:
                E_n_i.append(sum(energies**n_i * statistical_weights))
            E_n_array.append(E_n_i)
        else:
            E_n_array.append(sum(energies**n_i * statistical_weights_array))

    # Si se especifica solo 1 valor de n:
    if len(E_n_array)==1:
        # Si se calcula para varios beta devolvemos lista con E_n deseado 
        # y lista con valores de beta ó,
        # si se calcula para un solo beta se devuelve un escalar para E_n 
        # y un escalar para beta
        return E_n_array[0], beta_array, Z_array, statistical_weights_array, energies, omegas
    # Si se especifican varios valores de n se devuelve una lista con n listas con
    # valores de E_n apara posiblemente uno o varios valores de beta
    else:
        return E_n_array, beta_array, Z_array, statistical_weights_array, energies, omegas

# Aproximación para la función partición: equivalencia con ensamble canónico
def approx_partition_func(microstate_energies_array=[None, None, None, None],
                          L_array=[ 2, 3, 4, 5], beta_min=0.00001, beta_max=2, N_beta=100,
                          read_data=False, energy_data_file_name=None, plot=True,
                          show_plot=True, save_plot=False, plot_file_Name=None,
                          **kwargs):
    
    # Si se desea graficar
    if plot:
        plt.figure()
        ax = plt.gca()
    
    # Se calcula la aproximación
    for i, L in enumerate(L_array):
        # Se calcula el promedio de E, <E> para los valores de beta deseados.
        E_1_array, beta_array, Z_array, statistical_weights_array, energies, omegas = \
            energies_momenta(microstate_energies_array[i], L, 1, beta_min, beta_max,
                             N_beta, read_data, energy_data_file_name)
        # Se hace interpolación para calcular Omega(<E>).
        omega_interp = interp1d(energies, omegas, kind='linear')
        # Se realiza la aproximación explicitampente omega(<E>) e^{-beta<E>}
        Z_approx_array = omega_interp(E_1_array) * np.exp(-beta_array * np.array(E_1_array))
        # Se grafica si se desea
        if plot:
            color = next(ax._get_lines.prop_cycler)['color']
            plt.plot(beta_array, np.log(Z_array),
                     label='$L\\times L = %d \\times %d$'%(L,L), color=color)
            plt.plot(beta_array, np.log(Z_approx_array), '--', color=color)
    if plot:
        plt.xlabel('$\\beta$')
        plt.ylabel('$\log Z(\\beta)$' '  ó  ' '$\log Z(\\beta)_{appx}$')
        plt.legend(loc='best', fancybox=True, framealpha=0.5)
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        plt.tight_layout()
        if save_plot:
            if not plot_file_Name:
                L_array_str = '_'.join([str(L) for L in L_array])
                plot_file_Name = 'ising-Z_approx-plot-L_' + L_array_str + '.pdf'
            plot_file_Name = script_dir + '/' + plot_file_Name
            plt.savefig(plot_file_Name)
        if show_plot:
            plt.show()
        plt.close()

    return Z_array, Z_approx_array

# Cálculo del calor específico por enumeración exacta
def specific_heat_cv(microstate_energies, L, beta_min=0.1, beta_max=None, N_beta=50,
                     read_data=False, energy_data_file_name=None, **kwargs):
    
    N = L * L
    n = [1,2]
    
    # Se calcula <E> y <E**2>
    energies_momenta_1_2, beta_array, *non_relevant = energies_momenta(microstate_energies, 
                                                        L, n, beta_min, beta_max, N_beta,
                                                        read_data, energy_data_file_name)
    
    avg_E = np.array(energies_momenta_1_2[0])
    avg_E_squared = np.array(energies_momenta_1_2[1])
    
    # Se calcula explicitamente cv = beta**2 * (<E**2> - <E>**2) / N
    sepcific_heat = beta_array**2 * (avg_E_squared - avg_E**2) / N

    return sepcific_heat, beta_array

# Se grafica calor específico para varios valores de L y en función de T
def plot_specific_heat_cv(microstate_energies_array=[None, None, None], L_array=[2, 3, 4],
                          beta_min=0.1, beta_max=10, N_beta=50, read_data=False,
                          energy_data_file_name=None, show_plot=True, save_plot=False,
                          plot_file_Name=None, save_cv_data=True,**kwargs):
    
    cv_arrays = []
    T_arrays = []
    plt.figure()
    # Se calcula cv(T) para diferentes valores de L y se grafican
    for i, L in enumerate(L_array):
        sepcific_heat, beta_array = specific_heat_cv(microstate_energies_array[i], L, beta_min,
                                                     beta_max, N_beta, read_data,
                                                     energy_data_file_name, **kwargs)
        cv_arrays.append(sepcific_heat)
        T_arrays.append(1/np.array(beta_array))
        sh_L = interp1d(1/beta_array, sepcific_heat, kind='cubic')
        max_sh_T = fmin(lambda T: -sh_L(T), 2.5)
        plt.plot(1/beta_array, sepcific_heat,
                 label = '$ N = %d \\times %d$,   $T_c=%.3f$'%(L, L, max_sh_T))
        plt.plot((max_sh_T,max_sh_T),(0,sh_L(max_sh_T)),'--',c='k',lw=0.5)
    plt.xlabel('$T$')
    plt.ylabel('$c_v$')
    plt.ylim(0)
    plt.legend(loc='best',fancybox=True, framealpha=0.5,
               title='$c_v = \\frac{1}{N T^2}\left( \langle E^2 \\rangle - \langle E \\rangle ^2 \\right)$')
    if save_plot:
        if not plot_file_Name:
            Ls_string = '_'.join([str(L) for L in L_array])
            plot_file_Name = 'ising-specific_heat-plot-L_' + Ls_string + '.pdf'
        plot_file_Name = script_dir + '/' + plot_file_Name
        plt.savefig(plot_file_Name)
    if show_plot:
        plt.show()
    plt.close()
    # Se guardan valores de cv en archivo CSV
    if save_cv_data:
        cv_data_file_name = script_dir + '/ising-specific-heat-parte-1.csv'
        relevant_info = ['L = ' + str(np.array(L_array))]
        headers = np.array([ ['Temperature', 'cv (L=%d)'%L] for L in L_array]).flatten()
        shape = (2*len(L_array), len(cv_arrays[0]))
        cv_data = np.array([[T, cv_arrays[i]] for i, T in enumerate(T_arrays)]).reshape(shape)
        save_csv(cv_data.transpose(), data_headers=headers, file_name=cv_data_file_name,
                 relevant_info=relevant_info, print_data=False)

    return

# Demostración de que la asimetría en Omega(E) vs E para L impares se debe a las
# condiciones de frontera periódicas.
def ising_odd_L_energy_asymmetry(L=3, show_plot=True, save_plot=False):
    # Se calculan vecinos
    ngbr = ising_neighbours(L)
    # Se calcula microestado de máxima energía 
    row = np.array([int((-1)**i) for i in range(L)])
    microstate_asymmetry_highest_E = \
        np.array([ row * int((-1)**i) for i in range(L) ]).flatten().tolist()
    # Se calcula microestado de máxima energía
    microstate_asymmetry_lowest_E = [-1 for i in range(L*L)]
    # Máxima energía para caso asimétrico
    print('E_highest = ', *ising_energy([microstate_asymmetry_highest_E], ngbr), '\n\n')
    # Mínima energía para caso asimétrico
    print('E_lowest = ', *ising_energy([microstate_asymmetry_lowest_E], ngbr), '\n\n')
    # Gráfica microestado máxima energía
    ising_microstate_plot(np.array(microstate_asymmetry_highest_E), show_plot, save_plot,
                          plot_file_name='ising-odd_asymmetry_highest_E-L_%d.pdf'%L)
    # Gráfica microestado mínima energía
    ising_microstate_plot(np.array(microstate_asymmetry_lowest_E), show_plot, save_plot,
                          plot_file_name='ising-odd_asymmetry_lowest_E-L_%d.pdf'%L)
    return