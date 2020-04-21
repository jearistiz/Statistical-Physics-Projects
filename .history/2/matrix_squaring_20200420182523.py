# -*- coding: utf-8 -*-
from __future__ import division
import os 
import numpy as np
import matplotlib.pyplot as plt
from time import time
import pandas as pd

# Author: Juan Esteban Aristizabal-Zuluaga
# date: 20200414

def rho_free(x,xp,beta):
    """Uso: devuelve elemento de matriz dsnsidad para el caso de una partícula libre en un toro infinito."""
    return (2.*np.pi*beta)**(-0.5) * np.exp(-(x-xp)**2 / (2 * beta) )

def harmonic_potential(x):
    """Devuelve valor del potencial armónico para una posición x dada"""
    return 0.5*x**2

def anharmonic_potential(x):
    """Devuelve valor de potencial anarmónico para una posición x dada"""
    # return np.abs(x)*(1+np.cos(x)) #el resultado de este potencial es interesante
    return 0.5*x**2 - x**3 + x**4

def QHO_canonical_ensemble(x,beta):
    """
    Uso:    calcula probabilidad teórica cuántica de encontrar al oscilador armónico 
            (inmerso en un baño térmico a temperatura inversa beta) en la posición x.
    
    Recibe:
        x: float            -> posición
        beta: float         -> inverso de temperatura en unidades reducidas beta = 1/T.
    
    Devuelve:
        probabilidad teórica cuántica en posición x para temperatura inversa beta. 
    """
    return (np.tanh(beta/2.)/np.pi)**0.5 * np.exp(- x**2 * np.tanh(beta/2.))

def Z_QHO(beta):
    """Uso: devuelve valor de función de partición para el QHO unidimensional"""
    return 0.5/np.sinh(beta/2)

def E_QHO_avg_theo(beta):
    """Uso: devuelve valor de energía interna para el QHO unidimensional"""
    return 0.5/np.tanh(0.5*beta)

def rho_trotter(x_max = 5., nx = 101, beta=1, potential=harmonic_potential):
    """
    Uso:    devuelve matriz densidad en aproximación de Trotter para altas temperaturas
            y bajo influencia del potencial "potential".

    Recibe:
        x_max: float    -> los valores de x estarán en el intervalo (-x_max,x_max).
        nx: int         -> número de valores de x considerados (igualmente espaciados).
        beta: float     -> inverso de temperatura en unidades reducidas.
        potential: func -> potencial de interacción. Debe ser función de x.

    Devuelve:
        rho: numpy array, shape=(nx,nx)     ->  matriz densidad en aproximación de Trotter para
                                                altas temperaturas y  potencial dado.
        grid_x: numpy array, shape=(nx,)    ->  valores de x en los que está evaluada rho.
        dx: float                           ->  separación entre valores contiguos de grid_x
    """
    # Valor de la discretización de posiciones según x_max y nx dados como input
    dx = 2. * x_max / (nx - 1)
    # Lista de valores de x teniendo en cuenta discretización y x_max
    grid_x = np.array([i*dx for i in range(-int((nx-1)/2), int(nx/2 + 1))])
    # Construcción de matriz densidad dada por aproximación de Trotter
    rho = np.array([ [ rho_free(x , xp, beta) * np.exp(-0.5*beta*(potential(x)+potential(xp))) for x in grid_x] for xp in grid_x])
    return rho, grid_x, dx

def density_matrix_squaring(rho, grid_x, N_iter = 1, beta_ini = 1, print_steps=True):
    """
    Uso:    devuelve matriz densidad luego de aplicarle algoritmo matrix squaring N_iter veces.
            En la primera iteración se usa matriz de densidad dada por el input rho (a 
            temperatura inversa beta_ini); en las siguientes iteraciones se usa matriz densidad 
            generada por la iteración inmediatamente anterior. El sistema asociado a la matriz 
            densidad obtenida (al final de aplicar el algoritmo) está a temperatura inversa
            beta_fin = beta_ini * 2**(N_iter).

    Recibe:
        rho: numpy array, shape=(nx,nx)     ->  matriz densidad discretizada en valores dados
                                                por x_grid.
        grid_x: numpy array, shape=(nx,)    ->  valores de x en los que está evaluada rho.
        N_iter: int                         ->  número de iteraciones del algoritmo.
        beta_ini: float                     ->  valor de inverso de temperatura asociado a la
                                                matriz densidad rho dada como input.
        print_steps: bool                   ->  decide si muestra valores de beta en cada 
                                                iteración.
    
    Devuelve:
        rho: numpy array, shape=(nx,nx)     ->  matriz densidad de estado rho a temperatura 
                                                inversa igual a beta_fin.
        trace_rho: float                    ->  traza de la matriz densidad a temperatura inversa
                                                igual a beta_fin. Por la definición que tomamos
                                                de rho, ésta es equivalente a la función 
                                                partición a dicha temperatura. 
        beta_fin: float                     ->  temperatura inversa del sistema asociado a rho.
    """
    # Valor de discretixación de las posiciones
    dx = grid_x[1] - grid_x[0]
    # Cálculo del valor de beta_fin según valores beta_ini y N_iter dados como input
    beta_fin = beta_ini * 2 **  N_iter
    # Imprime infromación relevante
    if print_steps:
        print('\nbeta_ini = %.3f'%beta_ini,
                '\n----------------------------------------------------------------')
    # Itera algoritmo matrix squaring
    for i in range(N_iter):
        rho = dx * np.dot(rho,rho)
        # Imprime información relevante
        if print_steps:
            print(u'Iteración %d)  2^%d * beta_ini --> 2^%d * beta_ini'%(i, i, i+1))
    if print_steps:
        print('----------------------------------------------------------------\n' + 
                u'beta_fin = %.3f'%beta_fin)
    # Calcula traza de rho
    trace_rho = np.trace(rho)*dx
    return rho, trace_rho, beta_fin

def save_csv(data, data_headers=None, file_name='file.csv', relevant_info=None, print_data=True):
    """
    Uso:    data debe contener listas que serán las columnas de un archivo CSV que se guardará con
            nombre file_name. relevant_info agrega comentarios en primeras líneas del archivo. 

    Recibe:
        data: array of arrays, shape=(nx,ny)  ->  cada lista es una columna del archivo.
        data_headers:  numpy array, shape=(nx,)     ->  nombres de las columnas
        file_name: str                      ->  nombre del archivo en el que se guardarán datos.
        relevant_info: list of str          ->  información que se agrega como comentario en 
                                                primeras líneas. Cada elemento de esta lista 
                                                se agrega como una nueva línea.
        print_data: bool                    ->  decide si imprime datos guardados, en pantalla.
    
    Devuelve:
        data_pdDF: pd.DataFrame             ->  archivo con datos formato "pandas data frame".
        guarda archivo con datos e inforamación relevante en primera línea.                                                 
    """
    # Almacena datos de probabilifad en diccionario: grid_x para posiciones y x_weights para
    # valores de densidad de probabilidad.
    if file_name=='file.csv':
        script_dir = os.path.dirname(os.path.abspath(__file__)) #path completa para este script
        file_name = script_dir + '/' + 'file_name'
    if len(data_headers)!=len(data) or data_headers is None:
        data_headers = range(len(data))
        print(  'Nota: no hay suficientes headers en data_headers para función save_csv().\n'+
                'Los headers usados en el archivo serán los números 0, 1, 2,...')
    data_dict = {}
    for i,column in enumerate(data):
        data_dict[data_headers[i]] = column
    # Pasamos datos a formato DataFrame de pandas.
    data_pdDF = pd.DataFrame(data=data_dict)
    # Crea archivo .csv y agrega comentarios relevantes dados como input
    if relevant_info is not None:
        with open(file_name,mode='w') as file_csv:
            for info in list(relevant_info):
                file_csv.write('# '+info+'\n')
        file_csv.close()
        # Usamos pandas para escribir en archivo en formato csv.
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

def run_pi_x_sq_trotter(x_max=5., nx=201, N_iter=7, beta_fin=4, potential=harmonic_potential,
                         potential_string =  'harmonic_potential', print_steps=True,
                         save_data=True, file_name=None, relevant_info=None,
                         plot=True, save_plot=True, show_plot=True):
    """
    Uso:    corre algoritmo matrix squaring iterativamente (N_iter veces). En la primera
            iteración se usa una matriz densidad en aproximación de Trotter a temperatura
            inversa beta_ini = beta_fin * 2**(-N_iter) para potencial dado por potential;
            en las siguientes iteraciones se usa matriz densidad generada por la iteración 
            inmediatamente anterior. Además ésta función guarda datos de pi(x;beta) vs. x
            en archivo de texto y grafica pi(x;beta) comparándolo con teoría para el oscilador 
            armónico cuántico.

    Recibe:
        x_max: float        ->  los valores de x estarán en el intervalo (-x_max,x_max).
        nx: int             ->  número de valores de x considerados.
        N_iter: int         ->  número de iteraciones del algoritmo matrix squaring.
        beta_ini: float     ->  valor de inverso de temperatura que queremos tener al final de
                                aplicar el algoritmo matrix squaring iterativamente. 
        potential: func     ->  potencial de interacción usado en aproximación de trotter. Debe 
                                ser función de x.
        potential_string: str   ->  nombre del potencial (con éste nombramos los archivos que
                                    se generan).
        print_steps: bool   ->  decide si imprime los pasos del algoritmo matrix squaring.
        save_data: bool     ->  decide si guarda los datos en archivo .csv.
        plot: bool          ->  decide si grafica.
        save_plot: bool     ->  decide si guarda la figura.
        show_plot: bool     ->  decide si muestra la figura en pantalla. 
    
    Devuelve:
        rho: numpy array, shape=(nx,nx)     ->  matriz densidad de estado rho a temperatura 
                                                inversa igual a beta_fin.
        trace_rho: float                    ->  traza de la matriz densidad a temperatura inversa
                                                igual a beta_fin. Por la definición que tomamos
                                                de "rho", ésta es equivalente a la función 
                                                partición en dicha temperatura.
        grid_x: numpy array, shape=(nx,)    ->  valores de x en los que está evaluada rho.
    """
    # Cálculo del valor de beta_ini según valores beta_fin y N_iter dados como input
    beta_ini = beta_fin * 2**(-N_iter)
    # Cálculo de rho con aproximación de Trotter
    rho, grid_x, dx = rho_trotter(x_max, nx, beta_ini, potential)
    # Aproximación de rho con matrix squaring iterado N_iter veces.
    rho, trace_rho, beta_fin_2 = density_matrix_squaring(   rho, grid_x, N_iter, 
                                                            beta_ini, print_steps   )
    print(  '----------------------------------------------------------------' +
            '--------------------------------------------------------\n'
            u'Matrix squaring: beta_ini = %.3f --> beta_fin = %.3f'%(beta_ini, beta_fin_2) +
            u'   N_iter = %d   Z(beta_fin) = Tr(rho(beta_fin)) = %.3E'%(N_iter,trace_rho))
    # Normalización de rho a 1 y cálculo de densidades de probabilidad para valores en grid_x.
    rho_normalized = np.copy(rho)/trace_rho
    x_weights = np.diag(rho_normalized)
    # Guarda datos en archivo .csv.
    script_dir = os.path.dirname(os.path.abspath(__file__)) #path completa para este script
    if save_data==True:
        # Nombre del archivo .csv en el que guardamos valores de pi(x;beta_fin).
        if file_name is None:
            csv_file_name = script_dir+u'/pi_x-ms-%s-x_max_%.3f-nx_%d-N_iter_%d-beta_fin_%.3f.csv'\
                                            %(potential_string,x_max,nx,N_iter,beta_fin)
        else:
            csv_file_name = script_dir + u'/pi_x-ms-' + file_name +'.csv'
        # Información relevante para agregar como comentario al archivo csv.
        if relevant_info is None:
            relevant_info = [   'pi(x;beta_fin) computed using matrix squaring algorithm and' + \
                                ' Trotter approximation. Parameters:',
                                u'%s   x_max = %.3f   nx = %d   '%(potential_string,x_max,nx) + \
                                u'N_iter = %d   beta_ini = %.3f   '%(N_iter,beta_ini,) + \
                                u'beta_fin = %.3f'%beta_fin ]
        # Guardamos valores  de pi(x;beta_fin) en archivo csv.
        pi_x_data = [grid_x.copy(),x_weights.copy()]
        pi_x_data_headers = ['position_x','prob_density']
        pi_x_data = save_csv(pi_x_data,pi_x_data_headers,csv_file_name,relevant_info,print_data=0)

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
            if file_name is None:
                plot_file_name = script_dir+u'/pi_x-ms-plot-%s-x_max_%.3f-nx_%d-N_iter_%d-beta_fin_%.3f.eps'%(potential_string,x_max,nx,N_iter,beta_fin)
            else:
                plot_file_name = script_dir+u'/pi_x-ms-plot-'+file_name+'.eps'
            plt.savefig(plot_file_name)
        if show_plot==True:
            plt.show()
        plt.close()
    return rho, trace_rho, grid_x

def Z_several_values(   temp_min=1./10, temp_max=1/2., N_temp=10, save_Z_csv=True,
                        Z_file_name = 'Z.csv', relevant_info_Z = None, print_Z_data = True,
                        x_max=7., nx=201, N_iter=7, potential = harmonic_potential,
                        potential_string = 'harmonic_potential', print_steps=False,
                        save_pi_x_data=False, pi_x_file_name=None, relevant_info_pi_x=None, 
                        plot=False, save_plot=False, show_plot=False   ):
    """
    """

    beta_max = 1./temp_min
    beta_min = 1./temp_max
    N_temp = int(N_temp)
    beta_array = np.linspace(beta_max,beta_min,N_temp)
    Z = []

    for beta_fin in beta_array:
        rho, trace_rho, grid_x = \
        run_pi_x_sq_trotter( x_max, nx, N_iter, beta_fin, potential, potential_string,
                             print_steps, save_pi_x_data, file_name, relevant_info, plot,
                             save_plot, show_plot)
        Z.append(trace_rho)

    Z_data = [beta_array.copy(),1./beta_array.copy(),Z.copy()]
    if Z_file_name == 'Z.csv':
        script_dir = os.path.dirname(os.path.abspath(__file__)) #path completa para este script
        Z_file_name = script_dir + '/' + Z_file_name
    if save_Z_csv == True:
        Z_data_headers = ['beta','temperature','Z']
        Z_data = save_csv(  Z_data, Z_data_headers, Z_file_name, relevant_info_Z,
                            print_data = False  )
    if print_Z_data == True:
        print(Z_data)
    return Z_data

def calculate_average_energy()

plt.rcParams.update({'font.size':15}) # Agranda tamaño de fuente en texto de figuras generadas
script_dir = os.path.dirname(os.path.abspath(__file__)) #path completa para este script

# Corre algoritmo matrix squaring
run_ms_algorithm = False
# Parámetros físicos del algoritmo
x_max = 5.
nx = 201
N_iter = 7
beta_fin = 4
potential, potential_string =  harmonic_potential, 'harmonic_potential'
# Parámetros técnicos
print_steps = True
save_data = True
file_name = None
relevant_info = None
plot = True
save_plot = True
show_plot = True
if run_ms_algorithm:
    rho, trace_rho, grid_x = \
        run_pi_x_sq_trotter( x_max, nx, N_iter, beta_fin, potential, potential_string,
                             print_steps, save_data, file_name, relevant_info, plot,
                             save_plot, show_plot)


# Algoritmo para cálculo de energía interna
# Parte1: algoritmo para cálculo de función de partición para varios valores de beta interna
calculate_several_Z = True

temp_min = 1./10
temp_max = 1./2
N_temp = 10
save_Z_csv = True
Z_file_name = script_dir+'/'+'partition-function-test-2.csv'
relevant_info_Z = None
print_Z_data = False
x_max=7.
nx=201
N_iter=7
potential, potential_string = harmonic_potential,  'harmonic_potential'
print_steps=False
save_pi_x_data=False
pi_x_file_name=None
relevant_info_pi_x=None
plot=False
save_plot = False
show_plot=False

if calculate_several_Z:
    t_0 = time()
    Z_data = Z_several_values(  temp_min, temp_max, N_temp, save_Z_csv, Z_file_name,
                                relevant_info_Z, print_Z_data, x_max, nx, N_iter, potential,
                                potential_string, print_steps, save_pi_x_data, pi_x_file_name,
                                relevant_info_pi_x, plot, save_plot, show_plot   )
    t_1 = time()
    print(  '--------------------------------------------------------\n'+
            '%d values of Z(beta)   -->   %.3f sec.'%(N_temp,t_1-t_0))

# READ DATA IS OK
Z_file_name = script_dir+'/'+'partition-function-test-2.csv'
Z_file_read =  pd.read_csv(Z_file_name, index_col=0, comment='#')
beta_read = Z_file_read['beta']
beta_read = beta_read.to_numpy()
temp_read = Z_file_read['temperature']
temp_read = temp_read.to_numpy()
Z_read = Z_file_read['Z']
Z_read = Z_read.to_numpy()

E_avg = np.gradient(-np.log(Z_read),beta_read)


plt.figure()
plt.plot(temp_read,E_avg,label=u'$< E > Path Integral$')
plt.plot(temp_read,E_QHO_avg_theo(beta_read),label=u'$< E > theory$')
plt.plot(temp_read,Z_read,'v-',label=u'$ Z(T) Matrix Squaring $')
plt.plot(temp_read,Z_QHO(beta_read),'^-',label=u'$ Z(T) Matrix Squaring $')
plt.legend(loc='best')
plt.xlabel(u'$T$')
plt.ylabel(u'$< E >$ or $Z(T)$')
plt.show()
plt.close()