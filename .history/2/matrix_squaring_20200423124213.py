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
    """Uso:    devuelve elemento de matriz dsnsidad para el caso de una partícula libre en
    un toro infinito.
    """
    return (2.*np.pi*beta)**(-0.5) * np.exp(-(x-xp)**2 / (2 * beta))

def harmonic_potential(x):
    """Uso: Devuelve valor del potencial armónico para una posición x dada"""
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

def rho_trotter(x_max=5., nx=101, beta=1, potential=harmonic_potential):
    """
    Uso:    devuelve matriz densidad en aproximación de Trotter para altas temperaturas
            y bajo influencia del potencial "potential".

    Recibe:
        x_max: float    -> los valores de x estarán en el intervalo (-x_max,x_max).
        nx: int         -> número de valores de x considerados (igualmente espaciados).
        beta: float     -> inverso de temperatura en unidades reducidas.
        potential: func -> potencial de interacción. Debe ser solo función de x.

    Devuelve:
        rho: numpy array, shape=(nx,nx)     ->  matriz densidad en aproximación de Trotter para
                                                altas temperaturas y  potencial dado.
        grid_x: numpy array, shape=(nx,)    ->  valores de x en los que está evaluada rho.
        dx: float                           ->  separación entre valores contiguos de grid_x
    """
    
    nx = int(nx)
    
    # Si nx es par lo cambiamos al impar más cercano para incluir al 0 en valores de x
    if nx%2 == 0:
        nx = nx + 1
    
    # Valor de la discretización de posiciones según x_max y nx dados como input
    dx = 2 * x_max/(nx-1)

    # Lista de valores de x teniendo en cuenta discretización y x_max
    grid_x = [i*dx for i in range(-int((nx-1)/2),int((nx-1)/2 + 1))]

    # Construcción de matriz densidad dada por aproximación de Trotter
    rho = np.array([[rho_free(x , xp, beta) * np.exp(-0.5*beta*(potential(x)+potential(xp)))
                     for x in grid_x]
                     for xp in grid_x])

    return rho, grid_x, dx

def density_matrix_squaring(rho, grid_x, N_iter=1, beta_ini=1, print_steps=True):
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
        trace_rho: float                    ->  traza de la matriz densidad a temperatura
                                                inversa igual a beta_fin. Por la definición que
                                                tomamos de rho, ésta es equivalente a la función
                                                partición a dicha temperatura.
        beta_fin: float                     ->  temperatura inversa del sistema asociado a rho.
    """
    
    # Valor de discretización de las posiciones
    dx = grid_x[1] - grid_x[0]
    
    # Cálculo del valor de beta_fin según valores beta_ini y N_iter dados como input
    beta_fin = beta_ini * 2 **  N_iter
    
    # Itera algoritmo matrix squaring
    if print_steps:
        print('\nbeta_ini = %.3f'%beta_ini,
                '\n----------------------------------------------------------------')
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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if file_name==None:
        #path completa para este script
        file_name = script_dir + '/' + 'file_name.csv'

    # Crea archivo CSV y agrega comentarios relevantes dados como input
    if relevant_info is not None:
        
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

def run_pi_x_sq_trotter(x_max=5., nx=201, N_iter=7, beta_fin=4, potential=harmonic_potential,
                        potential_string='harmonic_potential', print_steps=True,
                        save_data=True, csv_file_name=None, relevant_info=None,
                        plot=True, save_plot=True, show_plot=True, plot_file_name=None):
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
        file_name: str      ->  nombre de archivo CSV en que se guardan datos. Si valor es None,
                                se guarda con nombre conveniente según parámetros relevantes.
        plot: bool          ->  decide si grafica.
        save_plot: bool     ->  decide si guarda la figura.
        show_plot: bool     ->  decide si muestra la figura en pantalla.
    
    Devuelve:
        rho: numpy array, shape=(nx,nx)     ->  matriz densidad de estado rho a temperatura
                                                inversa igual a beta_fin.
        trace_rho: float                    ->  traza de la matriz densidad a temperatura
                                                inversa igual a beta_fin. Por la definición que
                                                tomamos de "rho", ésta es equivalente a la
                                                función partición en dicha temperatura.
        grid_x: numpy array, shape=(nx,)    ->  valores de x en los que está evaluada rho.
    """
    # Cálculo del valor de beta_ini según valores beta_fin y N_iter dados como input
    beta_ini = beta_fin * 2**(-N_iter)

    # Cálculo de rho con aproximación de Trotter
    rho, grid_x, dx = rho_trotter(x_max, nx, beta_ini, potential)
    grid_x = np.array(grid_x)
    
    # Aproximación de rho con matrix squaring iterado N_iter veces.
    rho, trace_rho, beta_fin_2 = density_matrix_squaring(rho, grid_x, N_iter, 
                                                         beta_ini, print_steps) 
    print('---------------------------------------------------------'
          + '---------------------------------------------------------\n'
          + u'Matrix squaring: beta_ini = %.3f --> beta_fin = %.3f'%(beta_ini, beta_fin_2)
          + u'   N_iter = %d   Z(beta_fin) = Tr(rho(beta_fin)) = %.3E \n'%(N_iter,trace_rho)
          + '---------------------------------------------------------'
          + '---------------------------------------------------------'
          )

    # Normalización de rho a 1 y cálculo de densidades de probabilidad para valores en grid_x.
    rho_normalized = np.copy(rho)/trace_rho
    x_weights = np.diag(rho_normalized)

    
    
    # Guarda datos en archivo CSV.
    script_dir = os.path.dirname(os.path.abspath(__file__)) #path completa para este script

    if save_data:

        # Prepara datos a guardar y headers
        pi_x_data = np.array([grid_x.copy(),x_weights.copy()])
        pi_x_data_headers = ['position_x','prob_density']

        # Nombre del archivo .csv en el que guardamos valores de pi(x;beta_fin).
        if csv_file_name is None:
            csv_file_name = (u'pi_x-ms-%s-beta_fin_%.3f-x_max_%.3f-nx_%d-N_iter_%d.csv'
                             %(potential_string,beta_fin,x_max,nx,N_iter))
        
        csv_file_name = script_dir + '/' + csv_file_name
        
        # Información relevante para agregar como comentario al archivo csv.
        if relevant_info is None:
            relevant_info = ['pi(x;beta_fin) computed using matrix squaring algorithm and'
                             + ' Trotter approximation. Parameters:',
                             u'%s   x_max = %.3f   nx = %d   '%(potential_string,x_max,nx)
                             + u'N_iter = %d   beta_ini = %.3f   '%(N_iter,beta_ini,)
                             + u'beta_fin = %.3f'%beta_fin]
        
        # Guardamos valores  de pi(x;beta_fin) en archivo csv.
        pi_x_data = save_csv(pi_x_data.transpose(), pi_x_data_headers, None, csv_file_name,
                             relevant_info,print_data=0)

    # Gráfica y comparación con teoría
    if plot:

        plt.figure(figsize=(8,5))
        plt.plot(grid_x, x_weights, 
                 label = 'Matrix squaring +\nfórmula de Trotter.\n$N=%d$ iteraciones\n$dx=%.3E$'
                          %(N_iter,dx))
        plt.plot(grid_x, QHO_canonical_ensemble(grid_x,beta_fin), label=u'Valor teórico QHO')
        plt.xlabel(u'x')
        plt.ylabel(u'$\pi^{(Q)}(x;\\beta)$')
        plt.legend(loc='best',title=u'$\\beta=%.2f$'%beta_fin)
        plt.tight_layout()
        
        if save_plot:
            
            if plot_file_name is None:
                plot_file_name = \
                           (u'pi_x-ms-plot-%s-beta_fin_%.3f-x_max_%.3f-nx_%d-N_iter_%d.eps'
                             %(potential_string,beta_fin,x_max,nx,N_iter))
            
            plot_file_name = script_dir + '/' + plot_file_name
            
            plt.savefig(plot_file_name)
        
        if show_plot:
            plt.show()
        plt.close()

    return rho, trace_rho, grid_x

def Z_several_values(temp_min=1./10, temp_max=1/2., N_temp=10, save_Z_csv=True,
                     Z_file_name = None, relevant_info_Z = None, print_Z_data = True,
                     x_max=7., nx=201, N_iter=7, potential = harmonic_potential,
                     potential_string = 'harmonic_potential', print_steps=False,
                     save_pi_x_data=False, pi_x_file_name=None, relevant_info_pi_x=None, 
                     plot=False, save_plot=False, show_plot=False, 
                     pi_x_plot_file_name=None):
    """
    Uso:    calcula varios valores para la función partición, Z, usando operador densidad
            aproximado aproximado por el algoritmo matrix squaring.
    
    Recibe:
        temp_min: float         ->  Z se calcula para valores de beta en (1/temp_min,1/temp_max)
                                    con N_temp valores igualmente espaciados.
        temp_max: float.
        N_temp: int.
        save_Z_csv: bool        ->  decide si guarda valores calculados en archivo CSV.
        Z_file_name: str        ->  nombre del archivo en el que se guardan datos de Z. Si valor
                                    es None, se guarda con nombre conveniente según parámetros
                                    relevantes.
        relevant_info_Z: list   ->  infrmación relevante se añade a primeras líneas del archivo.
                                    Cada str separada por una coma en la lista se añade como una
                                    nueva línea. 
        print_Z_data: bool      ->  imprime datos de Z en pantalla.
        *args: tuple            ->  argumentos de run_pi_x_sq_trotter

    Devuelve:
        Z_data: list, shape=(3,)
        Z_data[0]: list, shape(N_temp,) -> contiene valores de beta en los que está evaluada Z.
        Z_data[1]: list, shape(N_temp,) -> contiene valores de T en los que está evaluada Z.
        Z_data[2]: list, shape(N_temp,) -> contiene valores de Z.
                                           Z(beta) = Z(1/T) =
                                           Z_data[0](Z_data[1]) = Z_data[0](Z_data[2])
    """
    
    # Transforma valores de beta en valores de T y calcula lista de beta.
    beta_max = 1./temp_min
    beta_min = 1./temp_max
    N_temp = int(N_temp)
    beta_array = np.linspace(beta_max,beta_min,N_temp)
    Z = []

    # Calcula valores de Z para valores de beta especificados en beta_array.
    for beta_fin in beta_array:
        rho, trace_rho, grid_x = run_pi_x_sq_trotter(x_max, nx, N_iter, beta_fin, potential,
                                                     potential_string, print_steps,
                                                     save_pi_x_data, pi_x_file_name,
                                                     relevant_info_pi_x, plot, save_plot,
                                                     show_plot, pi_x_plot_file_name)
        Z.append(trace_rho)

    # Calcula el output de la función.
    Z_data = np.array([beta_array.copy(), 1./beta_array.copy(), Z.copy()], dtype=float)

    # Guarda datos de Z en archivo CSV.
    if save_Z_csv == True:

        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        if Z_file_name is None:
            Z_file_name = ('Z-ms-%s-beta_max_%.3f-'%(potential_string,1./temp_min)
                           + 'beta_min_%.3f-N_temp_%d-x_max_%.3f-'%(1./temp_max,N_temp,x_max)
                           + 'nx_%d-N_iter_%d.csv'%(nx, N_iter))
        
        Z_file_name = script_dir + '/' + Z_file_name
        
        if relevant_info_Z is None:
            relevant_info_Z = ['Partition function at several temperatures',
                               '%s   beta_max = %.3f   '%(potential_string,1./temp_min)
                               + 'beta_min = %.3f   N_temp = %d   '%(1./temp_max,N_temp)
                               + 'x_max = %.3f   nx = %d   N_iter = %d'%(x_max,nx, N_iter)]
        
        Z_data_headers = ['beta', 'temperature', 'Z']

        Z_data = save_csv(Z_data.transpose(), Z_data_headers, None, Z_file_name, relevant_info_Z,
                          print_data=False)

    if print_Z_data == True:
        print(Z_data)

    return Z_data

def average_energy(read_Z_data=True, generate_Z_data=False, Z_file_name = None,
                   plot_energy=True, save_plot_E=True, show_plot_E=True,
                   E_plot_name=None,
                   temp_min=1./10, temp_max=1/2., N_temp=10, save_Z_csv=True,
                   relevant_info_Z=None, print_Z_data=True,
                   x_max=7., nx=201, N_iter=7, potential=harmonic_potential,
                   potential_string='harmonic_potential', print_steps=False,
                   save_pi_x_data=False, pi_x_file_name=None, relevant_info_pi_x=None,
                   plot_pi_x=False, save_plot_pi_x=False, show_plot_pi_x=False,
                   plot_pi_x_file_name=None):
    """
    Uso:    calcula energía promedio, E, del sistema en cuestión dado por potential.
            Se puede decidir si se leen datos de función partición o se generan,
            ya que E = - (d/d beta )log(Z).
            

    Recibe:
        read_Z_data: bool       ->  decide si se leen datos de Z de un archivo con nombre
                                    Z_file_name.
        generate_Z_data: bool   ->  decide si genera datos de Z.
        Nota: read_Z_data y generate_Z_data son excluyentes. Se analiza primero primera opción
        Z_file_name: str        ->  nombre del archivo en del que se leerá o en el que se
                                    guardarán datos de Z. Si valor es None, se guarda con nombre
                                    conveniente según parámetros relevantes.
        plot_energy: bool       ->  decide si gráfica energía.
        save_plot_E: bool       ->  decide si guarda gráfica de energía. Nótese que si
                                    plot_energy=False, no se generará gráfica.
        show_plot_E: bool       ->  decide si muestra gráfica de E en pantalla
        E_plot_name: str        ->  nombre para guardar gráfico de E.
        *args: tuple            ->  argumentos de Z_several_values
    
    Devuelve:
        E_avg: list             ->  valores de energía promedio para beta especificados por
                                    beta__read
        beta_read: list
    """

    # Decide si lee o genera datos de Z.
    if read_Z_data:
        Z_file_read =  pd.read_csv(Z_file_name, index_col=0, comment='#')
    elif generate_Z_data:
        t_0 = time()
        Z_data = Z_several_values(temp_min, temp_max, N_temp, save_Z_csv, Z_file_name,
                                  relevant_info_Z, print_Z_data, x_max, nx, N_iter, potential,
                                  potential_string, print_steps, save_pi_x_data, pi_x_file_name,
                                  relevant_info_pi_x, plot_pi_x,save_plot_pi_x, show_plot_pi_x,
                                  plot_pi_x_file_name)
        t_1 = time()
        print('--------------------------------------------------------------------------\n'
              + '%d values of Z(beta) generated   -->   %.3f sec.'%(N_temp,t_1-t_0))
        Z_file_read = Z_data
    else:
        print('Elegir si se generan o se leen los datos para la función partición, Z.\n'
              + 'Estas opciones son mutuamente exluyentes. Si se seleccionan las dos, el'
              + 'algoritmo escoge leer los datos.')
    

    beta_read = Z_file_read['beta']
    temp_read = Z_file_read['temperature']
    Z_read = Z_file_read['Z']

    # Calcula energía promedio.
    E_avg = np.gradient(-np.log(Z_read),beta_read)
    
    # Grafica.
    if plot_energy:

        plt.figure(figsize=(8,5))
        plt.plot(temp_read,E_avg,label=u'$\langle E \\rangle$ via path integral\nnaive sampling')
        plt.plot(temp_read,E_QHO_avg_theo(beta_read),label=u'$\langle E \\rangle$ teórico')
        plt.legend(loc='best')
        plt.xlabel(u'$T$')
        plt.ylabel(u'$\langle E \\rangle$')
        
        if save_plot_E:
        
            script_dir = os.path.dirname(os.path.abspath(__file__))
        
            if E_plot_name is None:
                E_plot_name = ('E-ms-plot-%s-beta_max_%.3f-'%(potential_string,1./temp_min)
                               + 'beta_min_%.3f-N_temp_%d-x_max_%.3f-'%(1./temp_max,N_temp,x_max)
                               + 'nx_%d-N_iter_%d.eps'%(nx, N_iter))
        
            E_plot_name = script_dir + '/' + E_plot_name
        
            plt.savefig(E_plot_name)
        
        if show_plot_E:
            plt.show()
        
        plt.close()
    
    return E_avg, beta_read.to_numpy()

def calc_error(x,xp,dx):
    """
    Uso:    error acumulado en cálculo computacional de pi(x;beta) comparado
            con valor teórico
    """
    x, xp = np.array(x), np.array(xp)
    N = len(x)
    if N != len(xp):
        raise Exception('x y xp deben ser del mismo tamaño.')
    else:        
        return np.sum(np.abs(x-xp))*dx

def optimization(generate_opt_data=True, read_opt_data=False, beta_fin=4, x_max=5, 
                 potential=harmonic_potential, potential_string='harmonic_potential',
                 nx_min=50, nx_max=1000, nx_sampling=50, N_iter_min=1, N_iter_max=20,
                 save_opt_data=False, opt_data_file_name=None, opt_relevant_info=None,
                 plot=True, show_plot=True, save_plot=True, opt_plot_file_name=None):
    """
    Uso:    calcula diferentes valores de error usando calc_error() para encontrar valores de
            dx y beta_ini óptimos para correr el alcoritmo (óptimos = que minimicen error)
    
    Recibe:
        generate_opt_data: bool ->  decide si genera datos para optimización.
        read_opt_data: bool     ->  decide si lee datos para optimización.
        Nota: generate_opt_data y read_opt_data son excluyentes. Se evalúa primero la primera. 
        nx_min: int  
        nx_max: int             ->  se relaciona  con dx = 2*x_max/(nx-1).
        nx_sampling: int        ->  se generan nx mediante range(nx_max,nx_min,-1*nx_sampling).
        N_iter_min: int  
        N_iter_max: int         ->  se relaciona con beta_ini = beta_fin **(-N_iter). Se gereran
                                    valores de N_iter con range(N_iter_max,N_iter_min-1,-1).
        save_opt_data: bool     ->  decide si guarda datos de optimización en archivo CSV.
        opt_data_file_name: str ->  nombre de archivo para datos de optimización.
        plot: bool              ->  decide si grafica optimización.
        show_plot: bool         ->  decide si muestra optimización.
        save_plot: bool         ->  decide si guarda optimización. 
        opt_plot_file_name: str ->  nombre de gráfico de optimización. Si valor es None, se
                                    guarda con nombre conveniente según parámetros relevantes.
    
    Devuelve: 
        error: list, shape=(nb,ndx) ->  valores de calc_error para diferentes valores de dx y
                                        beta_ini. dx incrementa de izquierda a derecha en lista
                                        y beta_ini incrementa de arriba a abajo.
        dx_grid: list, shape=(ndx,)         -> valores de dx para los que se calcula error.
        beta-ini_grid: list, shape=(nb,)    -> valores de beta_ini para los que calcula error.
    """
    
    t_0 = time()
    
    # Decide si genera o lee datos.
    if generate_opt_data:
        N_iter_min = int(N_iter_min)
        N_iter_max = int(N_iter_max)
        nx_min = int(nx_min)
        nx_max = int(nx_max)

        if nx_min%2==1:
            nx_min -= 1
        if nx_max%2==0:
            nx_max += 1
        
        # Crea valores de nx y N_iter (equivalente a generar valores de dx y beta_ini)
        nx_values = range(nx_max,nx_min,-1*nx_sampling)
        N_iter_values = range(N_iter_max,N_iter_min-1,-1)

        dx_grid = [2*x_max/(nx-1) for nx in nx_values]
        beta_ini_grid = [beta_fin * 2**(-N_iter) for N_iter in N_iter_values]
        
        error = []

        # Calcula error para cada valor de nx y N_iter especificado
        # (equivalentemente dx y beta_ini).
        for N_iter in N_iter_values:
            row = []
            for nx in nx_values:
                rho,trace_rho,grid_x = run_pi_x_sq_trotter(x_max, nx, N_iter, beta_fin,
                                                           potential, potential_string,
                                                           False, False, None, None, False,
                                                           False, False, None)
                grid_x = np.array(grid_x)
                dx = grid_x[1]-grid_x[0]
                rho_normalized = np.copy(rho)/trace_rho
                pi_x = np.diag(rho_normalized)
                theoretical_pi_x = QHO_canonical_ensemble(grid_x,beta_fin)
                error_comp_theo = calc_error(pi_x,theoretical_pi_x,dx)
                row.append(error_comp_theo)
            error.append(row)

    elif read_opt_data:
        error =  pd.read_csv(opt_data_file_name, index_col=0, comment='#')
        dx_grid = error.columns.to_numpy()
        beta_ini_grid = error.index.to_numpy()
        error = error.to_numpy()

    else:
        raise Exception('Escoja si generar o leer datos en optimization(.)')

    # Toma valores de error  en cálculo de Z (nan e inf) y los remplaza por
    # el valor de mayor error en el gráfico.
    try:
        error = np.where(np.isinf(error),0,error)
        error = np.where(np.isnan(error),0,error)
        nan_value = 1.3*np.max(error)
        error = np.where(error==0, float('nan'), error)
    except:
        nan_value = 0
    error = np.nan_to_num(error, nan=nan_value, posinf=nan_value, neginf=nan_value)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Guarda datos (solo si fueron generados y se escoje guardar)
    if generate_opt_data and save_opt_data:
        
        if opt_data_file_name is None:
            opt_data_file_name = ('pi_x-ms-opt-%s-beta_fin_%.3f'%(potential_string, beta_fin)
                                  + '-x_max_%.3f-nx_min_%d-nx_max_%d'%(x_max, nx_min, nx_max)
                                  + '-nx_sampling_%d-N_iter_min_%d'%(nx_sampling, N_iter_min)
                                  + '-N_iter_max_%d.csv'%(N_iter_max))
        
        opt_data_file_name = script_dir + '/' + opt_data_file_name

        if opt_relevant_info is None:
            opt_relevant_info = ['Optimization of parameters dx and beta_ini of matrix squaring'
                         + ' algorithm', '%s   beta_fin = %.3f   '%(potential_string, beta_fin)
                         + 'x_max = %.3f   nx_min = %d   nx_max = %d   '%(x_max, nx_min, nx_max)
                         + 'nx_sampling = %d N_iter_min = %d   '%(nx_sampling, N_iter_min)
                         + 'N_iter_max = %d'%(N_iter_max)]
        
        save_csv(error, dx_grid, beta_ini_grid, opt_data_file_name, opt_relevant_info)
    
    t_1 = time()

    # Grafica.
    if plot:

        fig, ax = plt.subplots(1, 1)

        DX, BETA_INI = np.meshgrid(dx_grid, beta_ini_grid)
        cp = plt.pcolormesh(DX,BETA_INI,error)
        plt.colorbar(cp)
        
        ax.set_ylabel(u'$\\beta_{ini}$')
        ax.set_xlabel('$dx$')
        plt.tight_layout()
        
        if save_plot:
            
            if opt_plot_file_name is None:
                opt_plot_file_name = \
                   ('pi_x-ms-opt-plot-%s-beta_fin_%.3f'%(potential_string, beta_fin)
                    + '-x_max_%.3f-nx_min_%d-nx_max_%d'%(x_max, nx_min, nx_max)
                    + '-nx_sampling_%d-N_iter_min_%d'%(nx_sampling, N_iter_min)
                    + '-N_iter_max_%d.eps'%(N_iter_max))
            
            opt_plot_file_name = script_dir + '/' + opt_plot_file_name

            plt.savefig(opt_plot_file_name)

        if show_plot:
            plt.show()

        plt.close()

    comp_time = t_1 - t_0

    return error, dx_grid, beta_ini_grid, comp_time
