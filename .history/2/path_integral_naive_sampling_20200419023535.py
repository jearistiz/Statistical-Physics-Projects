# -*- coding: utf-8 -*-
from __future__ import division
import os 
import numpy as np
import matplotlib.pyplot as plt
from time import time
import pandas as pd

# Author: Juan Esteban Aristizabal-Zuluaga
# date: 202004151200

def rho_free(x,xp,beta):
    """Uso: devuelve elemento de matriz dsnsidad para el caso de una partícula libre en un toro infinito."""
    return (2.*np.pi*beta)**(-0.5) * np.exp(-(x-xp)**2 / (2 * beta) ) 

def harmonic_potential(x):
    """Devuelve valor del potencial armónico para una posición x dada"""
    return 0.5* x**2

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

def path_naive_sampling( N_path = 10,beta = 4., N_iter = int(1e5), delta = 0.5, 
                         potential = harmonic_potential, append_every = 1 ):
    """
    Uso: 
    """
    dtau = beta/N_path
    path_x = [0.] * N_path
    pathss_x = [path_x[:]]
    t_0 = time()
    N_iter = int(N_iter)
    for step in range(N_iter):
        k = np.random.randint(0,N_path)
        #Periodic boundary conditions
        knext, kprev = (k+1) % N_path, (k-1) % N_path
        x_new = path_x[k] + np.random.uniform(-delta,delta)
        old_weight = (  rho_free(path_x[kprev],path_x[k],dtau) * 
                        np.exp(- dtau * potential(path_x[k]))  *
                        rho_free(path_x[k],path_x[knext],dtau)  )
        new_weight = (  rho_free(path_x[kprev],x_new,dtau) * 
                        np.exp(- dtau * potential(x_new))  *
                        rho_free(x_new,path_x[knext],dtau)  )
        if np.random.uniform(0,1) < new_weight/old_weight:
            path_x[k] = x_new
        if step%append_every == 0:
            pathss_x.append(path_x[:])
    t_1 = time()
    print('Path integral naive sampling:   %d iterations -> %.2E seconds'%(N_iter,t_1-t_0))
    pathss_x = np.array(pathss_x)
    return pathss_x

def figures_fn( pathss_x, beta = 4 , N_plot = 201, x_max = 3, N_iter=int(1e5), append_every=1,
                N_beta_ticks = 11, msq_file='file.csv', file_name='path-plot-prueba', 
                show_theory=True, show_matrix_squaring=True, show_path=True, 
                show_compare_hist=True, show_complete_path_hist=True,
                save_plot=True, show_plot=True):
    
    pathss_x = np.array(pathss_x)
    script_dir=os.path.dirname(os.path.abspath(__file__))
    x_plot = np.linspace(-x_max,x_max,N_plot)
    N_path = len(pathss_x[-1])

    # Agranda letra en texto en figuras generadas
    plt.rc('text', usetex=True) #usa latex en texto de figuras
    plt.rcParams.update({'font.size':15,'text.latex.unicode':True})

    # Crea figura
    fig, ax1 = plt.subplots()

    # Grafica histograma, teórico y si se pide un camino aleatorio
    ax1.set_xlabel(u'$x$')
    ax1.set_ylabel(u'$\pi^{(Q)} (x;\\beta)$')
    if show_theory:
        lns1 = ax1.plot(x_plot,QHO_canonical_ensemble(x_plot,beta),label=u'Teórico')
    if show_matrix_squaring:
        msq_file = script_dir + '/' + msq_file
        matrix_squaring_data = pd.read_csv(msq_file, index_col=0, comment='#')
        lns2 = ax1.plot(    matrix_squaring_data['position_x'],matrix_squaring_data['prob_density'],
                            label = u'Algoritmo Matrix\nSquaring') 
    lns3 = ax1.hist(pathss_x[:,0], bins=int(np.sqrt(N_iter/append_every)), normed=True,
                            label=u'Integral de camino\nnaive sampling',alpha=.40)
    if show_compare_hist:
        lns5 = ax1.hist(pathss_x[:,np.random.choice(np.arange(1,N_path))], bins=int(np.sqrt(N_iter/append_every)), normed=True,
                                label=u'Comparación hist. $x[k]$',alpha=.40)
    if show_complete_path_hist:
        pathss_x2 = pathss_x.copy()
        pathss_x2 = pathss_x2.flatten()
        lns6 = ax1.hist(pathss_x2, bins=int(np.sqrt(N_iter*N_path/append_every)), normed=True,
                                label=u'Comparación tomando\npath completo $\{x[k]\}_k$',alpha=.40)
    ax1.tick_params(axis='y')
    ax1.set_ylim(bottom=0)
    ax1.set_xlim(-x_max,x_max)
    if not show_path:
        plt.legend(loc = 'best', fontsize=12)
        if save_plot:
            plt.savefig(script_dir+'/'+file_name+'.eps')
        if show_plot:
            plt.show()
        plt.close()

    if show_path:
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        ax2.set_ylabel(u'$\\tau$')  # we already handled the x-label with ax1
        beta_plot = np.linspace(0,beta,N_path+1)
        path_plot = list(pathss_x[-1])
        path_plot.append(pathss_x[-1][0])
        lns4 = ax2.plot(path_plot, beta_plot,'o-',c='k',label=u'Path')
        ax2.tick_params(axis='y')
        beta_ticks = np.linspace(0,beta,N_beta_ticks)
        ax2.set_yticks(beta_ticks)
        ax2.set_yticklabels(u'$%.2f$'%b for b in beta_ticks)
        ax2.set_ylim(bottom=0)
        ax2.set_xlim(-x_max,x_max)
        # Parafernalia para agrgar labels correctamente cuando se usan dos ejes con
        # escalas diferentes
        if not show_theory:
            lns1 = [0]
        if not show_compare_hist:
            lns5 = [0]
        if not show_complete_path_hist:
            lns6 = [0]
        leg_test = lns1  + lns2 + lns4 + [lns3[2][0]]  + [lns5[2][0]] + [lns6[2][0]]
        labs = []
        leg = []
        for i,l in enumerate(leg_test):
            try:
                labs.append(l.get_label())
                leg.append(leg_test[i])
            except:
                pass
        ax1.legend(leg, labs, loc='best',title=u'$\\beta=%.2f$'%beta, fontsize=12)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        if save_plot:
            plt.savefig(script_dir+'/'+file_name+'.eps')
        if show_plot:
            plt.show()
        plt.close()
    return 0

def save_paths_x_csv(paths, file_name, relevant_info, print_data=True):
    """
    Uso: guarda datos de la distribución de probabilidad pi(x;beta) en un archivo .csv

    Recibe:
        grid_x: numpy array, shape=(nx,)    ->  valores de x en los que está evaluada pi(x;beta).
        x_weights: numpy array, shape=(nx,) ->  valores de pi(x;beta) para cada x en grid_x
        file_name: str                      ->  nombre del archivo en el que se guardarán datos.
        relevant_info: list of str          ->  información que se agrega como comentario en 
                                                primeras líneas. Cada elemento de esta lista 
                                                se agrega como una nueva línea.
        print_data: bool                    ->  decide si imprime datos guardados, en pantalla.
    
    Devuelve:
        pi_x_data: pd.DataFrame             ->  valores de pi(x;beta) para x en grid_x en formato
                                                "pandas".
    """
    # Almacena datos de probabilifad en diccionario: grid_x para posiciones y x_weights para
    # valores de densidad de probabilidad. 
    pi_x_data = {'position_x': grid_x,
                'prob_density': x_weights}
    # Pasamos datos a formato DataFrame de pandas.
    pi_x_data = pd.DataFrame(data=pi_x_data)
    # Crea archivo .csv y agrega comentarios relevantes dados como input
    with open(file_name,mode='w') as rho_csv:
        for info in list(relevant_info):
            rho_csv.write('# '+info+'\n')
    rho_csv.close()
    # Usamos pandas para escribir en archivo en formato csv.
    with open(file_name,mode='a') as rho_csv:
        pi_x_data.to_csv(rho_csv)
    rho_csv.close()
    # Imprime en pantalla datos de posiciones y probabilidades. 
    if print_data==True:
        print(pi_x_data)
    return pi_x_data

N_path = 9
beta = 4.
N_iter = int(1e6)
delta = 0.5
potential, potential_string = harmonic_potential, 'harmonic_potential'
append_every = 1
msq_file = 'pi_x-ms-harmonic_potential-x_max_5.000-nx_201-N_iter_7-beta_fin_4.000.csv'
N_plot = 201
x_max = 3
x_plot = np.linspace(-x_max,x_max,N_plot)
show_theory=True
show_matrix_squaring=True
show_path=True
show_compare_hist=True
show_complete_path_hist=True
save_plot = True
show_plot = True
relevant_info = [   'Paths generated by path_integral_naive_sampling.py',
                    'pi_x-pi-plot-%s-x_max_%.3f-N_path_%d-N_iter_%d-beta_fin_%.3f'\
                    %(potential_string,x_max,N_path,N_iter,beta)]

plot_file_name =    'pi_x-pi-plot-%s-x_max_%.3f-N_path_%d-N_iter_%d-beta_fin_%.3f-theory_%d-ms_%d-path_%d-compare_%d-complete_path-%d'\
                    %(potential_string,x_max,N_path,N_iter,beta,show_theory,show_matrix_squaring,show_path,show_compare_hist,show_complete_path_hist)

pathss_x = path_naive_sampling( N_path = N_path, beta = beta, N_iter = N_iter, delta = 0.5, 
                                potential = harmonic_potential, append_every = 1 )

csv_file_name =     'pi_x-pi-plot-%s-x_max_%.3f-N_path_%d-N_iter_%d-beta_fin_%.3f'\
                    %(potential_string,x_max,N_path,N_iter,beta)

figures_fn( pathss_x, beta = beta , N_plot = N_plot, x_max = x_max, N_iter=N_iter, 
            append_every=1, N_beta_ticks = 10, msq_file=msq_file,
            file_name=plot_file_name, show_theory=show_theory, 
            show_matrix_squaring=show_matrix_squaring, show_path=show_path, 
            show_compare_hist=show_compare_hist, 
            show_complete_path_hist=show_complete_path_hist,
            save_plot=save_plot, show_plot=show_plot)