# -*- coding: utf-8 -*-

from __future__ import division
import os 
from time import time

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from matrix_squaring import (rho_free, harmonic_potential, anharmonic_potential,
                             QHO_canonical_ensemble, save_csv)



# Author: Juan Esteban Aristizabal-Zuluaga
# dated: 202004151200



def path_naive_sampling(beta=4., N_path=10, N_iter=int(1e5), delta=0.5, 
                        potential=harmonic_potential, potential_string='harmonic_potential',
                        append_every=1, save_paths_data=True, paths_file_name=None,
                        paths_relevant_info=None, *args, **kwargs):
    """Path integral naive sampling

    Uso:    genera camninos aleatorios con método montecarlo para calcular pi(x;beta) para un
            potencial (potential) dado.

    Recibe:
        N_path: int                     ->  determina el tamaño del camino aleatorio.
        beta: float                     ->  inverso de temperatura en unidades reducidas = 1/T.
        N_iter: int                     ->  iteraciones del algoritmo.
        delta: float                    ->  máximo tamaño en proposición del paso para el camino
                                            aleatorio.
        potential: function             ->  potencial al cual está sometido el sistema.
        potential_string: str           ->  nombre del potencial.
        append every: int               ->  cada append_every iteraciones se guarda el estado
                                            del camino aleatorio.
        save_paths_data: bool           ->  decide si guarda paths que resultan del algoritmo en
                                            archivo CSV.
        paths_file_name: str            ->  nombre del archivo en el que se guardarán los datos.
        paths_relevant_info: list       ->  información relevante para añadir al archivo como
                                            comentario en primeras líneas del archivo. Cada
                                            elemento de la lista debe ser un str, el cual. Cada
                                            elemento de la lista se comentará como una línea
                                            diferente.

        Devuelve:
            pathss_x: list  -> contiene los paths generados cada int(append_every) iteraciones.
    """
    # Parámetros relevantes.
    dtau = beta/N_path
    # Path a ser modificado por cada iteración se inicializa en ceros.
    path_x = [0.] * N_path
    # Lista que guardará todos los paths, comenzando por el path inicial. 
    pathss_x = [path_x[:]]
    N_iter = int(N_iter)
    
    t_0 = time()
    for step in range(N_iter):

        k = np.random.randint(0,N_path)
        
        #Periodic boundary conditions
        knext, kprev = (k+1) % N_path, (k-1) % N_path
        
        x_new = path_x[k] + np.random.uniform(-delta,delta)

        old_weight = (rho_free(path_x[kprev],path_x[k],dtau)
                      * np.exp(- dtau * potential(path_x[k]))
                      * rho_free(path_x[k],path_x[knext],dtau))
        
        new_weight = (rho_free(path_x[kprev],x_new,dtau)
                      * np.exp(- dtau * potential(x_new))
                      * rho_free(x_new,path_x[knext],dtau))
        
        # Decide si acepta proposición x_new para x[k]
        if np.random.uniform(0,1) < new_weight/old_weight:
            path_x[k] = x_new
        
        # Se añade el path generado cada int(append_every) iteraciones
        if step%append_every == 0:
            pathss_x.append(path_x[:])
    t_1 = time()
    
    print('Path integral naive sampling:   %d iterations -> %.2E seconds'%(N_iter,t_1-t_0))
    
    pathss_x = np.array(pathss_x)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Guarda datos en archivo CSV. 
    if save_paths_data:

        if not paths_file_name:
            paths_file_name = ( 
                    'pi_x-pi-%s-beta_%.3f-N_path_%d-N_iter_%d-delta_%.3f-append_every_%d.csv'
                    %(potential_string, beta, N_path, N_iter, delta, append_every)
                    )
        
        paths_file_name = script_dir + '/' + paths_file_name

        if not paths_relevant_info:
            paths_relevant_info = [
                'All paths generated by path_integral_naive_sampling.py each row is a path',
                '%s    beta = %.3f    N_path = %d    N_iter = %d    delta = %.3f'
                %(potential_string, beta, N_path, N_iter, delta)
                ]
        
        #Tamaño en memoria es menor y no afecta los cálculos en casos armónico y anarmónico
        pathss_x = np.round(pathss_x,5)   

        save_csv(pathss_x, None, None, paths_file_name, paths_relevant_info)
    
    return pathss_x


def figures_fn(pathss_x=None, read_paths_data=False, paths_file_name=None,
               beta=4., N_path=10, N_iter=int(1e5), delta=0.5, 
               potential_string='harmonic_potential', append_every=1, 
               N_plot=201, x_max=3, N_beta_ticks=11, msq_file=None,
               plot_file_name=None, show_QHO_theory=True, show_matrix_squaring=True,
               show_path=True, show_compare_hist=True, show_complete_path_hist=True,
               save_plot=True, show_plot=True, *args, **kwargs):
    """
    Uso: 
        Genera figuras del algoritmo path integral naive sampling.

    Recibe:
        pathss_x: list                  ->  contiene paths generados por path_naive_sampling.
        read_paths_data: bool           ->  decide si lee datos. True ignora pathss_x.
        paths_file_name: str            ->  nombre del archivo en que estan guardados datos
                                            de paths. Solo necesario si read_paths_data = True.
                                            Si read_paths_data = True y paths_file_name = None
                                            se intenta abrir archivo con nombre igual al
                                            generado automáticamente por path_naive_sampling y
                                            los parámetros especificados en figures_fn. 
        **kwargs                        ->  argumentos de función path_naive_sampling.
        N_plot: int                     ->  número de puntos para graficar función teórica.
        x_max: int                      ->  límite de eje x en gráfica es (-|x_max|,|x_max|)
        N_beta_ticks: int               ->  Número de valores de beta mostrados al graficar un
                                            camino aleatorio. Recomendable  valor < 12.
        msq_file: str                   ->  archivo en que están guardados datos de algoritmo
                                            matrix squaring para comparar con path integral.
                                            Necesario solo si show_matrix_squaring=True.
        plot_file_name: str             ->  nombre de la figura que se genera. Necesario solo si
                                            show_plot=True.
        show_QHO_theory: bool           ->  decide si muestra valor teórico del oscilador
                                            armónico.
        show_matrix_squaring: bool      ->  decide si muestra resultado de algoritmo Matrix
                                            Squaring. Si True es necesario especificar msq_file.
        show_path: bool                 ->  decide si muestra un camino aleatorio generado.
        show_compare_hist: bool         ->  decide si muestra comparación de histograma x[0]
                                            con histograma x[k], con k aleatorio entre [1,N-1].
        show_complete_path_hist: bool   ->  decide si muestra comparación de histograma x[0]
                                            con histograma de todo el camino {x[k]}.
        save_plot: bool                 ->  decide si guarda figura.
        show_plot: bool                 ->  decide si muestra figura en pantalla. 
    """

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Decide si lee datos e ignora input pathss_x
    if read_paths_data:
        if not paths_file_name:
            paths_file_name = ( 
                    'pi_x-pi-%s-beta_%.3f-N_path_%d-N_iter_%d-delta_%.3f-append_every_%d.csv'
                    %(potential_string, beta, N_path, N_iter, delta, append_every)
                    )
        paths_file_name = script_dir + '/' + paths_file_name
        pathss_x = pd.read_csv(paths_file_name, index_col=0, comment='#')
        pathss_x = pathss_x.to_numpy()

    pathss_x = np.array(pathss_x)
    N_path = len(pathss_x[-1])
    x_max = np.abs(x_max)

    # Crea figura
    fig, ax1 = plt.subplots()

    # Grafica histograma, teórico y si se pide un camino aleatorio
    ax1.set_xlabel(u'$x$')
    ax1.set_ylabel(u'$\pi^{(Q)} (x;\\beta)$')

    # Muestra teórico del oscilador armónico cuántico.
    if show_QHO_theory:
        x_plot = np.linspace(-x_max,x_max,N_plot)
        lns1 = ax1.plot(x_plot, QHO_canonical_ensemble(x_plot,beta), 
                        label=u'Oscilador Armónico\n(Teórico)')
    
    # Muestra valores generados por algoritmo matrix squaring.
    # Hay que especificar nombre del archivo en que están guardados estos datos. 
    if show_matrix_squaring:

        msq_file = script_dir + '/' + msq_file
        matrix_squaring_data = pd.read_csv(msq_file, index_col=0, comment='#')
        
        lns2 = ax1.plot(matrix_squaring_data['position_x'],matrix_squaring_data['prob_density'],
                        label = u'Algoritmo Matrix\nSquaring')
    
    # Muestra histograma generado con los las frecuencias de los valores de x[0] de cada path.
    lns3 = ax1.hist(pathss_x[:,0], bins=int(np.sqrt(N_iter/append_every)), normed=True,
                    label=u'Integral de camino\nnaive sampling $x[0]$',alpha=.40)
    
    # Muestra histograma generado con los las frecuencias de los valores de x[k] de cada path
    # con k aleatorio entre 1 y N_path-1
    if show_compare_hist:
        lns5 = ax1.hist(pathss_x[:,np.random.choice(np.arange(1,N_path))], 
                        bins=int(np.sqrt(N_iter/append_every)), normed=True,
                        label=u'Comparación hist. $x[k]$',alpha=.40)
    
    # Muestra histograma generado con las frecuencias de todas las posiciones {x[k]} de todos 
    # los caminos aleatorios
    if show_complete_path_hist:
        pathss_x2 = pathss_x.copy()
        pathss_x2 = pathss_x2.flatten()
        lns6 = ax1.hist(pathss_x2, bins=int(np.sqrt(N_iter*N_path/append_every)), normed=True,
                        label=u'Comparación tomando\npath completo $\{x[k]\}_k$',alpha=.40)
    
    # Parámetros del aspecto de la figura
    ax1.tick_params(axis='y')
    ax1.set_ylim(bottom=0)
    ax1.set_xlim(-x_max,x_max)

    if save_plot:
        
        if not plot_file_name:
            plot_file_name = ('pi_x-pi-plot-%s-beta_%.3f-'%(potential_string, beta)
                              + 'N_path_%d-N_iter_%d-delta_%.3f-'%(N_path, N_iter, delta)
                              + 'append_every_%d-x_max_%.3f-'%(append_every,x_max)
                              + 'theory_%d-ms_%d-'%(show_QHO_theory,show_matrix_squaring)
                              + 'path_%d-compare_%d-'%(show_path,show_compare_hist)
                              + 'complete_path-%d.eps'%(show_complete_path_hist))
                    
        plot_file_name = script_dir + '/' + plot_file_name

    if not show_path:
        plt.legend(loc = 'best', fontsize=12)
        if save_plot:
            plt.savefig(plot_file_name)
        if show_plot:
            plt.show()
        plt.close()

    if show_path:
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        ax2.set_ylabel(u'$\\tau$', rotation=270)
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
        
        # Parafernalia para agregar labels correctamente cuando se usan dos ejes con
        # escalas diferentes
        if not show_QHO_theory: lns1 = [0]
        if not show_matrix_squaring: lns2 = [0]
        if not show_compare_hist: lns5 = [0]
        if not show_complete_path_hist: lns6 = [0]
        try:
            lns3_test = [lns3[2][0]]
        except:
            lns3_test = [0]
        try:
            lns6_test = [lns6[2][0]]
        except:
            lns6_test = [0]
        try:
            lns5_test = [lns5[2][0]]
        except:
            lns5_test = [0]
        leg_test = lns1  + lns2 + lns4 + lns3_test  + lns6_test + lns5_test
        labs = []
        leg = []
        for i,l in enumerate(leg_test):
            try:
                labs.append(l.get_label())
                leg.append(leg_test[i])
            except:
                pass
        ax1.legend(leg, labs, loc='best',title=u'$\\beta=%.2f$'%beta, fontsize=12)

        fig.tight_layout()

        if save_plot:
            plt.savefig(plot_file_name)
        if show_plot:
            plt.show()
        plt.close()
    
    return
