# -*- coding: utf-8 -*-

from __future__ import division
import os 
from time import time

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from matrix_squaring import (rho_free, harmonic_potential, anharmonic_potential, QHO_canonical_ensemble, 
                             save_csv)



# Author: Juan Esteban Aristizabal-Zuluaga
# date: 202004151200



def path_naive_sampling( N_path=10, beta=4., N_iter=int(1e5), delta=0.5, 
                         potential=harmonic_potential, append_every=1, save_paths_data=True,
                         paths_file_name=None, paths_relevant_info=None):
    """
    Uso: 
    """
    
    dtau = beta/N_path
    path_x = [0.] * N_path
    pathss_x = [path_x[:]]
    N_iter = int(N_iter)
    
    t_0 = time()
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

    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if save_paths_data:
        if paths_file_name is None:
            csv_file_name =     script_dir + '/' + \
                    'pi_x-pi-%s-beta_%.3f-N_path_%d-N_iter_%d-delta_%.3f-append_every_%d.csv'\
                    %(potential_string,beta,N_path,N_iter,delta,append_every)
    
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
            plt.savefig(script_dir+'/'+file_name)
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
        if not show_matrix_squaring:
            lns2 = [0]
        if not show_compare_hist:
            lns5 = [0]
        if not show_complete_path_hist:
            lns6 = [0]
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

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        if save_plot:
            plt.savefig(script_dir+'/'+file_name)
        if show_plot:
            plt.show()
        plt.close()
    return 0
