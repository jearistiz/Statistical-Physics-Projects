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
                show_path=True, show_matrix_squaring=True, save_plot=True, show_plot=True):
    
    script_dir=os.path.dirname(os.path.abspath(__file__))
    x_plot = np.linspace(-x_max,x_max,N_plot)
    

    # Agranda letra en texto de figuras generadas
    plt.rc('text', usetex=True) #usa latex en texto de figuras
    plt.rcParams.update({'font.size':15,'text.latex.unicode':True})

    fig, ax1 = plt.subplots()

    ax1.set_xlabel(u'$x$')
    ax1.set_ylabel(u'$\pi^{(Q)} (x;\\beta)$')
    lns1 = ax1.plot(x_plot,QHO_canonical_ensemble(x_plot,beta),label=u'Teórico')
    if show_matrix_squaring:
        msq_file = script_dir + '/' + msq_file
        matrix_squaring_data = pd.read_csv(msq_file, index_col=0, comment='#')
        lns2 = ax1.plot(    matrix_squaring_data['position_x'],matrix_squaring_data['prob_density'],
                            label = u'Algoritmo Matrix\nSquaring') 
    lns3 = ax1.hist(pathss_x[:,0], bins=int(np.sqrt(N_iter/append_every)), normed=True,
                            label=u'Integral de camino\nnaive sampling')                 
    ax1.tick_params(axis='y')
    ax1.set_ylim(bottom=0)
    ax1.set_xlim(-x_max,x_max)
    if not show_path:
        plt.legend(loc = 'best')
        if save_plot:
            plt.savefig(script_dir+'/'+file_name+'.eps')
        if show_plot:
            plt.show()
        plt.close()

    if show_path:
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        ax2.set_ylabel(u'$\\tau$')  # we already handled the x-label with ax1
        N_path = len(pathss_x[-1])
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

        # Solution for having legends that share two different scales
        leg = lns1  + lns2 + [lns3[2][0]] + lns4
        labs = [l.get_label() for l in leg]
        ax1.legend(leg, labs, loc='best',title=u'$\\beta=%.2f$'%beta, fontsize=12)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        if save_plot:
            plt.savefig(script_dir+'/'+file_name+'-path_true.eps')
        if show_plot:
            plt.show()
        plt.close()
    return 0

N_path = 10
beta = 4.
N_iter = int(1e4)
delta = 0.5
potential, potential_string = harmonic_potential, 'harmonic_potential'
append_every = 1
path_naive_sampling(N_iter=N_iter)
pathss_x = path_naive_sampling( N_path = N_path, beta = beta, N_iter = int(1e5), delta = 0.5, 
                                potential = harmonic_potential, append_every = 1 )

#script_dir = os.path.dirname(os.path.abspath(__file__)) #path completa para este script
msq_file = 'pi_x-ms-harmonic_potential-x_max_5.000-nx_201-N_iter_7-beta_fin_4.000.csv'

N_plot = 201
x_max = 3
x_plot = np.linspace(-x_max,x_max,N_plot)
plot_file_name = 'pi_x-pi-plot-%s-x_max_%.3f-N_path_%d-N_iter_%d-beta_fin_%.3f'\
                                            %(potential_string,x_max,N_path,N_iter,beta)

figures_fn( pathss_x, beta = beta , N_plot = N_plot, x_max = x_max, N_iter=N_iter, 
            append_every=1, N_beta_ticks = N_path+1, msq_file=msq_file,
            file_name=plot_file_name, show_path=True, show_matrix_squaring=True,
             save_plot=True, show_plot=True)