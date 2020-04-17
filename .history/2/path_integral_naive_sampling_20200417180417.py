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

beta = 4.
N = 10
dtau = beta/N
n_steps = int(1e5)
delta = 0.5
path_x = [0.] * N
pathss_x = [path_x[:]]
potential = harmonic_potential
append_every = 1
t_0 = time()
for step in range(n_steps):
    k = np.random.randint(0,N)
    #Periodic boundary conditions
    knext, kprev = (k+1) % N, (k-1) % N
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
pathss_x = np.array(pathss_x)
print('%d iterations -> %.2E seconds'%(n_steps,t_1-t_0))

N_plot = 201
x_max = 3
x_plot = np.linspace(-x_max,x_max,N_plot)


script_dir = os.path.dirname(os.path.abspath(__file__)) #path completa para este script
msq_file=script_dir+'/'+'pi_x-harmonic_potential-x_max_5.000-nx_201-N_iter_7-beta_fin_4.000.csv'
matrix_squaring_data = pd.read_csv(msq_file, index_col=0, comment='#')
print(matrix_squaring_data)
# Agranda letra en texto de figuras generadas
plt.rc('text', usetex=True)
plt.rcParams.update({'font.size':15,'text.latex.unicode':True})

fig, ax1 = plt.subplots()

ax1.set_xlabel(u'$x$')
ax1.set_ylabel(u'$\pi^{(Q)} (x;\\beta)$')
lns1 = ax1.plot(x_plot,QHO_canonical_ensemble(x_plot,beta),label=u'Teórico')
lns2 = ax1.hist(pathss_x[:,0], bins=int(np.sqrt(n_steps/append_every)), normed=True,
                        label=u'Integral de camino\nnaive sampling')
lns3 = ax1.plot(    matrix_squaring_data['position_x'],matrix_squaring_data['prob_density'],
                    label = u'Algoritmo Matrix Squaring')                   
ax1.tick_params(axis='y')
ax1.set_ylim(bottom=0)
plt.legend(loc='best')

np.random.triangular

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

ax2.set_ylabel(u'$\\tau$')  # we already handled the x-label with ax1
beta_plot = np.linspace(0,beta,N+1)
path_plot = list(pathss_x[-1])
path_plot.append(pathss_x[-1][0])
lns4 = ax2.plot(path_plot, beta_plot,'o-',c='k',label=u'Path')
ax2.tick_params(axis='y')
N_ticks = N + 1
beta_ticks = np.linspace(0,beta,N_ticks)
ax2.set_yticks(beta_ticks)
ax2.set_yticklabels(u'$%.2f$'%b for b in beta_ticks)
ax2.set_ylim(bottom=0)

# Solution for having two legends
leg = lns1  + [lns2[2][0]] + lns3 + lns4
labs = [l.get_label() for l in leg]
ax1.legend(leg, labs, loc='best',title=u'$\\beta=%.2f$'%beta, fontsize=12)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(script_dir+'/path-plot-prueba.eps')



plt.show()