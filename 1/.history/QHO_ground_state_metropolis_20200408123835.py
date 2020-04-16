# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from time import time

def QHO_ground(x):
    return np.pi**(-0.25)*np.exp(-x**2/2.)

def metropolis(x0=0.0,delta=0.5,N=int(1e6),prob_amplitude_sampling=QHO_ground):
    x_hist = [x0]
    for k in range(N):
        xnew = x_hist[-1] + np.random.uniform(-delta,delta)
        acceptance_prob = (np.abs(prob_amplitude_sampling(xnew)/prob_amplitude_sampling(x_hist[-1])))**2
        if np.random.uniform() < acceptance_prob:
            x_hist.append(xnew)
        else:
            x_hist.append(x_hist[-1])
    return x_hist

def run_metropolis(N=1e5, prob_amplitude_sampling=False, 
                    showplot=True, savefig=[True,'QHO_ground_state.eps'],
                    xlim = 5*2**(-0.5), x_plot = np.linspace(-xlim,xlim,200)):
    """
    Uso: corre el algoritmo Metrópolis que muestrea valores de x de la densidad de  
    probabilidad definida por la amplitud de probabilidad prob_amplitude_sampling y
    grafica el histograma que resulta del algoritmo metrópolis, contrastado con la 
    densidad de probabilidad teórica.
    
    N:int                       -> Número de iteraciones para el algoritmo Metrópolis
    prob_amplitude_sampling     -> Función de densidad de probabilidad a muestrear por el algoritmo.
    showplot = True / False                     -> Elige si muestra o no la gráfica.
    savefig = [True / False, 'name of fig']     -> Elige si guarda o no la gráfica. Nombre del archivo 'name of fig'

    """
    t_0 = time()
    x_hist = metropolis(N=N,prob_amplitude_sampling=QHO_ground)
    t_1 = time()
    print('Metropolis algorithm QHO ground state: %.3f seconds for %.0E iterations'%(t_1-t_0,N))
    
    plt.rcParams.update({'font.size':12})
    plt.figure(figsize=(8,5))
    plt.plot(x_plot,prob_amplitude_sampling(x_plot)**2,
                label=u'QHO densidad de probabilidad\ndel estado base: $|\psi_0(x)|^2$')
    plt.hist(x_hist,bins=int(N**0.5),normed=True,
                label=u'Histograma usando algoritmo\nMetrópolis con %.0E iteraciones'%(N))
    plt.xlim(-xlim,xlim)
    plt.xlabel(u'$x$')
    plt.ylabel(u'$|\psi_0(x)|^2$')
    plt.legend(loc='lower right')
    if savefig==True:
        plt.savefig(savefig[1])
    if showplot==True:
        plt.show()
    plt.close()

    return 0

N = int(1e6)
xlim = 5*2**(-0.5)
x_plot = np.linspace(-xlim,xlim,200)

run_metropolis(N=1e5, prob_amplitude_sampling=False,
                xlim = xlim, x_plot = x_plot, showplot=True, savefig=False)

# t_0 = time()
# x_hist = metropolis(N=N,prob_amplitude_sampling=QHO_ground)
# t_1 = time()
# print('Metropolis algorithm: %.3f seconds for %.0E iterations'%(t_1-t_0,N))

# plt.rcParams.update({'font.size':12})
# plt.figure(figsize=(8,5))
# plt.plot(x_plot,QHO_ground(x_plot)**2,
#             label=u'QHO densidad de probabilidad\ndel estado base: $|\psi_0(x)|^2$')
# plt.hist(x_hist,bins=int(N**0.5),normed=True,
#             label=u'Histograma usando algoritmo\nMetrópolis con %.0E iteraciones'%(N))
# plt.xlim(-xlim,xlim)
# plt.xlabel(u'$x$')
# plt.ylabel(u'$|\psi_0(x)|^2$')
# plt.legend(loc='lower right')
# plt.savefig('QHO_ground_state.eps')
# plt.show()
# plt.close()