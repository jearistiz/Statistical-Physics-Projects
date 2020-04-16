# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from time import time

def QHO_ground(x):
    """
    Uso: devuelve amplitud de probabilidad del estado base del Oscilador Armónico cuántico
    """
    return np.pi**(-0.25)*np.exp(-x**2/2.)

def metropolis(N=int(1e6),x0=0.0,delta=0.5,prob_amplitude_sampling=QHO_ground):
    """
    Uso: devuelve x_hist lista con N valores de x muestreados de la densidad de probabilidad
    (definida por la amplitud de probabilidad prob_amplitude_sampling) por el algoritmo
    Metrópolis.

    N: int                          ->  número de iteraciones para el algoritmo Metrópolis. 
    x0: float                       ->  valor de x con el que el algoritmo inicia el muestreo.
    delta: float                    ->  tamaño máximo del paso en cada iteración de "camino
                                        aleatorio" 
                                        usado por la cadena de Markov.
    prob_amplitude_sampling: func   ->  función de densidad de probabilidad a muestrear
    """
    # Iniciamos lista que almacena valores de posiciones escogidos por el algoritmo
    x_hist = [x0]
    N = int(N)
    for k in range(N):
        # Proponemos nueva posición para x con distribución uniforme centrada en valor anterior
        xnew = x_hist[-1] + np.random.uniform(-delta,delta)
        # Calculamos probabilidad de aceptancia del algoritmo Metrópolis
        acceptance_prob = min(1,(np.abs(prob_amplitude_sampling(xnew)/prob_amplitude_sampling(x_hist[-1])))**2)
        # Escogemos si aceptamos o no el valor de x propuesto
        if np.random.uniform() < acceptance_prob:
            x_hist.append(xnew)
        else:
            x_hist.append(x_hist[-1])
    return x_hist

def run_metropolis(N=1e5, x0=0.0, delta_x=0.5, prob_amplitude_sampling=QHO_ground, 
                    plot=True, showplot=True, savefig=[True,'plot_QHO_ground_state.eps'],
                    xlim = 3.5, N_plot = 201):
    """
    Uso: corre el algoritmo Metrópolis que muestrea valores de x de la densidad de  
    probabilidad definida por la amplitud de probabilidad prob_amplitude_sampling y
    grafica el histograma que resulta del algoritmo metrópolis, contrastado con la 
    densidad de probabilidad teórica.
    
    Recibe:
        N: int                  -> Número de iteraciones para el algoritmo Metrópolis
        x0: float               -> valor de x con el que el algoritmo inicia el muestreo.
        delta: float            -> tamaño máximo del paso en cada iteración de "camino 
                                   aleatorio" 
        prob_amplitude_sampling ->  Función de densidad de probabilidad a muestrear por el 
                                    algoritmo.
        showplot = True / False -> Elige si muestra o no la gráfica.
        savefig = [True / False, 'name of fig'] ->  Elige si guarda o no la gráfica. 
                                                    Nombre del archivo 'name of fig'
        x_lim: float            ->  límite en x para la gráfica
        N_plot: list            ->  número de valores de x para los que se grafica densidad 
                                    de probabilidad
    
    Devuelve:
        x_hist: list            ->  Lista con valores de x (posiciones)  obtenidos mediante 
                                    cadena de Markov.
        grafica histograma y comparación con teoría si plot=True 
    """
    N = int(N)
    # Corre el algoritmo metrópolis y mide tiempo de cómputo
    t_0 = time()
    x_hist = metropolis(N, x0, delta_x, prob_amplitude_sampling)
    t_1 = time()
    print('Metropolis algorithm QHO ground state: %.3f seconds for %.0E iterations'%(t_1-t_0,N))
    # Gráfica del histograma y comparación con densidad de probabilidad original
    if plot==True:
        x_plot = np.linspace(-xlim,xlim,N_plot)
        plt.figure(figsize=(8,5))
        plt.plot(x_plot,prob_amplitude_sampling(x_plot)**2,
                    label=u'QHO densidad de probabilidad\ndel estado base: $|\psi_0(x)|^2$')
        plt.hist(x_hist,bins=int(N**0.5),normed=True,
                    label=u'Histograma Metrópolis\ncon %.0E iteraciones'%(N))
        plt.xlim(-xlim,xlim)
        plt.xlabel(u'$x$')
        plt.ylabel(u'$|\psi_0(x)|^2$')
        plt.legend(loc='lower right')
        if savefig[0]==True:
            plt.savefig(savefig[1])
        if showplot==True:
            plt.show()
        plt.close()

    return x_hist

# Corremos el código usando función run_metropolis(), ésta graficará y guardará el histograma
plt.rcParams.update({'font.size':12})
x_hist = run_metropolis(N=1e6)