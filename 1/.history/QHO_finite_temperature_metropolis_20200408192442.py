# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from time import time

def psi_0_1(x_limit = 5, N_points_x = 101):  #creates first two energy eigenfunctions
    N_points_x = int(N_points_x)
    if N_points_x%2 ==0:
        N_points_x = N_points_x + 1
    delta = x_limit/(N_points_x-1)
    grid_x = [i*delta for i in range(-int((N_points_x-1)/2),int((N_points_x-1)/2 + 1))]
    psi = {}
    for x in grid_x:
        psi[x] = [np.exp(-x**2/2.) * np.pi**(-0.25)]
        psi[x].append(2**0.5 * x * psi[x][0])
    return psi, grid_x

def add_energy_level(psi):            #adds new energy eigenfunction to psi
    n = len(psi[0.0])
    for x in psi.keys():
        psi[x].append((2./n)**0.5 * x * psi[x][n-1] - 
                            ((n-1)/n)**0.5 * psi[x][n-2])
    return psi

def add_x_value(psi,x):  #adds new x value to psi
    psi[x] = [np.exp(-x**2/2.) * np.pi**(-0.25)]
    psi[x].append(2**0.5 * x * psi[x][0])
    n_max = len(psi[0.0])-1
    for n in range(2,n_max+1):
                psi[x].append((2./n)**0.5 * x * psi[x][n-1] - 
                                    ((n-1)/n)**0.5 * psi[x][n-2])
    return psi

def canonical_ensemble_prob(delta_E,beta):     #canonical ensemble probability factor in metropolis algorithm
    return np.exp(-beta * delta_E)

def boltzmann_probability(En,beta):
    return 2.*np.sinh(beta/2.)*np.exp(-beta*En)

def metropolis_finite_temp(x0=0.0, delta_x=0.5, N=1e3, prob_sampling=[psi_0_1()[0],canonical_ensemble_prob], beta=5):
    x_hist = [ x0 ]       #histogram for positions
    n_hist = [ 0 ]        #histogram for energy levels
    prob_sampling = [prob_sampling[0].copy(),prob_sampling[1]]
    for k in range(int(N)):
        ##### Spatial monte carlo P(x -> x') #####
        x_new = x_hist[-1] + np.random.uniform(-delta_x,delta_x)
        try:   #check if position x_new is part of the wavefunction dictionary (psi)
            prob_sampling[0][x_new][0]
        except:
            prob_sampling[0] = add_x_value(prob_sampling[0],x_new)
        acceptance_prob_1 = ( prob_sampling[0][x_new][n_hist[-1]] / prob_sampling[0][x_hist[-1]][n_hist[-1]] )**2
        if np.random.uniform() < min(1,acceptance_prob_1):
            x_hist.append(x_new)
        else:
            x_hist.append(x_hist[-1])
        ##### Energy level monte carlo P(n -> n') #####
        n_new = n_hist[-1] + np.random.choice([1,-1])   #propose new n
        if n_new < 0: #check if n_new is negative
            n_hist.append(n_hist[-1])
        else:
            current_n_max = len(prob_sampling[0][0])-1    #current maximum energy level in wavefunction
            if n_new > current_n_max:  #check if n_new is greater than maximum energy level in wavefunction, if true, add new energy level. 
                prob_sampling[0] = add_energy_level(prob_sampling[0])
            ##### Monte Carlo section:
            acceptance_prob_2 = ( prob_sampling[0][x_hist[-1]][n_new] / prob_sampling[0][x_hist[-1]][n_hist[-1]] )**2 * \
                                prob_sampling[1]( n_new-n_hist[-1],  beta)
            if np.random.uniform() < min(1,acceptance_prob_2):
                n_hist.append(n_new)
            else:
                n_hist.append(n_hist[-1])
    return x_hist, n_hist, prob_sampling[0]

def CHO_canonical_ensemble(x,beta=5,plot=False,savefig=True,showplot=False):
    if plot==True:
        x = np.linspace(-3,3,201)
        plt.figure()
        for beta0 in list(beta):
            pdf = (beta0/(2.*np.pi))**0.5 * np.exp(-x**2*beta0 / 2.)
            plt.plot(x,pdf,label=u'$\\beta = %.1f$'%beta0)
        plt.xlim(-3,3)
        plt.xlabel('$x$')
        plt.ylabel('$\pi^{(C)}(x;\\beta)$')
        plt.legend(loc='best')
        if savefig==True:
            plt.savefig('CHO_finite_temp_several_beta.eps')
        if showplot==True:
            plt.show()
        plt.close()
        return 0 
    else:
        return (beta/(2.*np.pi))**0.5 * np.exp(-x**2*beta / 2.)

def QHO_canonical_ensemble(x,beta):
    return (np.tanh(beta/2.)/np.pi)**0.5 * np.exp(- x**2 * np.tanh(beta/2.))

def run_metropolis(psi_0_1 = psi_0_1, x_limit = 5., N_points_x = 51,
                    x0 = 0.0, delta_x = 0.5, N_metropolis = int(1e5), beta = 5., canonical_ensemble_prob = canonical_ensemble_prob,
                    plot=True, showplot = True, savefig = True, legend_loc = 'best', x_plot_0 = 7):
    """
    Uso:    Corre algoritmo Metrópolis para el oscilador armónico cuántico en un baño térmico. 
            Grafica el histograma de posiciones obtenido contrastándolo con los resultados 
            teóricos cuántico y clásico. Grafica histograma de niveles de energía visitados por
            el algoritmo.

    Recibe:
        psi_0_1: función    -> función que inicializa las autofunciones del hamiltoniano.
        x_limit: float      -> las autofunciones se inicializan en intervalo (-x_limit,x_limit).
        N_points_x: int     -> la rejilla para inicializar autofunciones tiene N_points_x puntos.
        x0: float           -> valor de x con el que el algoritmo inicia el muestreo.
        delta_x: float      -> tamaño máximo del paso en cada iteración de "camino aleatorio".
        N_metropolis: int   -> número de iteraciones para algoritmo metrópolis.
        beta: float         -> inverso de temperatura del baño térmico en unidades reducidas beta = 1/T.
        canonical_ensemble_prob: función   -> función que genera factor de Boltzmann exp(-B*deltaE).
        plot: bool                         -> escoge si grafica o no los histogramas
        showplot: bool                     -> escoge si muestra o no la gráfica
        savefig: [bool,'name of fig']      -> escoge si guarda o no la figura y el nombre del archivo.
        legend_loc: 'position'             -> posición de la legenda para la figura
        x_plot_0: float                    -> dominio de la gráfica en x será (-x_plot,x_plot)
    
    Devuelve:   
        x_hist: list        ->  Lista con valores de x (posiciones)  obtenidos mediante cadena de Markov. 
        n_hist: list        ->  Lista con valores de n (niveles de energía) obtenidos mediante cadena de Markov.
        psi_final: dict     ->  Diccionario con autofunciones de energía \psi_{n}(x) = psi[x][n] 
                                    para valores de x y n en x_hist y n_hist.
    """
    # Inicializamos autofunciones de energía en diccionario psi generado por función psi_0_1()
    psi, grid_x = psi_0_1(x_limit,N_points_x)
    
    # Almacenamos probabilidades en una lista:      la amplitud de probabilidad psi de las autofunciones
    #                                               y el factor de Boltzmann del ensamble canónico
    prob_sampling = [psi, canonical_ensemble_prob]

    # Ejecutamos algoritmo metropolis y medimos tiempo de cómputo
    t_0 = time()
    x_hist, n_hist, psi_final = metropolis_finite_temp(x0=x0, delta_x=delta_x,N=N_metropolis, 
                                                        prob_sampling=prob_sampling, beta=beta)
    t_1 = time()
    print('Metropolis algorithm (beta = %.2f): %.3f seconds for %.0E iterations'%(beta,t_1-t_0,N_metropolis))

    if plot==True:
        # Graficamos histograma para posiciones
        x_plot = np.linspace(-x_plot_0,x_plot_0,251)
        plt.rcParams.update({'font.size':12})
        plt.figure(figsize=(8,5))
        plt.plot(x_plot,CHO_canonical_ensemble(x_plot,beta),
                    label=u'$\pi^{(C)}(x;\\beta)$')
        plt.plot(x_plot,QHO_canonical_ensemble(x_plot,beta),
                    label=u'$\pi^{(Q)}(x;\\beta)$')
        plt.hist(x_hist,bins=int(N_metropolis**0.5),normed=True,
                    label='Histograma Metrópolis\ncon %.0E iteraciones'%(N_metropolis))
        plt.xlim(-x_plot_0,x_plot_0)
        plt.xlabel(u'$x$')
        plt.ylabel(u'$\pi(x;\\beta)$')
        plt.legend(loc=legend_loc, title=u'$\\beta=%.2f$'%beta)
        if savefig==True:
            plt.savefig('plot_QHO_finite_temp_beta_%d_%d.eps'%(beta,(beta-int(beta))*100))
        if showplot==True:
            plt.show()
        plt.close()

        # Graficamos histograma para niveles de energía
        n_plot = np.arange(len(psi_final[0])) 
        plt.hist(n_hist,normed=True,bins=np.arange(len(psi_final[0])+1)-0.5,label='Histograma Metrópolis\nniveles de energía')
        plt.plot(n_plot,boltzmann_probability(n_plot+1/2),label=u'$e^{-\\beta E_n}/Z(\\beta)$')
        plt.xlabel(u'$n$')
        plt.ylabel(u'$\pi(n;\\beta)$')
        plt.legend(loc='best', title=u'$\\beta=%.2f$'%beta)
        if savefig==True:
            plt.savefig('plot_QHO_n_hist_beta_%d_%d.eps'%(beta,(beta-int(beta))*100))
        if showplot==True:
            plt.show()
        plt.close()

    return x_hist, n_hist, psi_final


# Corremos algoritmo metrópolis usando función run_metropolis()
beta_array = [0.2, 1, 5, 60]
legend_loc =['lower center', 'lower right', 'best', 'best']
for i,beta in enumerate(beta_array):
    run_metropolis(N_metropolis=1e5,beta=beta,showplot=False)

# Corremos 
beta_array_CHO = [5,20,60,100]
CHO_canonical_ensemble(0,beta=beta_array_CHO,plot=True,showplot=True)