import numpy as np
import matplotlib.pyplot as plt
from time import time

def psi_0_1(x_limit = 8, N_points_x = 201):  #creates first two energy eigenfunctions
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

def metropolis_finite_temp(x0=0.0, delta_x=0.5, N=int(1e3), prob_sampling=[psi_0_1,canonical_ensemble_prob], beta=0.2):
    x_hist = [ x0 ]       #histogram for positions
    n_hist = [ 0 ]        #histogram for energy levels
    prob_sampling = [prob_sampling[0].copy(),prob_sampling[1]]
    for k in range(N):
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

def C_QHO_canonical_ensemble(x,beta):
    return (beta/(2.*np.pi))**0.5 * np.exp(-x**2*beta / 2.)

def Q_QHO_canonical_ensemble(x,beta):
    return (np.tanh(beta/2.)/np.pi)**0.5 * np.exp(- x**2 * np.tanh(beta/2.))

x_limit = 5.
N_points_x = 51    # small odd number!!!

x0 = 0.0
delta_x = 0.5 
N_metropolis = int(1e6)
beta_array = [30] #[0.2,1,5,7,9]
legend_loc = ['best'] #['lower center','lower right','best','best','best']

for i, beta in enumerate(beta_array):
    psi, grid_x = psi_0_1(x_limit,N_points_x)   # Creates first two energy levels and stores them into dictionary psi[x][n]
    prob_sampling = [psi,canonical_ensemble_prob]

    ###### Call metropolis algorithm
    t_0 = time()
    x_hist, n_hist, psi_final = metropolis_finite_temp(x0=x0, delta_x=delta_x,N=N_metropolis, 
                                                        prob_sampling=prob_sampling, beta=beta)
    t_1 = time()
    print('Metropolis algorithm (beta = %.2f): %.3f seconds for %.0E iterations'%(beta,t_1-t_0,N_metropolis))


    x_limit_plot = 6
    x_plot = np.linspace(-x_limit_plot,x_limit_plot,200)

    plt.rcParams.update({'font.size':12})
    plt.figure(figsize=(8,5))
    plt.plot(x_plot,C_QHO_canonical_ensemble(x_plot,beta),
                label=u'Densidad de probabilidad\nteórica clásica')
    plt.plot(x_plot,Q_QHO_canonical_ensemble(x_plot,beta),
                label=u'Densidad de probabilidad\nteórica cuántica')
    plt.hist(x_hist,bins=int(N_metropolis**0.5),normed=True,
                label='Algoritmo Metrópolis\ncon %.0E iteraciones'%(N_metropolis))
    plt.xlim(-x_limit_plot,x_limit_plot)
    plt.xlabel(u'$x$')
    plt.ylabel(u'$\pi(x)$')
    plt.legend(loc=legend_loc[i], title=u'$\\beta=%.1f$'%beta)
    plt.savefig('QHO_finite_temp_beta_%d_%d.eps'%(beta,(beta-int(beta))*100))
    #plt.show()
    plt.close()

    # plt.plot()
    # plt.hist(n_hist,normed=True,bins=len(psi_final[0]))
    # plt.show()
    # plt.close()

###### Check if all functions are working properly
for i in range(5):
    psi = add_energy_level(psi)             #Check if add_energy_level() is working properly
    grid_x.append(grid_x[-1] + x_limit/(N_points_x-1))       
    psi = add_x_value(psi,grid_x[-1])       #Check if add_x_value() is working properly

plt.rcParams.update({'font.size':12})
plt.figure(figsize=(8,5))
n_max = len(psi[0])-1
if n_max <= 12:
    True
else:
    n_max = 5
for n in range(n_max+1):
    psi_x_n = np.array([psi[x][n] for x in grid_x])
    plt.plot(grid_x, psi_x_n,label = u'$n = %d$'%n)
plt.xlabel(u'$x$')
plt.ylabel(u'QHO autofunciones de energía: $\psi_n(x)$')
plt.legend()
plt.show()
plt.close()