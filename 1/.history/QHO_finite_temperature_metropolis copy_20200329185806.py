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

def add_energy_level(psi,grid_x):            #adds new energy eigenfunction to psi
    n = np.len(psi[0.0])
    for x in grid_x:
        psi[x].append((2./n)**0.5 * x * psi[x][n-1] - 
        ((n-1)/n)**0.5 * psi[x][n-2])
    return psi

def add_x_value(psi,x):
    ####################
    ####################
    #### ADD CODE HERE
    ####################
    ####################
    return 0
def canonical_ensemble_prob(E0,E1,beta):     #canonical ensemble probability factor in metropolis algorithm
    return np.exp(-beta * (E1-E0))

def metropolis(grid_x, x0=0.0, delta=0.5, N=int(1e3), prob_sampling=[psi,canonical_ensemble_prob], beta=0.2):
    x_hist = [x0]
    n = [0]
    for k in range(N):
        ##### Spatial monte carlo P(x -> x')
        xnew = x_hist[-1] ####################### +  AAAAADDDDDD CODE HERE 
        acceptance_prob_1 = (prob_sampling[0][xnew][n[-1]]/prob_sampling[0][x_hist[-1]][n[-1]])**2
        if np.random.uniform() < min(1,acceptance_prob_1):
            x_hist.append(xnew)
        else:
            x_hist.append(x_hist[-1])
        
        ##### Energy level monte carlo P(n -> n')
        n_new = n[-1] + np.random.choice(1,-1)   #propose new n
        if n_new < 0: #check if n is negative
            n.append[n[-1]]
        else:
            current_n_max = len(prob_sampling[0][0])-1    
            if n_new > current_n_max:
                prob_sampling[0] = add_energy_level(psi,grid_x)

            acceptance_prob_2 = ( prob_sampling[0][x_hist[-1]][n_new] / prob_sampling[0][x_hist[-1]][n[-1]] )**2 * \
                                canonical_ensemble_prob( n_new + 1./2. , n[-1] + 1./2. ,  beta)
            if np.random.uniform() < min(1,acceptance_prob_2):
                n.append(xnew)
            else:
                n.append(n[-1])

    return x_hist, n

n_energy_levels = 5
x_limit = 5
N_points_x = 50 

psi, grid_x = psi_0_1()   # Creates first two energy levels and stores them into dictionary psi[x][n]

plt.rcParams.update({'font.size':12})
plt.figure(figsize=(8,5))
for n in range(n_energy_levels):
    psi_x_n = np.array([psi[x][n] for x in grid_x])
    plt.plot(grid_x, psi_x_n,label = u'$n = %d$'%n)
    #print('Energy level %d: '%(n),np.round(psi_x_n,3))
plt.xlabel(u'$x$')
plt.ylabel(u'QHO autofunciones de energía: $\psi_n(x)$')
plt.legend()
plt.show()
plt.close()