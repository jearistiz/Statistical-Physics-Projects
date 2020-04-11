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
    n = np.len(psi[0.0])
    for x in psi.keys():
        psi[x].append((2./n)**0.5 * x * psi[x][n-1] - 
                            ((n-1)/n)**0.5 * psi[x][n-2])
    return psi

def add_x_value(psi,x):  #adds new x value to psi
    psi[x] = [np.exp(-x**2/2.) * np.pi**(-0.25)]
    psi[x].append(2**0.5 * x * psi[x][0])
    n_max = np.len(psi[0.0])-1
    for n in range(2,n_max+1):
                psi[x].append((2./n)**0.5 * x * psi[x][n-1] - 
                                    ((n-1)/n)**0.5 * psi[x][n-2])
    return psi

def canonical_ensemble_prob(E0,E1,beta):     #canonical ensemble probability factor in metropolis algorithm
    return np.exp(-beta * (E1-E0))


def metropolis(x0=0.0, delta_x=0.5, delta_steps=20, N=int(1e3), prob_sampling=[psi,canonical_ensemble_prob], beta=0.2):
    x_hist = [x0]
    n_hist = [0]
    for k in range(N):
        ##### Spatial monte carlo P(x -> x') #####
        xnew = x_hist[-1] + np.random.choice(range(-delta_steps,delta_steps+1))*delta_x####################### +  AAAAADDDDDD CODE HERE
        try:
            prob_sampling[0][xnew][0]
        except:
            prob_sampling = add_x_value(prob_sampling[0],xnew)
        acceptance_prob_1 = (prob_sampling[0][xnew][n_hist[-1]]/prob_sampling[0][x_hist[-1]][n_hist[-1]])**2
        if np.random.uniform() < min(1,acceptance_prob_1):
            x_hist.append(xnew)
        else:
            x_hist.append(x_hist[-1])
        
        ##### Energy level monte carlo P(n -> n') #####
        n_new = n_hist[-1] + np.random.choice([1,-1])   #propose new n

        if n_new < 0: #check if n is negative
            n_hist.append[n_hist[-1]]

        else:
            current_n_max = len(prob_sampling[0][0])-1    #current maximum energy level in wavefunction
            if n_new > current_n_max:  #check if proposed new n is greater than maximum energy level in wavefunction, if true, add new energy level. 
                prob_sampling[0] = add_energy_level(prob_sampling[0])

            acceptance_prob_2 = ( prob_sampling[0][x_hist[-1]][n_new] / prob_sampling[0][x_hist[-1]][n_hist[-1]] )**2 * \
                                prob_sampling[1]( n_new + 1./2. , n_hist[-1] + 1./2. ,  beta)
            if np.random.uniform() < min(1,acceptance_prob_2):
                n_hist.append(xnew)
            else:
                n_hist.append(n[-1])

    return x_hist, n_hist

n_energy_levels = 5
x_limit = 5
N_points_x = 50 
delta_x = x_limit/(N_points_x-1)

psi, grid_x = psi_0_1(x_limit,N_points_x)   # Creates first two energy levels and stores them into dictionary psi[x][n]

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