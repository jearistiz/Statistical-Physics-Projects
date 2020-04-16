import numpy as np
import matplotlib.pyplot as plt
from time import time

def canonical_ensemble_prob(E0,E1,beta):
    return np.exp(-beta * (E1-E0))

def energy_levels(n_energy_levels = 5, x_limit = 8, N_points_x = 201):
    grid_x = [i*x_limit/(N_points_x-1) for i in range(-int((N_points_x-1)/2),int((N_points_x-1)/2 + 1))]
    psi = {}
    for x in grid_x:
        psi[x] = [np.exp(-x**2/2.) * np.pi**(-0.25)]
        psi[x].append(2**0.5 * x * psi[x][0])
        for n in range(2,n_energy_levels):
            psi[x].append((2./n)**0.5 * x * psi[x][n-1] - 
                            ((n-1)/n)**0.5 * psi[x][n-2])
    return psi, grid_x

# def metropolis(grid_x, x0=0.0, delta=0.5, N=int(1e3), prob_sampling=[psi,canonical_ensemble_prob], beta=0.2):
#     x_hist = [x0]
#     n = [0]

#     for k in range(N):
#         xnew = np.random.choice(grid_x)
#         acceptance_prob_1 = (prob_sampling[0][xnew][n[-1]]/prob_sampling[0][x_hist[-1]][n[-1]])**2
#         if np.random.uniform() < acceptance_prob_1:
#             x_hist.append(xnew)
#         else:
#             x_hist.append(x_hist[-1])

#         n_new = n[-1] + np.random.choice(1,-1)
#         acceptance_prob_2 = ( prob_sampling[0][x_hist[-1]][n_new] / prob_sampling[0][x_hist[-1]][n[-1]] )**2 * \
#                             canonical_ensemble_prob( n_new + 1./2. , n[-1] + 1./2. ,  beta)
#         if np.random.uniform < acceptance_prob_2:
#             n.append(xnew)
#     return x_hist, n

n_energy_levels = 5
x_limit = 5
N_points_x = 50 

psi, grid_x = energy_levels()

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