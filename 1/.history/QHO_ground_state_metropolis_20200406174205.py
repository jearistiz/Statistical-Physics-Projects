import numpy as np
import matplotlib.pyplot as plt
from time import time

def QHO_ground(x):
    return np.pi**(-0.25)*np.exp(-x**2/2)

def metropolis(x0=0.0,delta=0.5,N=int(1e6),prob_amplitude_sampling=QHO_ground):
    x_hist = [x0]
    for k in range(N):
        xnew = x_hist[-1] + np.random.uniform(-delta,delta)
        acceptance_prob = (prob_amplitude_sampling(xnew)/prob_amplitude_sampling(x_hist[-1]))**2
        if np.random.uniform() < acceptance_prob:
            x_hist.append(xnew)
        else:
            x_hist.append(x_hist[-1])
    return x_hist

N = int(1e5)

t_0 = time()
x_hist = metropolis(N=N,prob_amplitude_sampling=QHO_ground)
t_1 = time()
print('Metropolis algorithm: %.3f seconds for %.0E iterations'%(t_1-t_0,N))

xlim = 5*2**(-0.5)
x_plot = np.linspace(-xlim,xlim,200)

plt.rcParams.update({'font.size':12})
plt.figure(figsize=(8,5))
plt.plot(x_plot,QHO_ground(x_plot)**2,
            label=u'QHO densidad de probabilidad\ndel estado base: $|\psi_0(x)|^2$')
plt.hist(x_hist,bins=int(N**0.5),normed=True,
            label='Histograma usando algoritmo\nMetrópolis con %.0E iteraciones'%(N))
plt.xlim(-xlim,xlim)
plt.xlabel(u'$x$')
plt.ylabel(u'$|\psi_0(x)|^2$')
plt.legend(loc='lower right')
#plt.savefig('QHO_ground_state.eps')
plt.show()



