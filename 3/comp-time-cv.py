# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# Usar latex en texto de figuras y agrandar tamaño de fuente
plt.rc('text', usetex=True) 
plt.rcParams.update({'font.size':15,'text.latex.unicode':True})


def expo(t,a,m):
    return a*np.exp(m*t)

L = np.array([2,3,4,5])
T = np.array([0.0077,0.153,32,3400])/(3600*24)

popt, pcov = curve_fit(expo, L, T, p0=np.array([1e-12,2.5]))

L_plot = np.linspace(2, 8, 100)
T_plot = expo(L_plot, *popt)
t_8 = expo(8,*popt)
print(popt,'\nTiempo para calcular cv 8x8 = ',expo(8,*popt), ' días')


plt.figure()
plt.plot(L,T,'--*', label='Tiempo computacional')
plt.plot(L_plot, T_plot, label='Ajuste exponencial\n$t(L) = t_0 e^{m L}$\n$t_0=%.2e$  $m = %.2f$'%(popt[0],popt[1]))
plt.plot(8, t_8, '*', c='r', ms=10 , label='Predicción $t(8) \\approx 47000$ días')
plt.xlim(1.5, 8.5)
plt.xlabel('$L$')
plt.ylabel('Tiempo (días)')
plt.legend()
plt.yscale('log')
plt.tight_layout()
plt.savefig('computation_time.pdf')
plt.show()



