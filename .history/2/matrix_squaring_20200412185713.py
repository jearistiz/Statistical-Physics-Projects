# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from time import time

def rho_free(x,xp,beta):
    """
    Uso: devuelve elemento de matriz dsnsidad para el caso de una partícula libre en un toro infinito.
    """
    return (2.*np.pi*beta)**(-0.5) * np.exp(-(x-xp)**2 / (2 * beta) )

def harmonic_potential(x):
    return 0.5*x**2

def anharmonic_potential(x):
    return 0.5*x**2 - x**3 + x**4

def rho_trotter(grid, beta, potential=harmonic_potential):
    """
    Uso:    devuelve matriz densidad en aproximación de Trotter para altas temperaturas y un potencial dado

    Recibe:
        grid: list      -> lista de dimensión N
        beta: float     -> inverso de temperatura en unidades reducidas
        potential: func -> potencial de interacción

    Devuelve:
        matrix          -> matriz densidad de dimension NxN
    """
    return np.array([ [ rho_free(x , xp,beta) * np.exp(-0.5*beta*(potential(x)+potential(xp))) for x in grid] for xp in grid])

x_max = 5.
nx = 101
dx = 2. * x_max / (nx - 1)
grid_x = [i*dx for i in range(-int((nx-1)/2), int(nx/2 + 1))]
beta_ini = 2.**(-5)
beta_fin = 2.**2
rho = rho_trotter(grid_x,beta_ini)
while beta_ini < beta_fin:
    rho = np.dot(rho,rho)
    rho *= dx
    beta_ini *= 2.
    print('beta: %E -> %E'%(beta_ini/2,beta_ini))
