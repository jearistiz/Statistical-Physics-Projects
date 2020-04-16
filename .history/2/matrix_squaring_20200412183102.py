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

def

def rho_armonic_trotter(grid, beta, potential):
    """
    Uso:    devuelve matriz densidad en aproximación de Trotter para altas temperaturas y un potencial dado


    Recibe:
        grid: list      -> lista
        beta: float     -> inverso de temperatura en unidades reducidas
        potential: func -> potencial de interacción
    """
    return np.array([ [rho_free(x,xp,beta)*np.exp(-0.5*beta*0.5*(x**2 + xp**2)) for x in grid] for xp in grid])