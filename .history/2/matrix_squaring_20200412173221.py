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

def rho_armonic_trotter(grid,beta):
    """
    """
    return np.array([ [rho_free(x,xp,beta)*np.exp(-0.5*beta*(x**2 + xp**2)) for x in grid] for xp in grid])