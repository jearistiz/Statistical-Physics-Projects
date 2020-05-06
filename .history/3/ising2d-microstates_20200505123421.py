# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from time import time



def ising_microstates(L=2):
    # Tamaño del sistema
    N = L * L
    
    # Lista en la que se guardan explícitamente todos los microestados (2**N en total)
    config = np.array([[0] * N] * 2**N)
    
    # La primera mitad de los microestados
    for i in range(N):
        index_factor = int(2**N  / 2**(i+1))
        for j in range(2**i):
            config[j*index_factor : (j+1)*index_factor, i] = (-1)**j
    
    # La segunda mitad de los microestados son los estados opuestos a la primera mitad
    config[int((2 ** N) / 2):,:] = - np.copy(config[:int((2 ** N) / 2), :])

    return config

def neighbours(L=2):
    

print(ising_microstates())