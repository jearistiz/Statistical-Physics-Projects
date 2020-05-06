# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from time import time



def ising_microstates(L=2):
    N = L * L
    config = np.array([[0] * N] * 2**N)
    print(config.shape)
    for i in range(N):
        index_factor = int(2**N  / 2**(i+1))
        for j in range(2**i):
            config[j*index_factor : (j+1)*index_factor, i] = (-1)**j
            config[(j+1)*index_factor : (j+2)*index_factor, i] = (-1)**(j+1)
    config[]
    return config

print(ising_microstates())