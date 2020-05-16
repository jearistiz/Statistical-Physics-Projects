# -*- coding: utf-8 -*-
import numpy as np
from ising2d_metropolis import ising_neighbours, ising_energy

def ising_metropolis_energies(microstate=np.ones(36,dtype=np.int64), 
                              read_ini_microstate_data=False, L=6, beta=1., J=1,
                              N_steps=10000, N_transient=100):

    N = L * L
    # Calcula vecinos
    ngbrs = ising_neighbours(L)

    # Si los datos se no se leyeron, genera microestado inicial aleatoriamente
    if read_ini_microstate_data:
        pass
    else: 
        microstate = np.random.choice(np.array([1,-1]), N)
    
    # Calcula energía inicial
    energy = ising_energy([microstate], ngbrs, J=J, print_log=False)[0]
    # Arreglo donde se guardarán energías de los microestados muestreados
    energies = []

    # En el transiente no se guardan las energías,
    # se espera a que el sistema se termalice.
    for i in range(N_transient):
        k = np.random.randint(N)
        delta_E = (2. * J * microstate[k]
                   * np.sum(np.array([microstate[ngbr_i] for ngbr_i in ngbrs[k]])))
        if  np.random.uniform(0,1) < np.exp(-beta * delta_E):
            microstate[k] *= -1
            energy += delta_E
    # Pasado el transiente, se comienzan a guardar las energías
    for i in range(N_steps):
        k = np.random.randint(N)
        delta_E = (2. * J * microstate[k]
                   * np.sum(np.array([microstate[ngbr_i] for ngbr_i in ngbrs[k]])))
        if np.random.uniform(0,1) < np.exp(-beta * delta_E):
            microstate[k] *= -1
            energy += delta_E
        energies.append(energy)
    
    # Se calcula la energía media por espín del microestado final
    N_steps2 = np.array(len(energies),dtype=np.int64)
    avg_energy_per_spin = np.float(np.sum(np.array(energies))/(N_steps2 * N * 1.))

    # Se devuelven las energías, el microestado final y la energía media
    # por espín del microestado final. 
    return energies, microstate, avg_energy_per_spin