from ising2d-microstates import *

L = 4
microstates = ising_microstates(L)
print('All microstates, each in a single row:\n',pd.DataFrame(microstates),'\n')
neighbours = ising_neighbours(L)
energies = ising_energy(microstates, neighbours, save_data=True)
ising_energy_plot(energies, L, save_plot=True)


microstate_rand_index = 2 ** (L*L) - np.random.randint(1, 2 ** (L*L))
microstate_rand = microstates[microstate_rand_index,:]
print('One random microstate as a 2D grid:\n', 
      pd.DataFrame(microstate_rand.reshape((L,L))),
      '\n')
ising_microstate_plot(microstate_rand, save_plot=True)