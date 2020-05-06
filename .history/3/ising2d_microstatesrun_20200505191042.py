from ising2d_microstates import *


################################################################################################
# PARÁMETROS GENERALES PARA LAS FIGURAS
################################################################################################

# Usar latex en texto de figuras y agrandar tamaño de fuente
plt.rc('text', usetex=True) 
plt.rcParams.update({'font.size':15,'text.latex.unicode':True})

# Obtenemos path para guardar archivos en el mismo directorio donde se ubica el script
script_dir = os.path.dirname(os.path.abspath(__file__))



L = 3
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