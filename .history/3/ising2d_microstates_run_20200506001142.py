from ising2d_microstates import *


################################################################################################
# PANEL DE CONTROL
################################################################################################

# Decide si corre algoritmo matrix squaring con aproximación de trotter
run_microstates_algorithm = False

# Decide si corre algoritmo para cálculo de energía interna
run_avg_energy = True

# Decide si corre algoritmo para optimización de dx y beta_ini
run_optimization = False



################################################################################################
# PARÁMETROS GENERALES PARA LAS FIGURAS
################################################################################################

# Usar latex en texto de figuras y agrandar tamaño de fuente
plt.rc('text', usetex=True) 
plt.rcParams.update({'font.size':15,'text.latex.unicode':True})

# Obtenemos path para guardar archivos en el mismo directorio donde se ubica el script
script_dir = os.path.dirname(os.path.abspath(__file__))

# or just a list of the list of key value pairs
# list_key_value = [ [k,v] for k, v in dict.items() ]

if run_microstates_algorithm:
    L = 4
    microstates = ising_microstates(L)
    print('All microstates, each in a single row:')
    print(pd.DataFrame(microstates),'\n')
    neighbours = ising_neighbours(L)
    energies = ising_energy(microstates, neighbours, save_data=True)
    ising_energy_plot(energies, L, save_plot=True)

    microstate_rand_index = 2 ** (L*L) - np.random.randint(1, 2 ** (L*L))
    microstate_rand = microstates[microstate_rand_index,:]
    print('One random microstate as a 2D grid:')
    print(pd.DataFrame(microstate_rand.reshape((L,L))), '\n')
    ising_microstate_plot(microstate_rand, save_plot=True)

L = 5
energy_data_file_name = 'ising-energy-data-L_%d.csv'%L
# energies, frequencies = energies_to_frequencies(read_energy_data(energy_data_file_name))
# print(energies)
# print(frequencies)

#ising_energy_plot(None, L, read_data=True, energy_data_file_name=energy_data_file_name, show_plot=True, save_plot=True, plot_file_Name=None)
