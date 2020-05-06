from ising2d_microstates import *


################################################################################################
# PANEL DE CONTROL
################################################################################################

# Decide si corre algoritmo matrix squaring con aproximación de trotter
run_microstates_algorithm = True

# Decide si corre algoritmo para cálculo de energía interna
run_Z_contributions = True

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

if run_Z_contributions:
    
    energies
    L = 3
    kwargs = {
              'beta'=4,
              'beta_max'=None,
              'N_values'=None,
              'read_data'=False,
              'energy_data_file_name'=None,
              'plot_histogram'=False,
              'show_plot'=True,
              'save_plot'=False,
              'plot_file_Name'=None,
              }


    
    pass
