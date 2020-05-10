from ising2d_microstates import *


################################################################################################
# PANEL DE CONTROL
################################################################################################

# Decide si corre algoritmo para calcular microestados de energía
run_microstates_algorithm = False

# Decide si corre algoritmo para cálculo de contribuciones a la función partición
# por cada valor de energía 
run_Z_contributions_algorithm = False

# Decide si corre algoritmo de aproximación de función partición
run_Z_approx_algorithm = True

# Decide si corre algoritmo para optimización de dx y beta_ini
run_specific_heat_algorithm = False



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

if run_Z_contributions_algorithm:

    kwargs = {
        'microstate_energies': None,
        'L': 5,
        'beta': 0.5,
        'beta_max': None,
        'N_beta': 100,
        'read_data': True,
        'energy_data_file_name': None,
        'plot_histogram': True,
        'show_plot': True,
        'save_plot': True,
        'plot_file_Name': None,
        }
    
    Z_array, statistical_weights_array, beta_array, energies, omegas = \
        partition_func_stat_weights(**kwargs)
    
    # Acá está la clave para  la pregunta de la mitad en la pag 5   
    # el gráfico de log(Z) es lineal a altos beta ya que Z es aproximadamente una exponencial para estos valores
    # plt.plot(beta_array, np.log(Z_array))
    # plt.show()

if run_Z_approx_algorithm:
    approx_partition_func(read_data=True, save_plot=True)

if run_specific_heat_algorithm:
    
    kwargs = {
        'microstate_energies_array': [None, None, None, None],
        'L_array': [2, 3, 4, 5],
        'beta_min': 1/8,
        'beta_max': 10,
        'N_beta': 1000,
        'read_data': True,
        'energy_data_file_name': None,
        'show_plot': True,
        'save_plot': True,
        'plot_file_Name': None,
        }

    plot_specific_heat_cv(**kwargs)