from ising2d_microstates import *


################################################################################################
# PANEL DE CONTROL
################################################################################################

# Decide si corre algoritmo para calcular microestados de energía
run_microstates_algorithm = False

# Decide si corre algoritmo para cálculo de contribuciones a la función partición
# por cada valor de energía 
run_Z_contributions_algorithm = True

# Decide si corre algoritmo de aproximación de función partición
run_Z_approx_algorithm = False

# Decide si corre algoritmo para optimización de dx y beta_ini
run_specific_heat_algorithm = False

# Decide si corre demostración de asimetría para L impares
run_odd_asymmetry = False



################################################################################################
# PARÁMETROS GENERALES PARA LAS FIGURAS
################################################################################################

# Usar latex en texto de figuras y agrandar tamaño de fuente
plt.rc('text', usetex=True) 
plt.rcParams.update({'font.size':15,'text.latex.unicode':True})

# Obtenemos path para guardar archivos en el mismo directorio donde se ubica el script
script_dir = os.path.dirname(os.path.abspath(__file__))


# Algoritmo para calcular microestados
if run_microstates_algorithm:
    # Tamaño del sistema
    L = 2
    # Decide si pone condiciones de frontera libres
    free_boundary_conditions = False
    energy_plot_kwargs = {
                          'microstate_energies': None,
                          'L': L,
                          'read_data': True,
                          'energy_data_file_name': None,
                          'interpolate_energies': False,
                          'show_plot': True,
                          'save_plot': False,
                          'plot_file_Name': None,
                          }

    print('--------------------------------------')
    print('--------------------------------------')
    print('Microstates algorithm')
    print('--------------------------------------\n')
    print('--------------------------------------------')
    print('Grid: L x L = %d x %d'%(L, L))
    print('--------------------------------------------')

    # Calcula los microestados del sistema solo si read_data=False.
    if not energy_plot_kwargs['read_data']:
        
        # Genera todos los microestados posibles 
        microstates = ising_microstates(L)
        
        # Calcula los vecinos
        neighbours = ising_neighbours_free(L) if free_boundary_conditions \
                                            else ising_neighbours(L)
        
        # Cálculo de energía para cada microestado
        t_0 = time()    
        energies = ising_energy(microstates, neighbours,
                                save_data = not free_boundary_conditions)
        t_1 = time()
        comp_time = t_1-t_0
        # Imprime log del algoritmo
        print('--------------------------------------------------------\n'
            + 'Explicit energies:  L = %d --> computation time = %.3f \n'%(L,comp_time)
            + '--------------------------------------------------------\n')
        
        energy_plot_kwargs['microstate_energies'] = energies
        print('--------------------------------------')
        print('All microstates, each in a single row:')
        print('--------------------------------------')
        print(pd.concat([pd.DataFrame(microstates),
                        pd.DataFrame({'Energy': energies})],
                        axis=1, 
                        sort=False
                        ),
              '\n')

    # Grafica histograma de energías \Omega(E)
    ising_energy_plot(**energy_plot_kwargs)

    microstate_rand = np.random.choice([-1,1], L*L)
    print('-----------------------------------')
    print('One random microstate as a 2D grid:')
    print('-----------------------------------')
    print(pd.DataFrame(microstate_rand.reshape((L,L))), '\n')

    # Grafica un microestado aleatorio
    ising_microstate_plot(microstate_rand, save_plot=True)


# Algoritmo para calcular contribuciones a la función partición: 
# Omega(E)*e^{-beta E}, que es proporcional a p_E
if run_Z_contributions_algorithm:
    print('--------------------------------------')
    print('--------------------------------------')
    print('Z contributions algorithm')
    print('--------------------------------------')
    kwargs = {
        'microstate_energies': None,
        'L': 5,
        'beta': 1.,
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

# Algoritmo de aproximación de la función partición: equivalencia con ensamble
# microcanónico
if run_Z_approx_algorithm:
    print('--------------------------------------')
    print('--------------------------------------')
    print('Z approximation algorithm')
    print('--------------------------------------')
    approx_partition_func(read_data=True, save_plot=True)

# Algoritmo para graficar calor específico
if run_specific_heat_algorithm:
    print('--------------------------------------')
    print('--------------------------------------')
    print('Specific Heat algorithm')
    print('--------------------------------------\n')
    kwargs = {
        'microstate_energies_array': [None, None, None, None],
        'L_array': [2, 3, 4, 5],
        'beta_min': 1/5,
        'beta_max': 1.,
        'N_beta': 1000,
        'read_data': True,
        'energy_data_file_name': None,
        'show_plot': True,
        'save_plot': True,
        'plot_file_Name': None,
        'save_cv_data': True,
        }

    plot_specific_heat_cv(**kwargs)

# Algoritmo para mostrar que la asimetría en histograma de Omega para L impar 
# se debe a las condiciones de frontera periódicas
if run_odd_asymmetry:
    print('--------------------------------------')
    print('--------------------------------------')
    print('L odd energy asymmetry demonstration')
    print('--------------------------------------\n')
    L = 3
    ising_odd_L_energy_asymmetry(L, save_plot=True)