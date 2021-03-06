from ising2d_metropolis import *

################################################################################################
# PANEL DE CONTROL
################################################################################################

# Decide si corre algoritmo para calcular microestados de energía
run_metropolis_energies_algorithm = False

# Decide si corre algoritmo que muestra la termalización
run_thermalization_algorithm = False

# Decide si corre algoritmo de calor específico
run_specific_heat_algorithm = True



################################################################################################
# PARÁMETROS GENERALES PARA LAS FIGURAS
################################################################################################

# Usar latex en texto de figuras y agrandar tamaño de fuente
plt.rc('text', usetex=True) 
plt.rcParams.update({'font.size':15,'text.latex.unicode':True})



# Muestreo de energías usando algoritmo Metrópolis
if run_metropolis_energies_algorithm:
    
    # Decide si lee o guarda datos y asigna nombres a los archivos
    read_ini_microstate_data = False
    save_final_microstate_data = True
    microstate_data_file_name = None
    save_energy_data = False
    energy_data_file_name = None

    # Muestra parámetros y tiempo de cómputo
    print_log = True

    # Decide si grafica microestado final
    plot_microstate = True
    # Parámetros para figura de microestado
    show_microstate_plot = True
    save_microstate_plot = True
    microstate_plot_file_name = None

    # Parámeros del algoritmo metrópolis para calcular energías
    T = 2.27
    beta = 1/T
    L = 128
    # Como se está usando numba, en microstate siempre hay que 
    # entregar el siguiente array con dtype=np.int64
    microstate = np.ones(L * L, dtype=np.int64)
    J = 1
    N_steps = int(1e8)  # int(L * L * 10000)
    N_transient = 0     # int( N_steps)

    # Asigna nombre a archivo con datos de microestado inicial/final
    if read_ini_microstate_data or save_final_microstate_data:
        if not microstate_data_file_name:
            microstate_data_file_name = \
                ('ising-metropolis-final-config-L_%d-temp_%.3f'%(L, T)
                 + '-N_steps_%d-N_transient_%d.csv'%(N_steps, N_transient))
        microstate_data_file_name = script_dir + '/' + microstate_data_file_name
        if read_ini_microstate_data:
            microstate = pd.read_csv(microstate_data_file_name, index_col=0, comment='#')
            microstate = np.array(microstate.to_numpy().transpose()[0], dtype=np.int64)
    
    metropolis_args = (microstate, read_ini_microstate_data,
                       L, beta, J, N_steps, N_transient)
    

    # Decide si grafica histograma de energías
    # (contribuciones proporcionales al factor de boltzmann  Omega(E) * e**(-beta E) / Z(beta))
    plot_energy_hist = False
    # Parámetros para graficar histograma de energías.
    energy_hist_plot_file_name = \
        ('ising-metropolis-Z_contributions-plot-L_'
         + '%d-temp_%.3f-N_steps_%d-N_transient_%d.pdf'%(L, T, N_steps, N_transient))
    energy_plot_kwargs = {
                        'microstate_energies': None,
                        'L': L,
                        'read_data': False,
                        'energy_data_file_name': None,
                        'interpolate_energies': False,
                        'show_plot': True,
                        'save_plot': True,
                        'normed': True,
                        'plot_file_Name': energy_hist_plot_file_name,
                        'x_lim': None,
                        'y_label': '$\Omega(E) e^{-\\beta E}/Z(\\beta)$',
                        'legend_title': 'Metrópolis. $T=%.3f$'%T,
                        }

    # Corre algoritmo metrópolis con parámetros dados e imprime tiempo de cómputo
    t_0 = time()
    energies, microstate, avg_energy_per_spin = ising_metropolis_energies(*metropolis_args)
    t_1 = time()

    # Imprime información relevante
    if print_log:
        comp_time = t_1 - t_0
        print_params = (L, T, N_steps, N_transient)
        print('\n----------------------------------------------------------------------------\n'
            + 'Ising 2D Metropolis:\n'
            + 'L = %d,  T = %.3f,  N_steps = %d,  N_transient = %d\n'%print_params
            + '<E>/N = %.4f\n'%avg_energy_per_spin
            + '--> computation time = %.3f \n'%comp_time
            + '----------------------------------------------------------------------------\n')

    # Guarda datos de energías muestreadas o microestado final en archivo CSV
    if save_energy_data:
        if not energy_data_file_name:
            energy_data_file_name = ('ising-metropolis-energy-data-L_%d-temp_%.3f'%(L, T)
                                     + '-N_steps_%d-N_transient_%d.csv'%(N_steps, N_transient))
        energy_data_file_name = script_dir + '/' + energy_data_file_name
        relevant_info = ['2D Ising ENERGIES, metropolis algorithm: L=%d   T=%.4f'%(L, T)
                         + '   N_steps=%d   N_transient=%d'%(N_steps, N_transient)]
        headers = ['energy']
        save_csv(energies, data_headers=headers, file_name=energy_data_file_name, 
                 relevant_info=relevant_info, print_data=False)
    if save_final_microstate_data:
        relevant_info = ['2D Ising FINAL MICROSTATE, metropolis algorithm: L=%d  '%(L)
                         + 'T=%.4f  N_steps=%d  N_transient=%d'%(T, N_steps, N_transient)]
        headers = ['spin']
        save_csv(microstate, data_headers=headers, file_name=microstate_data_file_name,
                 relevant_info=relevant_info, print_data=False)

    # Grafica microestado final. 
    if plot_microstate:
        mstate_plot_args = (np.array(microstate), L, beta, J, N_steps, N_transient,
                            show_microstate_plot, save_microstate_plot, microstate_plot_file_name)
        ising_metropolis_microstate_plot(*mstate_plot_args)

    if plot_energy_hist:
        energy_plot_kwargs['microstate_energies'] = np.array(energies)
        ising_energy_plot(**energy_plot_kwargs)
    
    del energies


# Algoritmo de termalización
if run_thermalization_algorithm:

    # Decide si imprime info del algoritmo
    print_log = True

    # Parámetros de algoritmo de termalización
    beta = np.array([1 /10., 1/3., 1/2., 1/1.])
    L = 2
    microstates_ini = np.ones( (len(beta), L * L), dtype=np.int64)
    read_ini_microstate_data = False
    J = 1
    N_steps = int(L * L * 1e4)
    N_transient = 0

    thermalization_args = \
        (microstates_ini, read_ini_microstate_data, L, beta, J, N_steps, N_transient)

    # Corre algoritmo de termalización
    t_0 = time()
    avg_energy_per_spin_array, beta, *dont_need = thermalization_demo(*thermalization_args)
    t_1 = time()
    
    if print_log:
        comp_time = t_1 - t_0
        print_params = (L, N_steps, N_transient)
        print('\n----------------------------------------------------------------------------\n'
            + 'Ising 2D Metropolis thermalization:\n'
            + 'T = ' + str(list(1/np.array(beta))) + '\n'
            + 'L = %d,  N_steps = %d,  N_transient = %d\n'%print_params
            + '<E>/N = ' + str([E_over_N[-1] for E_over_N in avg_energy_per_spin_array]) + '\n'
            + '--> computation time = %.3f \n'%comp_time
            + '----------------------------------------------------------------------------\n')
    
    # Parámetros de figura de termalización
    thermaization_data_file_name = None
    show_plot = True
    save_plot = True
    plot_file_Name = None

    thermalization_plot_args = (avg_energy_per_spin_array, beta, L, J, N_steps,
                                N_transient, thermaization_data_file_name, show_plot,
                                save_plot, plot_file_Name)

    plot_thermalization_demo(*thermalization_plot_args)

# Algoritmo de calor específico
if run_specific_heat_algorithm:
    
    # Si read_cv_data=True, el algoritmo no corre, sino que se leen los datos de un archivo.
    read_cv_data = True
    save_cv_data = True
    cv_data_file_name = None


    # Decide si imprime info del algoritmo
    print_log = True
    # Parámetros del algoritmo
    L_array = np.array([2, 3, 4, 5, 8, 16, 32, 64])
    N_steps_factor = int(8e4)
    N_transient_factor = 0.7
    J = 1
    T_min = 1.0
    T_max = 5.0
    N_temp = 100

    several_cv_args = (L_array, N_steps_factor, N_transient_factor,
                       J, T_min, T_max, N_temp)

    # Corre el algoritmo
    t_0 = time()
    if not read_cv_data:
        cv_arrays, T_arrays, L_array, N_steps_factor = \
                                     several_specific_heats(*several_cv_args)
    else: 
        cv_arrays, T_arrays = None, None
    t_1 = time()
    
    # Imprime info del algoritmo
    if print_log or save_energy_data:
        comp_time = t_1 - t_0
        line0 = '---------------------------------------------------------------------------\n'
        line1 = 'Ising 2D Metropolis specific heat (cv) plot:'
        line2 = 'T_min = %.3f,  T_max = %.3f,  N_temp = %d'%(T_min, T_max, N_temp)
        line3 = 'L = ' + str(list(L_array))
        line4 = ('N_steps_factor = %d '%N_steps_factor
                 + '(N_steps = L*L*N_steps_factor, '
                 + 'N_transient = %.2f N_steps)'%N_transient_factor)
        line5 = '--> computation time = %.3f'%comp_time
        if print_log and not read_cv_data:
            print('\n' + line0 + line1 + '\n' + line2 + '\n' + line3 + '\n' + line4 + '\n'
                  + line5 + '\n' + line0)
        if print_log and read_cv_data:
            print('Los datos se leyeron de un archivo, no se generaron en este momento.')

    # Guarda datos de energías muestreadas o microestado final en archivo CSV
    if save_cv_data and not read_cv_data:
        if not cv_data_file_name:
            L_string = '_'.join([str(L) for L in L_array])
            cv_data_file_name = ('ising-metropolis-specific_heat-plot-L_' + L_string
                + '-N_steps_factor_%d-N_transient_factor_%d-T_min_%.3f-T_max_%.3f-N_temp_%d.csv'
                % (N_steps_factor, N_transient_factor, T_min, T_max, N_temp))
        cv_data_file_name = script_dir + '/' + cv_data_file_name
        relevant_info = [line1, line2 , line3, line4, line5]
        headers = np.array([ ['Temperature', 'cv (L=%d)'%L] for L in L_array]).flatten()
        shape = (2*len(L_array), len(cv_arrays[0]))
        cv_data = np.array([[T, cv_arrays[i]] for i, T in enumerate(T_arrays)]).reshape(shape)
        save_csv(cv_data.transpose(), data_headers=headers, file_name=cv_data_file_name,
                 relevant_info=relevant_info, print_data=False)
    if save_cv_data and read_cv_data:
        print('Se escogió leer los datos del calor específico de un archivo de texto.')

    # Parámetros de la gráfica
    show_plot = True
    save_plot = True
    read_cv_data_part_1 = True
    plot_file_Name = None

    cv_plot_args = (cv_arrays, T_arrays, L_array, N_steps_factor, N_transient_factor,
                    T_min, T_max, N_temp, J, read_cv_data_part_1, read_cv_data,
                    cv_data_file_name, show_plot, save_plot, plot_file_Name)

    specific_heat_plot(*cv_plot_args)

    pass