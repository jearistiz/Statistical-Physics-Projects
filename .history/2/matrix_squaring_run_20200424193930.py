from matrix_squaring import *


################################################################################################
# PANEL DE CONTROL
################################################################################################

# Decide si corre algoritmo matrix squaring con aproximación de trotter
run_ms_algorithm = False

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



################################################################################################
# CORRE ALGORITMO MATRIX SQUARING
################################################################################################

if run_ms_algorithm:
    
    # Parámetros físicos del algoritmo
    physical_kwargs = {
        'x_max': 5.,
        'nx': 201,
        'N_iter': 7,
        'beta_fin': 4,
        'potential': harmonic_potential,
        'potential_string':   'harmonic_potential'
        }

    # Parámetros técnicos (generar archivos y figuras, etc.)
    technical_kwargs = {
        'print_steps': False,
        'save_data': True,
        'csv_file_name': None,
        'relevant_info': None,
        'plot': True,
        'save_plot': True,
        'show_plot': True,
        'plot_file_name': None
        }

    kwargs = {**physical_kwargs, **technical_kwargs}

    rho, trace_rho, grid_x = run_pi_x_sq_trotter(**kwargs)



################################################################################################
# CORRE ALGORITMO PARA CÁLCULO DE ENERGÍA INTERNA
################################################################################################

if run_avg_energy:

    # Parámetros técnicos función partición y cálculo de energía 
    technical_Z_kwargs = {
        'read_Z_data': False,
        'generate_Z_data': True,
        'Z_file_name': None,
        'plot_energy': True,
        'save_plot_E': True,
        'show_plot_E': True,
        'E_plot_name': None
    }

    # Parámetros físicos para calcular Z y <E>
    physical_kwargs = {
        'temp_min': 1./10,
        'temp_max': 1./2,
        'N_temp': 10,
        'potential': harmonic_potential,
        'potential_string': 'harmonic_potential'
    }

    # Más parámetros técnicos
    more_technical_kwargs = {
        'save_Z_csv': True,
        'relevant_info_Z': None,
        'print_Z_data': False,
        'x_max': 7.,
        'nx': 201,
        'N_iter': 7,
        'print_steps': False,
        'save_pi_x_data': False,
        'pi_x_file_name': None,
        'relevant_info_pi_x': None,
        'plot_pi_x': False,
        'save_plot_pi_x': False,
        'show_plot_pi_x': False,
        'plot_pi_x_file_name': None
    }

    kwargs = {**technical_Z_kwargs, **physical_kwargs, **more_technical_kwargs}

    average_energy(**kwargs)



################################################################################################
# CORRE ALGORITMO PARA OPTIMIZACIÓN DE DX Y BETA_INI
################################################################################################

if run_optimization:

    # Parámetros físicos
    physical_kwargs = {
        'beta_fin': 4,
        'x_max': 5,
        'potential': harmonic_potential,
        'potential_string': 'harmonic_potential',
        'nx_min': 10,
        'nx_max': 310,
        'nx_sampling': 60,
        'N_iter_min': 8,
        'N_iter_max': 20
    }

    # Parámetros técnicos
    technical_kwargs = {
        'generate_opt_data': True,
        'read_opt_data': False,
        'save_opt_data': True,
        'opt_data_file_name': None,
        'opt_relevant_info': None,
        'plot_opt': True,
        'show_opt_plot': True,
        'save_plot_opt': True,
        'opt_plot_file_name': None,
        'print_summary': True,
    }

    kwargs = {**physical_kwargs, **technical_kwargs}

    error, dx_grid, beta_ini_grid, comp_time = optimization(**kwargs)
