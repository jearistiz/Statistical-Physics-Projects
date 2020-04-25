from path_integral_naive_sampling import *


################################################################################################
# PANEL DE CONTROL
#
# Decide si corre algoritmo matrix squaring
run_path_int = False
# Decide si corre algoritmo para cálculo de energía interna
run_plot_1 = True
# Decide si corre algoritmo para optimización de dx y beta_ini
run_plot_2 = True
# Decide si corre algoritmo para optimización de dx y beta_ini
run_plot_3 = True
#
#
################################################################################################



###############################################################################################
# PARÁMETROS GENERALES PARA LAS FIGURAS
#
# Usar latex en texto de figuras y agrandar tamaño de fuente
plt.rc('text', usetex=True) 
plt.rcParams.update({'font.size':15,'text.latex.unicode':True})
# Obtenemos path para guardar archivos en el mismo directorio donde se ubica el script
script_dir = os.path.dirname(os.path.abspath(__file__))
#
###############################################################################################



################################################################################################
# Corre algoritmo y guarda caminos en pathss_x. Además guarda datos en archivo CSV
#
#

# Parámetros físicos
N_path = 10
beta = 4.
N_iter =  int(1e6) # int(1e5)
delta = 0.5
potential, potential_string = harmonic_potential, 'harmonic_potential'
append_every = 1
save_paths_data = True
paths_file_name = 'pi_x-pi-harmonic_potential-beta_4.000-N_path_10-N_iter_1000000-delta_0.500-append_every_1.csv'  #None
paths_relevant_info = None
pathss_x = None

# Calcula caminos de integral de camino pathss_x = [camino_1, camino_2, ...]
# camino_i = [x_i[0], x_i[1], x_i[2], ..., x_i[N_path-1]]
if run_path_int:
    pathss_x = path_naive_sampling(N_path, beta, N_iter, delta, potential, potential_string,
                                append_every, save_paths_data, paths_file_name,
                                paths_relevant_info)

#
#
################################################################################################

################################################################################################
# Primera figura: muestra histograma x[0] y un path.
#
#

# Parámetros técnicos de figuras
pathss_x = pathss_x
read_paths_data = False
paths_file_name = None
N_plot = 201
x_max = 3
N_beta_ticks = N_path+1
show_theory = False
show_matrix_squaring = True
show_path = True
show_compare_hist = False
show_complete_path_hist = False
save_plot = False # True
show_plot = True
msq_file = 'pi_x-ms-harmonic_potential-beta_fin_4.000-x_max_5.000-nx_201-N_iter_7.csv'

plot_file_name =    'pi_x-pi-plot-%s-beta_%.3f-N_path_%d-N_iter_%d-delta_%.3f-append_every_%d-x_max_%.3f-theory_%d-ms_%d-path_%d-compare_%d-complete_path-%d.eps'\
                    %(potential_string,beta,N_path,N_iter,delta,append_every,x_max,show_theory,show_matrix_squaring,show_path,show_compare_hist,show_complete_path_hist)


figures_fn(pathss_x=None, read_paths_data=False, paths_file_name=None,
               beta=4., N_path=10, N_iter=int(1e5), delta=0.5, 
               potential_string='harmonic_potential', append_every=1, 
               N_plot=201, x_max=3, N_beta_ticks=11, msq_file=None,
               plot_file_name=None, show_theory=True, show_matrix_squaring=True,
               show_path=True, show_compare_hist=True, show_complete_path_hist=True,
               save_plot=True, show_plot=True)

if run_plot_1:
    figures_fn(pathss_x, read_paths_data, paths_file_name, beta, N_plot, x_max, N_iter,
            append_every, N_beta_ticks, msq_file, plot_file_name,show_theory, show_matrix_squaring,
            show_path, show_compare_hist, show_complete_path_hist, save_plot, show_plot)
#
#
################################################################################################



################################################################################################
# Segunda figura: compara histograma x[0] con histograma hecho con x[0],...,x[N-1]
#
#
show_theory=False
show_matrix_squaring=True
show_path=False
show_compare_hist=True
show_complete_path_hist=False
save_plot = False # True
show_plot = True

plot_file_name =    'pi_x-pi-plot-%s-beta_%.3f-N_path_%d-N_iter_%d-delta_%.3f-append_every_%d-x_max_%.3f-theory_%d-ms_%d-path_%d-compare_%d-complete_path-%d.eps'\
                    %(potential_string,beta,N_path,N_iter,delta,append_every,x_max,show_theory,show_matrix_squaring,show_path,show_compare_hist,show_complete_path_hist)

if run_plot_2:
    figures_fn(pathss_x, read_paths_data, paths_file_name, beta, N_plot, x_max, N_iter,
            append_every, N_path+1, msq_file, plot_file_name,show_theory, show_matrix_squaring,
            show_path, show_compare_hist, show_complete_path_hist, save_plot, show_plot)
#
#
################################################################################################



################################################################################################
# Tercera figura: compara histograma x[0] con histograma hecho con x[0],...,x[N-1]
#
#

show_theory=False
show_matrix_squaring=True
show_path=False
show_compare_hist=False
show_complete_path_hist=True
save_plot = False # True
show_plot = True

plot_file_name =    'pi_x-pi-plot-%s-beta_%.3f-N_path_%d-N_iter_%d-delta_%.3f-append_every_%d-x_max_%.3f-theory_%d-ms_%d-path_%d-compare_%d-complete_path-%d.eps'\
                    %(potential_string,beta,N_path,N_iter,delta,append_every,x_max,show_theory,show_matrix_squaring,show_path,show_compare_hist,show_complete_path_hist)

if run_plot_3:
    figures_fn(pathss_x, read_paths_data, paths_file_name, beta, N_plot, x_max, N_iter,
            append_every, N_path+1, msq_file, plot_file_name,show_theory, show_matrix_squaring,
            show_path, show_compare_hist, show_complete_path_hist, save_plot, show_plot)
#
#
################################################################################################