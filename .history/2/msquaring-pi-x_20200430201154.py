from matrix_squaring import *

script_dir = os.path.dirname(os.path.abspath(__file__))
plt.rc('text', usetex=True) 
plt.rcParams.update({'font.size':15,'text.latex.unicode':True})

beta_fin = 4

msq_file = script_dir + '/pi_x-ms-harmonic_potential-beta_fin_4.000-x_max_5.000-nx_201-N_iter_9.csv'
matrix_squaring_data = pd.read_csv(msq_file, index_col=0, comment='#')
grid_x, x_weights_harm = matrix_squaring_data['position_x'], matrix_squaring_data['prob_density']

msq_file = script_dir + '/pi_x-ms-anharmonic_potential-beta_fin_4.000-x_max_5.000-nx_201-N_iter_9.csv'
matrix_squaring_data = pd.read_csv(msq_file, index_col=0, comment='#')
grid_x, x_weights_anharm = matrix_squaring_data['position_x'], matrix_squaring_data['prob_density']

plt.figure(figsize=(8,5))
plt.plot(grid_x, x_weights_harm, 
            label = 'Matrix squaring +\nfórmula de Trotter.\n(armónico))')
plt.plot(grid_x, x_weights_anharm,
            label = 'Matrix squaring +\nfórmula de Trotter.\n(anarmónico))')
plt.xlabel(u'$x$')
plt.ylabel(u'$\pi^{(Q)}(x;\\beta)$')
plt.legend(loc='best',title=u'$\\beta=%.2f$'%beta_fin)
plt.tight_layout()

plot_file_name = 'pi_x-ms-plot-compare_potential-beta_fin_4.000-x_max_5.000-nx_201-N_iter_9.eps'
plot_file_name = script_dir + '/' + plot_file_name
plt.savefig(plot_file_name)

plt.show()
plt.close()