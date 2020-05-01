from matrix_squaring import *

plt.rc('text', usetex=True) 
plt.rcParams.update({'font.size':15,'text.latex.unicode':True})

script_dir = os.path.dirname(os.path.abspath(__file__))

Z_file_name = script_dir + '/Z-ms-harmonic_potential-beta_max_10.000-beta_min_2.000-N_temp_300-x_max_5.000-nx_201-N_iter_9.csv' 
Z_file_read =  pd.read_csv(Z_file_name, index_col=0, comment='#')

beta_read = Z_file_read['beta']
temp_read = Z_file_read['temperature']
Z_read = Z_file_read['Z']

E_avg_harm = np.gradient(-np.log(Z_read),beta_read)

Z_file_name = script_dir + '/Z-ms-anharmonic_potential-beta_max_10.000-beta_min_2.000-N_temp_300-x_max_5.000-nx_201-N_iter_9.csv' 
Z_file_read =  pd.read_csv(Z_file_name, index_col=0, comment='#')

beta_read = Z_file_read['beta']
temp_read = Z_file_read['temperature']
Z_read = Z_file_read['Z']

E_avg_anharm = np.gradient(-np.log(Z_read),beta_read)

# Grafica.
plt.figure(figsize=(8,5))
plt.plot(temp_read,E_avg_harm,label=u'$\langle E \\rangle$ via matrix squaring (armónico)')
plt.plot(temp_read,E_avg_anharm,label=u'$\langle E \\rangle$ via matrix squaring (anarmónico)')
plt.plot(temp_read,E_QHO_avg_theo(beta_read),label=u'$\langle E \\rangle$ teórico (armónico)')
plt.legend(loc='best')
plt.xlabel(u'$T$')
plt.ylabel(u'$\langle E \\rangle$')
E_plot_name = 'E-ms-plot-compare-beta_max_10.000-beta_min_2.000-N_temp_300-x_max_5.000-nx_201-N_iter_9.eps'
E_plot_name = script_dir + '/' + E_plot_name
plt.savefig(E_plot_name)
plt.show()
plt.close()