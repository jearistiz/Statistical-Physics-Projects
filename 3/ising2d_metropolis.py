# -*- coding: utf-8 -*-
from ising2d_microstates import *


# def energy(S, L, ngbrs):
#     N = L * L
#     E = 0.
#     for k in range(N):
#         E -= S[k] * sum(S[ngbr_i] for ngbr_i in ngbrs[k])
#     return 0.5 * E

def ising_metropolis_energies(L=6, beta=1., J=1, N_steps=10000, N_transient=100,
                              read_config_data=False,
                              save_config_data=False, config_data_file_name=None,
                              save_energy_data=False, energy_data_file_name=None,
                              print_log=True):
    N = L * L
    ngbrs = ising_neighbours(L)
    if read_config_data or save_config_data:
        if not config_data_file_name:
            config_data_file_name = ('ising-metropolis-final-config-L_%d-beta_%.3f'%(L, beta)
                                     + '-N_steps_%d-N_transient_%d.csv'%(N_steps, N_transient))
        config_data_file_name = script_dir + '/' + config_data_file_name
    if read_config_data:
       microstate = pd.read_csv(config_data_file_name, index_col=0, comment='#')
       microstate = microstate.to_numpy().tolist()
    else: 
        microstate = np.random.choice([1,-1], N).tolist()
    energy = ising_energy([microstate], ngbrs, J=J, print_log=False)[0]
    energies = []
    # Transiente
    for i in range(N_transient):
        k = np.random.randint(N)
        delta_E = 2. * J  * microstate[k] * np.sum([microstate[ngbr_i] for ngbr_i in ngbrs[k]])
        if np.random.uniform() < np.exp(-beta * delta_E):
            microstate[k] *= -1
            energy += delta_E
    # Se comienzan a guardar las energías
    for i in range(N_steps):
        k = np.random.randint(N)
        delta_E = 2. * J * microstate[k] * np.sum([microstate[ngbr_i] for ngbr_i in ngbrs[k]])
        if np.random.uniform() < np.exp(-beta * delta_E):
            microstate[k] *= -1
            energy += delta_E
        energies.append(energy)
    if save_energy_data:
        if not energy_data_file_name:
            energy_data_file_name = ('ising-metropolis-energy-data-L_%d-beta_%.3f'%(L, beta)
                                     + '-N_steps_%d-N_transient_%d.csv'%(N_steps, N_transient))
        energy_data_file_name = script_dir + '/' + energy_data_file_name
        relevant_info = ['2D Ising ENERGIES, metropolis algorithm: L=%d   beta=%.4f'%(L, beta)
                         + '   N_steps=%d   N_transient=%d'%(N_steps, N_transient)]
        headers = ['energy']
        save_csv(energies, data_headers=headers, file_name=energy_data_file_name, 
                 relevant_info=relevant_info, print_data=False)
    if save_config_data:
        relevant_info = ['2D Ising FINAL MICROSTATE, metropolis algorithm: L=%d  '%(L)
                         + 'beta=%.4f  N_steps=%d  N_transient=%d'%(N_steps, N_transient, beta)]
        headers = ['spin']
        save_csv(microstate, data_headers=headers, file_name=energy_data_file_name, 
                 relevant_info=relevant_info, print_data=False)
    print(sum(energies)/(len(energies) * N))   
    return energies, microstate


def ising_microstate_plot_metropolis(config, L, beta, J=1, N_steps=10000, N_transient=100,
                                     show_plot=True, save_plot=False, plot_file_name=None):
    
    if save_plot:
        if not plot_file_name:
            plot_file_name = ('ising-metropolis-config-plot-L_%d-beta_%.3f'%(L, beta)
                              + '-N_steps_%d-N_transient_%d.csv'%(N_steps, N_transient))
        plot_file_name = script_dir + '/' + plot_file_name
        plt.savefig(plot_file_name)
    
    ising_microstate_plot(np.array(config), show_plot, save_plot, plot_file_name)

t_0 = time()
energies, microstate = ising_metropolis_energies(L=6, beta=1/2., N_steps=100000, N_transient=30000)
t_1 = time()
ising_microstate_plot_metropolis(np.array(microstate), 6, 0.5)
print(t_1-t_0)