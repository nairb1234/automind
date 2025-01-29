#Brian2 functions related to additional inputs 
import brian2 as b2
import numpy as np

def DM(all_param_dict, mu1 = 1*10**-9, mu2 = 2.5*10**-9, sigma=6*10**-12):
    """
    DM task mimicking Wang 2002 https://www.cell.com/neuron/fulltext/S0896-6273(02)01092-9
    
    Returns:
        - stimulus_A: Stimulus for population A 
        - stimulus_B: Stimulus for population B 
    """

    #Dicts
    param_dict_settings = all_param_dict["params_settings"]

    #Set random seeds
    b2.seed(param_dict_settings["random_seed"])
    b2.defaultclock.dt = param_dict_settings["dt"]

    #Setting stimulation and simulation length 
    stim_steps = int(5 * b2.second / b2.defaultclock.dt) #Potentially change stim length to user added input 
    one_second = int(b2.second / b2.defaultclock.dt)
    buffer_period = 10 * one_second #Allow network to stabilise first
    total_steps = int(param_dict_settings['sim_time']/b2.defaultclock.dt)
    if stim_steps > total_steps:
        raise ValueError("Stimulus steps exceed total simulation steps.")

    # Generate Gaussian input
    stim_1, stim_2 = np.zeros(total_steps), np.zeros(total_steps)
    stim_2[buffer_period:buffer_period+stim_steps] = np.random.normal(mu1, sigma, stim_steps)
    stim_1[buffer_period:buffer_period+stim_steps] = np.random.normal(mu2, sigma, stim_steps)
    
    # Ensure non-negative firing rates
    stim_1 = np.maximum(stim_1, 0)
    stim_2 = np.maximum(stim_2,0)
    return stim_1, stim_2

#Collect cognitive tasks defined in other functions / test input strength required 
def cog_tasks(all_param_dict, mode, scaling):
    #Separate parameter dictionaries
    param_dict_settings = all_param_dict["params_settings"]
    param_dict_net = all_param_dict["params_net"]
    #Set random seeds
    b2.seed(param_dict_settings["random_seed"])
    b2.defaultclock.dt = param_dict_settings["dt"]

    #Setting stimulation and simulation length 
    stim_steps = int(10 * b2.second / b2.defaultclock.dt) #Potentially change stim length to user added input 
    one_second = int(b2.second / b2.defaultclock.dt)
    buffer_period = 10 * one_second #Allow network to stabilise first
    total_steps = int(param_dict_settings['sim_time']/b2.defaultclock.dt)
    if stim_steps > total_steps:
        raise ValueError("Stimulus steps exceed total simulation steps.")
    #N_pop numbers 
    n_neurons, exc_prop = param_dict_net["N_pop"], param_dict_net["exc_prop"]
    N_exc = int(n_neurons * exc_prop) 

    #Different stimulation modes 
    stim = np.zeros(total_steps) 
    #Test mode: Stimulation signal with step increasing signals to determine which level of input is suitable for networks
    if mode == 'test':
        for i in range(10):
            stim[buffer_period + one_second*i:buffer_period + one_second*(i+1)] += i * 2.5 * 10 ** -10
    if mode == 'DM':
        #np.random.seed(param_dict_settings["random_seed"])
        stim_1, stim_2= DM(all_param_dict)
        stim = [stim_1,stim_2]
    #input_signal = b2.TimedArray(stim*b2.amp, dt = b2.defaultclock.dt)

    if scaling == 'all':
        input_scaling = np.ones(n_neurons).astype(int)
    elif scaling == 'rand':
        input_scaling = np.random.rand(n_neurons)
    elif scaling == 'subset':
        input_scaling = np.random.randint(0,2,n_neurons)
    elif scaling == 'e_only':
        input_scaling = np.zeros(n_neurons).astype(int)
        input_scaling[:N_exc] = 1
    elif scaling == 'i_only':
        input_scaling = np.zeros(n_neurons).astype(int)
        input_scaling[N_exc:] = 1
    else:
        raise ValueError(f"Invalid input for 'scaling': {scaling}. Choose from 'all', 'rand', 'subset', 'e_only' or 'i_only'.")
    #Check shape of stim and get weights for each stimuli 
    n_input = np.shape(stim)[0]
    #print(n_input)
    all_weights = np.zeros([n_input,n_neurons])
    for i in range(n_input):
        all_weights[i,:] = input_scaling
    return stim, all_weights
    '''
    #Create a single noisy input signal between 0 and 1 
    stim_steps = int(1 * b2.second / b2.defaultclock.dt)
    total_steps = int(param_dict_settings['sim_time']/b2.defaultclock.dt)
    random_signal = np.random.rand(stim_steps)
    stim = np.zeros(total_steps)
    stim[0:stim_steps] = random_signal
    DM_input = b2.TimedArray(stim*b2.amp, dt = b2.defaultclock.dt)
    '''

def get_input_configs(cluster_list,stim_list,weight_list):
    #Returns dictionary of the variables to be used in the network operation 
    if not (len(cluster_list) == len(stim_list) == len(weight_list)):
        raise ValueError("Must provide equal number of cluster list, stimuli, and weights")
    
    # Create configurations
    input_configs = []
    for clusters, stim, weights in zip(cluster_list, stim_list, weight_list):
        config = {
            'clusters': clusters,
            'stim': stim,
            'weights': weights
        }
        input_configs.append(config)
    
    return input_configs

def create_input_operation(E_pop, input_configs, membership):
    """
    Creates a network operation to handle multiple inputs to different neuron groups.
    
    Parameters:
    -----------
    E_pop : brian2.NeuronGroup
        The excitatory neuron population
    input_configs : list of dict
        List of input configurations, each containing:
        {
            'clusters': list of cluster indices,
            'stim': array of stimulus values,
            'weights': array of weight values
        }
    membership : list of lists
        Cluster membership for each neuron
    """
    N_exc = len(E_pop)
    E_pop.namespace['stim1'] = input_configs[0]['stim'] #Make this part more general 
    E_pop.namespace['stim2'] = input_configs[1]['stim'] 
    # Pre-compute masks for each input
    masks = []
    for config in input_configs:
        mask = np.zeros(N_exc)
        for neuron_idx in range(N_exc):
            if any(cluster in config['clusters'] for cluster in membership[neuron_idx]):
                mask[neuron_idx] = 1
        masks.append(mask)
    
    # Convert stimuli to arrays and store with masks
    input_data = [(np.array(cfg['stim']), 
                   np.array(cfg['weights'][:N_exc]) * mask) 
                  for cfg, mask in zip(input_configs, masks)]

    # Create the network operation
    @b2.network_operation(dt=b2.defaultclock.dt)
    def input_operation(t):
        # Current timestep
        idx = int(t / b2.defaultclock.dt)
        
        # Sum all inputs for this timestep
        total_input = np.zeros(N_exc)
        for stim, weights in input_data:
            if idx < len(stim):  # Check if we still have stimulus data
                total_input += stim[idx] * weights
        
        # Apply the summed input to the membrane potential
        E_pop.I_ext = total_input * b2.amp
    
    return input_operation