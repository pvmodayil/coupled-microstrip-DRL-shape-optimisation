policy_kwargs: dict[str, int | list[int]] = dict(log_std_init=-3, net_arch=[400, 300]) # setting nn architecture
hyperparams: dict[str, int|float|bool|str] = {
    'ent_coef': 'auto_0.01',  # setting entropy coefficient to auto boosts exploration
    'target_entropy':-9,
    'batch_size':512,  
    'buffer_size': 20000,        
    'learning_rate': 0.0004,               
    'gamma': 0.98,      
    'tau':0.02,
    'learning_starts':1000,
    'use_sde':True      
}

def get_hyper_params() -> tuple[dict[str, int | list[int]], dict[str, int|float|bool|str]]:
     return policy_kwargs, hyperparams
 
# from stable_baselines3.common.noise import NormalActionNoise
# from coupledstrip_env import CoupledStripEnv
# import numpy as np

# policy_kwargs: dict[str, int | list[int]] = dict(log_std_init=-3, net_arch=[256, 512, 256]) # setting nn architecture
# hyperparams: dict[str, int|float|bool|str|NormalActionNoise] = {
#     'ent_coef': 'auto_0.01',  # setting entropy coefficient to auto boosts exploration
#     'target_entropy':-9,
#     'batch_size':512,  
#     'buffer_size': 20000,        
#     'learning_rate': 0.0004,               
#     'gamma': 0.98,      
#     'tau':0.02,
#     'learning_starts':1000,
#     'use_sde':False      
# }

# def get_hyper_params(env: CoupledStripEnv) -> tuple[dict[str, int | list[int]], dict[str, int|float|bool|str|NormalActionNoise]]:
#     n_actions: int = env.action_space.shape[-1]
#     mean: np.ndarray = np.zeros(n_actions)
#     std_dev = 0.1 * np.ones(n_actions)  # Adjust std deviation to your problem scale
#     action_noise: NormalActionNoise = NormalActionNoise(mean=mean, sigma=std_dev)
    
#     hyperparams['action_noise'] = action_noise
    
#     return policy_kwargs, hyperparams