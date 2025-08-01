policy_kwargs: dict[str, int | list[int]] = dict(log_std_init=-3, net_arch=[400, 300]) # setting nn architecture
hyperparams: dict[str, int|float|bool|str] = {
    'ent_coef': 'auto',  # setting entropy coefficient to auto boosts exploration
    'batch_size':256,  
    'buffer_size': 20000,        
    'learning_rate': 0.0007,               
    'gamma': 0.98,      
    'tau':0.02,
    'learning_starts':1000,
    'use_sde':True      
}

def get_hyper_params() -> tuple[dict[str, int | list[int]], dict[str, int|float|bool|str]]:
     return policy_kwargs, hyperparams