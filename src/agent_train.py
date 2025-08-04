#################################################
# Author        : Philip Varghese Modayil
# Last Update   : 18-06-2025
# Topic         : RL Agent Training Script
# Description   : This script sets up the training environment for a reinforcement learning agent.
#################################################

#####################################################################################
#                                     Imports
#####################################################################################
import os
import time
import pandas as pd

import numpy as np
from numpy.typing import NDArray

import torch

from .coupledstrip_lib import CoupledStripArrangement
from .coupledstrip_env import CoupledStripEnv
from ._hyper_parameter import get_hyper_params

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.base_class import BaseAlgorithm


import logging
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger: logging.Logger = logging.getLogger(__name__)

#####################################################################################
#                                     Functions
#####################################################################################
# Create directories
####################
def create_directories(**kwargs) -> None:
    """
    takes in n number of directory paths and creates directories
    """
    for dir_name in kwargs.values():
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            logger.info(f"Created directory: {dir_name}")
        else:
            print(f"Directory already exists: {dir_name}")



# Train, Load and Test
######################
def predict(env: CoupledStripEnv, model: SAC | BaseAlgorithm) -> NDArray[np.float64]:
    """
    Function to predict and take absoulte of the action

    Parameters
    ----------
    env : CoupledStripEnv
        environment
    model : BaseAlgorithm
        RL model

    Returns
    -------
    NDArray[np.float64]
        action array for bezier curve
    """
    obs_space: NDArray[np.float32]
    info: dict
    
    obs_space, info = env.reset()
    action: NDArray
    _states: tuple | None
    
    action, _states = model.predict(obs_space)
    
    return np.abs(action)
class IntermediatePredictionCallback(BaseCallback):
    def __init__(self, env: CoupledStripEnv, intermediate_pred_interval: int, intermediate_pred_dir: str) -> None:
        super(IntermediatePredictionCallback, self).__init__()
        self.env: CoupledStripEnv = env
        self.intermediate_pred_interval: int = intermediate_pred_interval
        self.entropy_coefficient_set = False
        
        # Action DF
        self.intermediate_pred_dir: str = intermediate_pred_dir
        self.action_df_path: str = os.path.join(self.intermediate_pred_dir,'action_dataframe.csv')
        self.action_df_rows: list[pd.DataFrame]= []
        self.column_names: list[str] = ['Timestep'] + [f'A{i}' for i in range(env.action_space.shape[0])]
        
    
    def _on_step(self) -> bool:
        if (self.num_timesteps == 1) or (self.num_timesteps % self.intermediate_pred_interval == 0):
            # initial prediction
            logger.info(f"Intermediate prediction at timestep {self.num_timesteps}")
            action: NDArray = predict(self.env,self.model)

            row: NDArray = np.insert(action,self.num_timesteps,0)
            row_df = pd.DataFrame([row], columns=self.column_names)
            self.action_df_rows.append(row_df)
            
            result_df: pd.DataFrame = pd.concat(self.action_df_rows, ignore_index=True)

            result_df.to_csv(self.action_df_path,index=False)
            logger.info("Resuming training......")
        
        return True
    
def train(env: CoupledStripEnv, 
          model_dir: str, 
          log_dir: str, 
          intermediate_pred_dir: str,
          device: torch.device,
          timesteps: int,
          intermediate_pred_interval: int) -> str:
    
    # Get the hyperparameters
    policy_kwargs: dict
    hyperparams: dict
    policy_kwargs, hyperparams = get_hyper_params()
    
    # Initialize the RL agent
    model = SAC("MlpPolicy", 
                env,  
                verbose=0, 
                policy_kwargs=policy_kwargs,
                tensorboard_log=log_dir,
                device=device,
                **hyperparams)  
    
    # Train the agent
    logger.info("Training started......")
    start_time: float = time.time()
    
    model.learn(total_timesteps=timesteps,
        log_interval=4,
        reset_num_timesteps=True, 
        tb_log_name="CSA_ODD",
        progress_bar=True,
        callback=IntermediatePredictionCallback(env=env,
                                                intermediate_pred_interval=intermediate_pred_interval,
                                                intermediate_pred_dir=intermediate_pred_dir))
    
    training_time: float = (time.time() - start_time)/60 # in minutes
    logger.info(f"Training ended with total training time: {training_time}......")
    
    # Save the trained model
    model_save_path: str = os.path.join(model_dir, "SAC_CSA_ODD")
    model.save(model_save_path, include="all")
    
    logger.info(f"Training completed and model saved at {model_save_path}.")      
    
    return model_save_path

def test(model_path: str, env: CoupledStripEnv) -> None:
    model: SAC = SAC.load(model_path)
    action: NDArray = predict(env,model)
    
    mid_point: int = int(env.action_space.shape[0]/2)
    action_left: NDArray = action[:mid_point]
    action_right: NDArray = action[mid_point:]
    
    x_left: NDArray
    g_left: NDArray
    _control: NDArray
    x_left,g_left,_control = env.get_bezier_curve(action=action_left,side='left')
    x_right: NDArray
    g_right: NDArray
    x_right,g_right,_control = env.get_bezier_curve(action=action_right,side='right')
    
    energy: float = env.calculate_energy(g_left=g_left,
                                        x_left=x_left,
                                        g_right=g_right,
                                        x_right=x_right)
    logger.info(f"Final predicted energy for the system: {energy} VAs")
    
# main called function
######################      
def main(CSA: CoupledStripArrangement) -> None:
    # environment type
    env_type: str = "caseL" if CSA.er1 == 1.0 else "caseD"
    
    cwd: str = os.getcwd()  
    model_dir: str = os.path.join(cwd,"training",env_type,"models") # training/env_type/models
    log_dir: str = os.path.join(cwd,"training",env_type,"logs") # training/env_type/logs
    image_dir: str = os.path.join(cwd,"training",env_type,"images") # training/env_type/images
    intermediate_pred_dir: str = os.path.join(cwd,"training",env_type,"intermediate_prediction") # training/env_type/intermediate_prediction
    create_directories(mdirRoot=model_dir, 
                    ldirRoot=log_dir, 
                    idirRoot=image_dir, 
                    inter_pred_dir=intermediate_pred_dir)
    
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Train the model
    env: CoupledStripEnv = CoupledStripEnv(CSA=CSA)
    
    model_save_path: str = train(env=env, 
                                 model_dir=model_dir, 
                                 log_dir=log_dir, 
                                 intermediate_pred_dir=intermediate_pred_dir,
                                 device=device,
                                 timesteps=50000,
                                 intermediate_pred_interval=5000)
    
    test(model_path=model_save_path,env=env)

if __name__ == "__main__":
    CSA: CoupledStripArrangement = CoupledStripArrangement(
        V0=1., # Potential of the sytem, used to scale the system which is defaulted at V0=1.0
        hw_arra=1., # half width of the arrangement, parameter a
        ht_arra=1., # height of the arrangement, parameter b
        ht_subs=1., # height of the substrate, parameter h
        space_bw_strps=1., # gap between the two microstrips, parameter s
        width_micrstr=1., # width of the microstrip, parameter w
        ht_micrstr=1., # height of the microstripm, parameter t
        er1=1., # dielectric constatnt for medium 1
        er2=1., # dielctric constant for medium 2
        num_fs=2000, # number of fourier series coefficients
        num_pts=100 # number of points for the piece wise linear approaximation
    )
    main(CSA=CSA)