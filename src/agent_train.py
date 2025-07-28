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
import numpy as np
import matplotlib.pyplot as plt
import torch
from .coupledstrip_env import CoupledStripEnv
from ._hyper_parameter import get_hyper_params

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback


import logging
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger: logging.Logger = logging.getLogger(__name__)

#####################################################################################
#                                     Functions
#####################################################################################
# cretae directories
###############################################################
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
 
def train(env: CoupledStripEnv, model_dir: str, log_dir: str, device: torch.device) -> None:
    """
    Train the RL agent using the specified environment.
    
    Parameters
    ----------
    env : CoupledStripEnv
        The environment in which the agent will be trained.
    model_dir : str
        Directory to save the trained model.
    log_dir : str
        Directory to save training logs.
    """
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
    model.learn(total_timesteps=10000)
    
    # Save the trained model
    model.save(os.path.join(model_dir, "sac_coupled_strip"))
    
    logger.info("Training completed and model saved.")      
       
def main() -> None:
    cwd: str = os.getcwd()  
    model_dir: str = os.path.join(cwd,"training","models")
    log_dir: str = os.path.join(cwd,"training","logs")
    image_dir: str = os.path.join(cwd,"training","images")
    create_directories(mdirRoot = model_dir, ldirRoot = log_dir, idirRoot = image_dir)
    
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    main()