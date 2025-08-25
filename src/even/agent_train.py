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

from coupledstrip_lib import CoupledStripArrangement
from coupledstrip_env import CoupledStripEnv
from _hyper_parameter import get_hyper_params

from utils import plot_train_metrics as plot_metric
from utils import plot_curve as plot_curve

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
    
    action, _states = model.predict(obs_space, deterministic=True)
    
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
            # logger.info(f"Intermediate prediction at timestep {self.num_timesteps}")
            action: NDArray = predict(self.env,self.model)

            row: NDArray = np.insert(action,0,self.num_timesteps)
            row_df = pd.DataFrame([row], columns=self.column_names)
            self.action_df_rows.append(row_df)
            
            result_df: pd.DataFrame = pd.concat(self.action_df_rows, ignore_index=True)

            result_df.to_csv(self.action_df_path,index=False)
            # logger.info("Resuming training......")
        
        return True
    
def train(env: CoupledStripEnv, 
          model_dir: str, 
          log_dir: str, 
          intermediate_pred_dir: str,
          device: torch.device,
          timesteps: int,
          intermediate_pred_interval: int,
          tb_log_name: str) -> str:
    
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
        tb_log_name=tb_log_name,
        progress_bar=True,
        callback=IntermediatePredictionCallback(env=env,
                                                intermediate_pred_interval=intermediate_pred_interval,
                                                intermediate_pred_dir=intermediate_pred_dir))
    
    training_time: float = (time.time() - start_time)/60 # in minutes
    logger.info(f"Training ended with total training time: {training_time}......")
    
    # Save the trained model
    model_save_path: str = os.path.join(model_dir, "SAC_CSA_EVEN")
    model.save(model_save_path, include="all")
    
    logger.info(f"Training completed and model saved at {model_save_path}.")      
    
    return model_save_path

def test(model_path: str, env: CoupledStripEnv, image_dir: str) -> None:
    import coupledstrip_lib as csa_lib
    
    # Load model
    model: SAC = SAC.load(model_path)
    
    # Set Prediction environment and predict (CaseD as it is the default here)
    ########################################################################
    actionD: NDArray = predict(env,model)
    
    mid_point: int = env.action_space.shape[0]//2
    action_leftD: NDArray = actionD[:mid_point]
    action_rightD: NDArray = actionD[mid_point:]
    
    x_leftD: NDArray
    g_leftD: NDArray
    _control: NDArray
    x_leftD,g_leftD,_control = env.get_bezier_curve(action=action_leftD,side='left')
    x_rightD: NDArray
    g_rightD: NDArray
    x_rightD,g_rightD,_control = env.get_bezier_curve(action=action_rightD,side='right')
    
    # Create a dictionary to hold the data
    dataD: dict[str, NDArray] = {
        'x_left': x_leftD,
        'g_left': g_leftD,
        'x_right': x_rightD,
        'g_right': g_rightD
    }

    # Create DataFrame from the dictionary
    pd.DataFrame(dataD).to_excel(os.path.join(image_dir,'CaseD_predicted_curve.xlsx'), index=False)
    pd.DataFrame(dataD).to_csv(os.path.join(image_dir,'CaseD_predicted_curve.csv'), index=False)
    plot_curve.plot_potential(x_left=x_leftD,g_left=g_leftD,x_right=x_rightD,g_right=g_rightD,image_dir=image_dir,name="CaseD")
    
    # Calculate Metrics
    energyD: float = env.calculate_energy(g_left=g_leftD,
                                        x_left=x_leftD,
                                        g_right=g_rightD,
                                        x_right=x_rightD)
    logger.info(f"Final predicted energy for the system: {energyD} VAs")
    
    # Set Prediction environment and predict for CaseL
    ##################################################
    env.CSA.er2 = 1.0
    actionL: NDArray = predict(env,model)
    
    mid_point: int = env.action_space.shape[0]//2
    action_leftL: NDArray = actionL[:mid_point]
    action_rightL: NDArray = actionL[mid_point:]
    
    x_leftL: NDArray
    g_leftL: NDArray
    _control: NDArray
    x_leftL,g_leftL,_control = env.get_bezier_curve(action=action_leftL,side='left')
    x_rightL: NDArray
    g_rightL: NDArray
    x_rightL,g_rightL,_control = env.get_bezier_curve(action=action_rightL,side='right')
    
    # Create a dictionary to hold the data
    dataL: dict[str, NDArray] = {
        'x_left': x_leftL,
        'g_left': g_leftL,
        'x_right': x_rightL,
        'g_right': g_rightL
    }

    # Create DataFrame from the dictionary
    pd.DataFrame(dataL).to_excel(os.path.join(image_dir,'CaseL_predicted_curve.xlsx'), index=False)
    pd.DataFrame(dataL).to_csv(os.path.join(image_dir,'CaseL_predicted_curve.csv'), index=False)
    plot_curve.plot_potential(x_left=x_leftL,g_left=g_leftL,x_right=x_rightL,g_right=g_rightL,image_dir=image_dir,name="CaseL")
    
    # Calculate Metrics
    energyL: float = env.calculate_energy(g_left=g_leftL,
                                        x_left=x_leftL,
                                        g_right=g_rightL,
                                        x_right=x_rightL)
    logger.info(f"Final predicted energy for the system: {energyL} VAs")
    
    # Calculate Capacitance, Impedance and Epsilon Eff
    CD: float = csa_lib.calculate_capacitance(V0=env.CSA.V0,W=energyD) 
    CL: float = csa_lib.calculate_capacitance(V0=env.CSA.V0,W=energyL) 
    
    ZD: float = csa_lib.calculate_impedance(cD=CD,cL=CL,env="caseD")
    ZL: float = csa_lib.calculate_impedance(cD=CD,cL=CL,env="caseL")
    
    epsEff: float = csa_lib.calculate_epsilonEff(cD=CD,cL=CL)
    
    metrics_data: dict[str, list[str|float]] = {
        "metrics": ['WD','WL','CD','CL','ZD','ZL','epsEff'],
        "value": [energyD,energyL,CD,CL,ZD,ZL,epsEff]
    }
    pd.DataFrame(metrics_data).to_excel(os.path.join(image_dir,'prediction_metrics.xlsx'), index=False)
# main called function
######################      
def main(CSA: CoupledStripArrangement) -> None:
    # environment type
    env_type: str = "caseL" if CSA.er2 == 1.0 else "caseD"
    
    cwd: str = os.getcwd()  
    model_dir: str = os.path.join(cwd,"training",CSA.mode,env_type,"models") # training/mode/env_type/models
    log_dir: str = os.path.join(cwd,"training",CSA.mode,env_type,"logs") # training/mode/env_type/logs
    image_dir: str = os.path.join(cwd,"training",CSA.mode,env_type,"images") # training/mode/env_type/images
    intermediate_pred_dir: str = os.path.join(cwd,"training",CSA.mode,env_type,"intermediate_prediction") # training/mode/env_type/intermediate_prediction
    create_directories(mdirRoot=model_dir, 
                    ldirRoot=log_dir, 
                    idirRoot=image_dir, 
                    inter_pred_dir=intermediate_pred_dir)
    
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Train and test the model
    env: CoupledStripEnv = CoupledStripEnv(CSA=CSA)
    
    model_save_path: str = train(env=env, 
                                 model_dir=model_dir, 
                                 log_dir=log_dir, 
                                 intermediate_pred_dir=intermediate_pred_dir,
                                 device=device,
                                 timesteps=50000,
                                 intermediate_pred_interval=5000,
                                 tb_log_name="CSA_EVEN")
    
    test(model_path=model_save_path,env=env,image_dir=image_dir)

    # plot the training metrics
    subdir_log_dir: list[str] = os.listdir(log_dir) # generally only one folder exists
    
    for sub_dir in subdir_log_dir:
        subdir_path: str = os.path.join(log_dir, sub_dir)
        log_files: list[str] = os.listdir(subdir_path)
        
        for log_file in log_files:
            log_file_path: str = os.path.join(subdir_path, log_file)
            try:
                plot_metric.plot_rewards(image_dir=image_dir, log_file_path=log_file_path)
                plot_metric.plot_loss(image_dir=image_dir, log_file_path=log_file_path)
                plot_metric.plot_entropy(image_dir=image_dir, log_file_path=log_file_path)
            except Exception as e:
                print(e)

if __name__ == "__main__":
    CSA: CoupledStripArrangement = CoupledStripArrangement(
        V0=1., # Potential of the sytem, used to scale the system which is defaulted at V0=1.0
        hw_arra=3E-3, # half width of the arrangement, parameter a
        ht_arra=2.76E-3, # height of the arrangement, parameter b
        ht_subs=112E-6, # height of the substrate, parameter h
        space_bw_strps=200E-6, # gap between the two microstrips, parameter s
        width_micrstr=150E-6, # width of the microstrip, parameter w
        ht_micrstr=0, # height of the microstripm, parameter t
        er1=1.0, # dielectric constatnt for medium 1
        er2=4.5, # dielctric constant for medium 2
        num_fs=2000, # number of fourier series coefficients
        num_pts=50, # number of points for the piece wise linear approaximation
        mode="Even"
    )
    main(CSA=CSA)