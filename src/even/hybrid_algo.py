#################################################
# Author        : Philip Varghese Modayil
# Last Update   : 13-08-2025
# Topic         : Hybrid RL-GA Algorithm
# Description   : The hw_arra parameter is chnaged within a range to find minimum energy configuration.
#                 This configuration is then used for GA optimisation.
#################################################

#####################################################################################
#                                     Imports
#####################################################################################
import os

import pandas as pd

import numpy as np
from numpy.typing import NDArray

import coupledstrip_lib as csa_lib
from coupledstrip_lib import CoupledStripArrangement
from coupledstrip_env import CoupledStripEnv

from utils import plot_curve as plot_curve

from stable_baselines3 import SAC

import logging

os.add_dll_directory(r'C:\mingw64\bin')
from ga_lib import ga_cpp #type: ignore
from _types import GAOptResult

# Set OpenMP to max threads before using parallel functions
ga_cpp.set_omp_to_max()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger: logging.Logger = logging.getLogger(__name__)

#####################################################################################
#                                     Functions
#####################################################################################
def create_directories(**kwargs) -> None:
    """
    takes in n number of directory paths and creates directories
    """
    for dir_name in kwargs.values():
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            # logger.info(f"Created directory: {dir_name}")
        else:
            pass
            # logger.info(f"Directory already exists: {dir_name}")

def predict(env: CoupledStripEnv, model: SAC) -> NDArray[np.float64]:
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

def possible_range(lower_bound: float, upper_bound: float) -> NDArray:
    # Available space
    gap: float = upper_bound - lower_bound
    
    # Step size is a small fraction of the available space
    step: float = 0.1*gap
    
    possible_range_array: NDArray = np.arange(lower_bound+step, upper_bound, step)
    
    return possible_range_array

def evaluate(env: CoupledStripEnv, model: SAC, new_hw_arra: float) -> float:
    env.CSA.hw_arra = new_hw_arra
    action: NDArray = predict(env=env, model=model)
    mid_point: int = env.action_space.shape[0]//2 + 1 

    x_left,g_left,_control = env.get_bezier_curve(action=action[:mid_point],side='left')
    x_right,g_right,_control = env.get_bezier_curve(action=action[mid_point:],side='right')
    
    energy: float = env.calculate_energy(g_left=g_left,x_left=x_left,g_right=g_right,x_right=x_right)

    return energy

def optimal_hw_arra(env: CoupledStripEnv, model: SAC, image_dir: str, case: str) -> float:
    # Get the possible set of values for hw_arra
    init_lower: float = env.CSA.width_micrstr + env.CSA.space_bw_strps/2
    init_upper: float = env.CSA.hw_arra
    variation: float = 0.2*abs(init_lower - init_upper)
    init_possible_range_array: NDArray = possible_range(lower_bound=init_lower, upper_bound=init_upper + variation)
    
    vectorised_evaluateInit = np.vectorize(lambda new_hw_arra: evaluate(env,model,new_hw_arra))
    init_energy_array: NDArray = vectorised_evaluateInit(init_possible_range_array)
    init_optimal_hw_arra_val: float = init_possible_range_array[np.argmin(init_energy_array)]
    
    data = {
        "a_vals": init_possible_range_array,
        "energy": init_energy_array
    }
    
    df_first = pd.DataFrame(data)
    plot_curve.plot_optimal_hw_arra(hw_arra_array=init_possible_range_array,energy_array=init_energy_array,image_dir=image_dir,name=f"{case}_Init")
    
    # Second run with tighter range
    variation: float = 0.3*abs(init_lower - init_optimal_hw_arra_val)
    final_lower: float = init_optimal_hw_arra_val - variation
    final_upper: float = init_optimal_hw_arra_val + variation
    final_possible_range_array: NDArray = possible_range(lower_bound=final_lower, upper_bound=final_upper)
    
    vectorised_evaluateFinal = np.vectorize(lambda new_hw_arra: evaluate(env,model,new_hw_arra))
    final_energy_array: NDArray = vectorised_evaluateFinal(final_possible_range_array)
    final_idx = np.argmin(final_energy_array)
    last_idx = np.size(final_energy_array)
    optimal_hw_arra_val: float = final_possible_range_array[final_idx]
    
    data = {
        "a_vals": final_possible_range_array,
        "energy": final_energy_array
    }
    df_second = pd.DataFrame(data)
    combined_df = pd.concat([df_first.add_suffix('_first'), df_second.add_suffix('_second')], axis=1)
    plot_curve.plot_optimal_hw_arra(hw_arra_array=final_possible_range_array,energy_array=final_energy_array,image_dir=image_dir,name=f"{case}_Final")
    
    # Edge case when the result is not satisfactory
    if final_idx == 0 or final_idx == last_idx:
        variation: float = 0.4*abs(init_lower - optimal_hw_arra_val)
        last_lower: float = optimal_hw_arra_val - variation
        last_upper: float = optimal_hw_arra_val + variation
        last_possible_range_array: NDArray = possible_range(lower_bound=last_lower, upper_bound=last_upper)
        
        vectorised_evaluateLast = np.vectorize(lambda new_hw_arra: evaluate(env,model,new_hw_arra))
        last_energy_array: NDArray = vectorised_evaluateLast(last_possible_range_array)
        last_idx = np.argmin(final_energy_array)
        optimal_hw_arra_val: float = final_possible_range_array[last_idx]
        
        data = {
        "a_vals": final_possible_range_array,
        "energy": final_energy_array
        }
        df_third = pd.DataFrame(data)
        combined_df = pd.concat([combined_df, df_third.add_suffix('_third')], axis=1)
        plot_curve.plot_optimal_hw_arra(hw_arra_array=last_possible_range_array,energy_array=last_energy_array,image_dir=image_dir,name=f"{case}_Last")
    
    combined_df.to_excel(os.path.join(image_dir,f"{case}_optimal_hw_arra_variation.xlsx"))
    return optimal_hw_arra_val
    
    
def hybrid_algorithm(env: CoupledStripEnv, model: SAC, image_dir: str, case: str) -> tuple[float,float]:
    env.CSA.hw_arra = optimal_hw_arra(env=env, model=model, image_dir=image_dir, case=case)
    
    action: NDArray = predict(env=env, model=model)
    mid_point: int = env.action_space.shape[0]//2 + 1

    x_left,g_left,_control = env.get_bezier_curve(action=action[:mid_point],side='left')
    x_right,g_right,_control = env.get_bezier_curve(action=action[mid_point:],side='right')
    
    rl_energy: float = env.calculate_energy(g_left=g_left,x_left=x_left,g_right=g_right,x_right=x_right)
    
    # Create a dictionary to hold the data
    data: dict[str, NDArray] = {
        'x_left': x_left,
        'g_left': g_left,
        'x_right': x_right,
        'g_right': g_right
    }

    pd.DataFrame(data).to_excel(os.path.join(image_dir,f'{case}_predicted_curve.xlsx'), index=False)
    pd.DataFrame(data).to_csv(os.path.join(image_dir,f'{case}_predicted_curve.csv'), index=False)
    plot_curve.plot_potential(x_left=x_left,g_left=g_left,x_right=x_right,g_right=g_right,image_dir=image_dir,name=case)
    
    # Call GA here
    logger.info("GA Optimization Started\n")
    num_fs: int = 1000
    noise_scale: float = 0.1
    population_size: int = 100
    num_generations: int = 300
    result: GAOptResult = ga_cpp.ga_optimize(env.CSA.V0,
                                    env.CSA.space_bw_strps,
                                    env.CSA.width_micrstr,
                                    env.CSA.ht_micrstr,
                                    env.CSA.hw_arra,
                                    env.CSA.ht_arra,
                                    env.CSA.ht_subs,
                                    env.CSA.er1,
                                    env.CSA.er2,
                                    num_fs,
                                    noise_scale,
                                    population_size,
                                    num_generations,
                                    x_left,g_left,
                                    x_right,g_right)
    
    ga_energy: float = result["best_energy"]
    data_opt: dict[str, NDArray] = {
        'x_left': x_left,
        'g_left': result["best_curve_left"],
        'x_right': x_right,
        'g_right': result["best_curve_right"]
    }
    
    convergence_data: dict[str, NDArray] = {
        'generation': np.arange(num_generations+1),
        'energy': result["energy_convergence"]
    }
    pd.DataFrame(data_opt).to_excel(os.path.join(image_dir,f'{case}_optimized_curve.xlsx'), index=False)
    pd.DataFrame(data_opt).to_csv(os.path.join(image_dir,f'{case}_optimized_curve.csv'), index=False)
    plot_curve.plot_potential(x_left=x_left,g_left=g_left,x_right=x_right,g_right=g_right,image_dir=image_dir,name=case+"_GA")
    pd.DataFrame(convergence_data).to_excel(os.path.join(image_dir,f'{case}_convergence_curve.xlsx'), index=False)

    return rl_energy, ga_energy

def evaluate_metrics(V0, energyD, energyL) -> dict[str,float]:
    cD: float = csa_lib.calculate_capacitance(V0=V0, W=energyD)
    cL: float = csa_lib.calculate_capacitance(V0=V0, W=energyL)
    
    zD: float = csa_lib.calculate_impedance(cD=cD,cL=cL,env="caseD")
    zL: float = csa_lib.calculate_impedance(cD=cD,cL=cL,env="caseL")
    
    epsEff: float = csa_lib.calculate_epsilonEff(cD=cD,cL=cL)
    
    data: dict[str, float] = {
        "wD": energyD,
        "wL": energyL,
        "cD": cD,
        "cL": cL,
        "zD": zD,
        "zL": zL,
        "epsEff": epsEff        
    }
    
    return data
def run(CSA: CoupledStripArrangement, model_path: str, ID: str) -> tuple[float, float]:
    # original hw_arra
    original_hw_arra: float = CSA.hw_arra
    cwd: str = os.getcwd()  
    test_dir: str = os.path.join(cwd,"test",CSA.mode,ID) # training/mode/env_type/images
    create_directories(test_dir=test_dir)
    # Model load
    model: SAC = SAC.load(model_path)
    
    # Case D
    envD: CoupledStripEnv = CoupledStripEnv(CSA=CSA)
    # Run hybrid RL-GA
    rl_energyD, ga_energyD = hybrid_algorithm(env=envD, model=model, image_dir=test_dir, case="CaseD")
    
    # Case L
    CSA.hw_arra = original_hw_arra
    CSA.er2 = 1.0
    envL: CoupledStripEnv = CoupledStripEnv(CSA=CSA)
    # Run hybrid RL-GA
    rl_energyL, ga_energyL = hybrid_algorithm(env=envL, model=model, image_dir=test_dir, case="CaseL")
    
    data_rl: dict[str, float] = evaluate_metrics(V0=CSA.V0, energyD=rl_energyD,energyL=rl_energyL)
    data_ga: dict[str, float] = evaluate_metrics(V0=CSA.V0, energyD=ga_energyD,energyL=ga_energyL) # Change to ga when it is implemented
    
    df_rl = pd.DataFrame([data_rl])
    df_rl["key"] = "rl"

    df_ga = pd.DataFrame([data_ga])
    df_ga["key"] = "ga"

    # Concatenate vertically
    df: pd.DataFrame = pd.concat([df_rl, df_ga], ignore_index=True)
    
    df.to_excel(os.path.join(test_dir,"hybrid_algo_metric.xlsx"))
    
    return data_ga["zD"], data_ga["zL"]
        
def main(CSA: CoupledStripArrangement, model_path: str) -> None:
    zD, zL = run(CSA=CSA, model_path=model_path,ID="TC-2")
    logger.info(f"The impedances are ZD: {zD} Ohm, ZL: {zL} Ohm")
    # df_test = pd.read_csv("./test/s-h_testcase.csv")
    # list_zD: list[float] = []
    # list_zL: list[float] = []
    # for index, row in df_test.iterrows():
    #     logger.info(f"ID: s/h = {row['s/h']}")
    #     CSA.er2 = 4.5
    #     CSA.space_bw_strps = row['s']*1E-6
    #     CSA.hw_arra = 3E-3 + CSA.space_bw_strps
    #     zD, zL = run(CSA=CSA, model_path=model_path,ID="s-h_"+str(row["s/h"]))
    #     list_zD.append(zD)
    #     list_zL.append(zL)
    
    # df_test["zD"] = list_zD
    # df_test["zL"] = list_zL
    
    # df_test.to_excel(os.path.join(os.getcwd(),"test",CSA.mode,"s-h_test_result.xlsx"))


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
        num_pts=30, # number of points for the piece wise linear approaximation
        mode="Even"
    )
    
    model_path = os.path.join("training","Even","hw_arra3","models","SAC_CSA_EVEN.zip")
    main(CSA=CSA,model_path=model_path)