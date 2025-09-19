#################################################
# Author        : Philip Varghese Modayil
# Last Update   : 04-08-2025
# Topic         : Potential curve
# Description   : Plot the potential curve generated using bezier curves.
#################################################

#####################################################################################
#                                     Imports
#####################################################################################
import os
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt

def plot_potential(x_left: NDArray, g_left: NDArray, 
                   x_right: NDArray, g_right: NDArray,
                   image_dir: str,
                   name: str) -> None:
    image_path: str = os.path.join(image_dir,f'{name}_predicted_curve.png')
    plt.figure(figsize=(15,10))
    plt.plot(x_left*1000, g_left, linewidth=2, color = 'green',label='_nolabel')
    plt.plot(np.array([x_left[-1],x_right[0]])*1000, [1,1], linewidth=2, color = 'green',label='_nolabel') # simulate the microstrip region
    plt.plot(x_right*1000, g_right, linewidth=2, color = 'green',label='_nolabel')
   
    # plt.legend(loc='upper right')
    plt.ylabel('V(x,y=c) [Volt]', fontsize=25)
    plt.xlabel('x axis [mm]', fontsize=25)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.grid(True)
    plt.savefig(image_path)
    plt.close()
    
def plot_optimal_hw_arra(hw_arra_array: NDArray, 
                    energy_array: NDArray, 
                    image_dir: str,
                    name: str) -> None:
    
    idx: np.intp = np.argmin(energy_array)
    
    
    image_path: str = os.path.join(image_dir,f'{name}_optimal_hw_arra_variation.png')
    plt.figure(figsize=(15,10))
    plt.plot(hw_arra_array*1000, energy_array, linewidth=2, color = 'green',label='_nolabel')
    plt.scatter(hw_arra_array[idx]*1000, energy_array[idx], color = 'orange',label='_nolabel')
   
    # plt.legend(loc='upper right')
    plt.ylabel('Energy VAs', fontsize=25)
    plt.xlabel('x axis [mm]', fontsize=25)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.grid(True)
    plt.savefig(image_path)
    plt.close()