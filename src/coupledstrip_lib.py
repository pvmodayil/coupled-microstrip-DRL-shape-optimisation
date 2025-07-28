#################################################
# Author        : Philip Varghese Modayil
# Last Update   : 28-07-2025
# Topic         : Coupled Strip Lib Fucntions
# Description   : This file sets up the required functions for various calculation regarding the coupled microstrip
#################################################

#####################################################################################
#                                     Imports
#####################################################################################
import numpy as np
from numpy.typing import NDArray

######################################################################################
#                            Necessary Conditions Check
######################################################################################
def is_monotone(g: NDArray[np.float64], decreasing: bool) -> bool:
    dx: NDArray[np.float64] = np.diff(g)
    if decreasing:
        return bool(np.all(dx < 0))
    return bool(np.all(dx > 0)) 
    
def is_convex(g: NDArray[np.float64]) -> bool:
    # Check if the array has at least 3 elements
    if len(g) < 3:
        raise ValueError("The array should have at least 3 elements to perform convexity check.\nPlease use more g-points")
    
    # Calculate the second differences
    dx2 = g[2:] - 2 * g[1:-1] + g[:-2]
    
    # Check if all second differences are non-negative
    # return the boolean value
    return bool(np.all(dx2 >= 0))

######################################################################################
#                              Potential & Potential Coeffs
######################################################################################

######################################################################################
#                                        ENERGY
######################################################################################

######################################################################################
#                        Capacitance, Impedance, Epsilon Effective
######################################################################################

######################################################################################
#                           Surface Charge Density
######################################################################################