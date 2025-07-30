#################################################
# Author        : Philip Varghese Modayil
# Last Update   : 28-07-2025
# Topic         : Coupled Strip Lib Fucntions
# Description   : This file sets up the required functions for various calculation regarding the coupled microstrip
#################################################

#####################################################################################
#                                     Imports
#####################################################################################
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

######################################################################################
#                            Couple Strip Arrangement
######################################################################################
@dataclass(frozen=True)
class CoupledStripArrangement:
    V0: float # Potential of the sytem, used to scale the system which is defaulted at V0=1.0
    hw_arra: float # half width of the arrangement, parameter a
    ht_arra: float # height of the arrangement, parameter b
    ht_subs: float # height of the substrate, parameter h
    w_gap_strps: float # gap between the two microstrips, parameter s
    w_micrstr: float # width of the microstrip, parameter w
    ht_micrstr: float # height of the microstripm, parameter t
    er1: float # dielectric constatnt for medium 1
    er2: float # dielctric constant for medium 2
    num_fs: int # number of fourier series coefficients
    num_pts: int # number of points for the piece wise linear approaximation

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