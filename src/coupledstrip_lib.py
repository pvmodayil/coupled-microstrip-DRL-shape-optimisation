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
def calculate_potential_coeffs(V0: float,
                               hw_arra: float,
                               w_micrstr: float,
                               w_gap_strps: float,
                               num_fs: int,
                               g_left: NDArray[np.float64], 
                               x_left: NDArray[np.float64],
                               g_right: NDArray[np.float64], 
                               x_right: NDArray[np.float64],) -> tuple[float, NDArray[np.float64]]:
    # Dimensionality checks
    if np.size(x_left) != np.size(g_left):
        raise ValueError(f'Dimensions of x-axis vector and g-points vector for left side do not match!\n\
                         size x_left:{np.size(x_left)}, size g_left:{np.size(g_left)}')
    if np.size(x_right) != np.size(g_right):
        raise ValueError(f'Dimensions of x-axis vector and g-points vector for left side do not match!\n\
                         size x_right:{np.size(x_right)}, size g_right:{np.size(g_right)}')
        
    # fourier coefficients temp array
    n: NDArray[np.int64] = np.arange(num_fs)
    
    M: int = np.size(g_left)
    N: int = np.size(g_right)
    
    # Fourier coefficients
    ######################
    a0: float = (1/hw_arra)*(
        np.sum((g_left[1:M] + g_left[0:M-1])*(x_left[1:M] + x_left[0:M-1]))/2
        + w_micrstr
        + np.sum((g_right[1:N] + g_right[0:N-1])*(x_right[1:N] + x_right[0:N-1]))/2
        )*V0
    
    outer_coeff: NDArray = V0*2/(n*np.pi) # 1xn
    
    # convert the array to a column vector
    x_left_vec: NDArray[np.float64] = np.reshape(x_left,(-1,1))  # Mx1
    x_right_vec: NDArray[np.float64] = np.reshape(x_right, (-1,1)) # Nx1
    
    sin_left: NDArray[np.float64] = np.sin((n*np.pi/hw_arra)*x_left_vec[1:M]) - np.sin((n*np.pi/hw_arra)*x_left_vec[0:M-1]) # Mxn
    fac_left: NDArray[np.float64] = (g_left[1:M]-g_left[0:M-1])/(x_left[1:M]-x_left[0:M-1]) # 1xM
    an1: NDArray[np.float64] = np.matmul(fac_left,sin_left) # 1xn = 1xM x Mxn
    
    an2 = np.sin((w_micrstr + w_gap_strps/2)*n*np.pi/hw_arra) - np.sin(w_gap_strps/2*n*np.pi/hw_arra) # 1xn
    
    sin_right: NDArray[np.float64] = np.sin((n*np.pi/hw_arra)*x_right_vec[1:N]) - np.sin((n*np.pi/hw_arra)*x_right_vec[0:N-1]) # Nxn
    fac_right: NDArray[np.float64] = (g_right[1:N]-g_right[0:N-1])/(x_right[1:N]-x_right[0:N-1]) # 1xN
    an3: NDArray[np.float64] = np.matmul(fac_right,sin_right) # 1xn = 1xN x Nxn
    
    an: NDArray[np.float64] = outer_coeff*(an1+an2+an3) # 1xn
    
    return a0, an

def calculate_potential(hw_arra: float,
                        a0: float,
                        an: NDArray[np.float64],
                        x: NDArray[np.float64]) -> NDArray[np.float64]:
    num_fs: int = np.size(an)
    
    n: NDArray[np.int64] = np.arange(num_fs)[:, np.newaxis] # nx1
    cos: NDArray[np.float64] = np.cos((np.pi*n/hw_arra)*x) # nxm
    VF: NDArray[np.float64] = a0 + np.matmul(an,cos) # 1xn x nxm = 1xm
    
    return VF
######################################################################################
#                                        ENERGY
######################################################################################

######################################################################################
#                        Capacitance, Impedance, Epsilon Effective
######################################################################################

######################################################################################
#                           Surface Charge Density
######################################################################################