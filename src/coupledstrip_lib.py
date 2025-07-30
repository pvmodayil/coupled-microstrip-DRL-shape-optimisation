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
                               x_right: NDArray[np.float64]) -> NDArray[np.float64]:
    # Dimensionality check
    if np.size(x_left) != np.size(g_left):
        raise ValueError(f'Dimensions of x-axis vector and g-points vector for left side do not match!\n\
                         size x_left:{np.size(x_left)}, size g_right:{np.size(g_left)}')  
    if np.size(x_right) != np.size(g_right):
        raise ValueError(f'Dimensions of x-axis vector and g-points vector for right side do not match!\n\
                         size x_right:{np.size(x_right)}, size g_right:{np.size(g_right)}')    
        
    # Repeating or constant terms
    #############################
    M: float = np.size(g_left)
    N: float = np.size(g_right)
    d: float = w_gap_strps/2
    
    n: NDArray[np.int64] = np.arange(1,num_fs+1) # 1xn
    
    alpha: NDArray[np.float64] = ((n*np.pi)/hw_arra).astype(dtype=np.float64) # 1xn
    m: NDArray[np.float64] = (g_left[1:M] - g_left[0:M-1])/(x_left[1:M] - x_left[0:M-1]) # 1xM-1
    m_prime: NDArray[np.float64] = (g_right[1:N] - g_right[0:N-1])/(x_right[1:N] - x_right[0:N-1]) # 1xN-1
    
    x_left_vec: NDArray[np.float64] = np.reshape(x_left,(-1,1)) # Mx1
    x_right_vec: NDArray[np.float64] = np.reshape(x_right,(-1,1)) # Nx1
    
    outer_coeff: float = 2*V0/hw_arra
    
    # vn1
    ######
    vn1: NDArray[np.float64] = (1/alpha**2)*(
        np.matmul(m, np.sin(alpha*x_left_vec[1:M]) - np.sin(alpha*x_left_vec[0:M-1]))
    ) # 1xn x [1xM-1 x M-1xn] = 1xn
    
    # vn2
    ######
    vn2: NDArray[np.float64] = (1/alpha)*(
        np.matmul(g_left[0:M-1], np.cos(alpha*x_left_vec[0:M-1]))
        - np.matmul(g_left[1:M], np.cos(alpha*x_left_vec[1:M]))
    ) # 1xn x [1xM-1 x M-1xn] = 1xn
    
    # vn3
    ######
    vn3: NDArray[np.float64] = (1/alpha)*(np.cos(alpha*d) - np.cos(alpha*(d+w_micrstr)))
    
    # vn4
    ######
    vn4: NDArray[np.float64] = (1/alpha**2)*(
        np.matmul(m_prime, np.sin(alpha*x_right_vec[1:N]) - np.sin(alpha*x_right_vec[0:N-1]))
    ) # 1xn x [1xN-1 x N-1xn] = 1xn
    
    # vn5
    ######
    vn5: NDArray[np.float64] = (1/alpha)*(
        np.matmul(g_right[0:N-1], np.cos(alpha*x_right_vec[0:N-1]))
        - np.matmul(g_right[1:N], np.cos(alpha*x_right_vec[1:N]))
    ) # 1xn x [1xN-1 x N-1xn] = 1xn
    
    # vn
    ######
    vn: NDArray[np.float64] = outer_coeff*(vn1+vn2+vn3+vn4+vn5)
    
    return vn.astype(dtype=np.float64)

def calculate_potential(hw_arra: float,
                        vn: NDArray[np.float64],
                        x: NDArray[np.float64]) -> NDArray[np.float64]:
    num_fs: int = np.size(vn)
    
    n: NDArray[np.int64] = np.arange(1,num_fs+1)[:, np.newaxis] # nx1
    alpha: NDArray[np.float64] = (n*np.pi/hw_arra).astype(dtype=np.float64)
    sin: NDArray[np.float64] = np.sin(alpha*x) # nxm
    VF: NDArray[np.float64] = np.matmul(vn,sin) # 1xn x nxm = 1xm
    
    return VF.astype(dtype=np.float64)
######################################################################################
#                                        ENERGY
######################################################################################

######################################################################################
#                        Capacitance, Impedance, Epsilon Effective
######################################################################################

######################################################################################
#                           Surface Charge Density
######################################################################################