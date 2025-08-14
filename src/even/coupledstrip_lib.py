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
from typing import Literal

from numba import njit
import numpy as np
from numpy.typing import NDArray

######################################################################################
#                            Couple Strip Arrangement
######################################################################################
@dataclass(frozen=False)
class CoupledStripArrangement:
    V0: float # Potential of the sytem, used to scale the system which is defaulted at V0=1.0
    hw_arra: float # half width of the arrangement, parameter a
    ht_arra: float # height of the arrangement, parameter b
    ht_subs: float # height of the substrate, parameter h
    space_bw_strps: float # gap between the two microstrips, parameter s
    width_micrstr: float # width of the microstrip, parameter w
    ht_micrstr: float # height of the microstripm, parameter t
    er1: float # dielectric constatnt for medium 1
    er2: float # dielctric constant for medium 2
    num_fs: int # number of fourier series coefficients
    num_pts: int # number of points for the piece wise linear approaximation
    mode: str # Even or Odd mode

######################################################################################
#                            Necessary Conditions Check
######################################################################################
@njit
def is_monotone(g: NDArray[np.float64], type: Literal["decreasing","increasing"]) -> bool:
    """
    Function to check for monotonicity, type  determines whether monotone decreasing or increasing

    Parameters
    ----------
    g : NDArray[np.float64]
        g points   for Piece Wise Linear function approximation
    type : Literal["decreasing","increasing"]
        flag to check for increasing or decreasing

    Returns
    -------
    bool
        returns True if monotone
    """
    dx: NDArray[np.float64] = np.diff(np.ascontiguousarray(g))
    if type == "decreasing":
        return bool(np.all(dx < 0))
    return bool(np.all(dx > 0)) 

@njit
def degree_monotonicity(g: NDArray[np.float64], type: Literal["decreasing","increasing"]) -> int:
    """
    Function to calculate the degree of monotonicity for the curve
    How many points follow the constraint

    Parameters
    ----------
    g : NDArray[np.float64]
        g points   for Piece Wise Linear function approximation
    type : Literal[&quot;decreasing&quot;,&quot;increasing&quot;]
        flag to check for increasing or decreasing

    Returns
    -------
    int
        number of points that follow the constraint
    """
    dx: NDArray[np.float64] = np.diff(np.ascontiguousarray(g))
    if type == "decreasing":
        return int(np.sum(dx < 0))
    return int(np.sum(dx > 0))

@njit   
def is_convex(g: NDArray[np.float64]) -> bool:
    """
    Function to check for convexity of the Piece Wise Linear function


    Parameters
    ----------
    g : NDArray[np.float64]
        g points   for Piece Wise Linear function approximation

    Returns
    -------
    bool
        returns True if convex

    Raises
    ------
    ValueError
        raise error  for insufficient number of g points
    """
    # Check if the array has at least 3 elements
    if len(g) < 3:
        raise ValueError("The array should have at least 3 elements to perform convexity check.\nPlease use more g-points")
    
    # Calculate the second differences
    g_contig: NDArray = np.ascontiguousarray(g)
    dx2 = g_contig[2:] - 2 * g_contig[1:-1] + g_contig[:-2]
    
    # Check if all second differences are non-negative
    return bool(np.all(dx2 >= 0))

@njit
def degree_convexity(g: NDArray[np.float64]) -> int:
    """
    Function to calculate the degree of convexity

    Parameters
    ----------
    g : g points   for Piece Wise Linear function approximation

    Returns
    -------
    int
        number of points that follow the constraint
    """
    # Calculate the second differences
    g_contig: NDArray = np.ascontiguousarray(g)
    dx2 = g_contig[2:] - 2 * g_contig[1:-1] + g_contig[:-2]
    
    # count number of positive values
    return int(np.sum(dx2>0))
######################################################################################
#                              Potential & Potential Coeffs
######################################################################################
@njit
def calculate_potential_coeffs(V0: float,
                               hw_arra: float,
                               width_micrstr: float,
                               space_bw_strps: float,
                               num_fs: int,
                               g_left: NDArray[np.float64], 
                               x_left: NDArray[np.float64],
                               g_right: NDArray[np.float64], 
                               x_right: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Function to calculate the Fourier potential coefficients for the Fourier series approximation

    Parameters
    ----------
    V0 : float
        potential at the microstrip, used to scale the equations
    hw_arra : float
        half width of the arrangement
    width_micrstr : float
        width of the microstrip
    space_bw_strps : float
        gap between the microstrip pair
    num_fs : int
        number of Fourier coefficients
    g_left : NDArray[np.float64]
        PWL g points for 0 <= x <= space_bw_strps/2
    x_left : NDArray[np.float64]
        x coordinates for g_left
    g_right : NDArray[np.float64]
        PWL g points for space_bw_strps/2 + width_micrstr <= x <= hw_arra
    x_right : NDArray[np.float64]
        x coordinates for g_right

    Returns
    -------
    NDArray[np.float64]
        Fourier coefficients

    Raises
    ------
    ValueError
        raises error if the dimensions of the left coordinates don't match
    ValueError
        raises error if the dimensions of the right coordinates don't match
    """
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
    d: float = space_bw_strps/2
    
    n: NDArray[np.int64] = np.arange(0,num_fs) # 1xn
    
    beta: NDArray[np.float64] = ((2*n+1)*np.pi/(2*hw_arra)).astype(np.float64) # 1xn
    m: NDArray[np.float64] = (g_left[1:M] - g_left[0:M-1])/(x_left[1:M] - x_left[0:M-1]) # 1xM-1
    m_prime: NDArray[np.float64] = (g_right[1:N] - g_right[0:N-1])/(x_right[1:N] - x_right[0:N-1]) # 1xN-1
    
    x_left_vec: NDArray[np.float64] = np.reshape(np.ascontiguousarray(x_left),(-1,1)) # Mx1
    x_right_vec: NDArray[np.float64] = np.reshape(np.ascontiguousarray(x_right),(-1,1)) # Nx1
    
    outer_coeff: float = 2*V0/hw_arra
    
    # vn1
    ######
    vn1: NDArray[np.float64] = (1/beta**2)*(
        np.dot(np.ascontiguousarray(m), np.ascontiguousarray(np.cos(beta*x_left_vec[1:M]) - np.cos(beta*x_left_vec[0:M-1])))
    ) # 1xn x [1xM-1 x M-1xn] = 1xn
    
    # vn2
    ######
    vn2: NDArray[np.float64] = (1/beta)*(
        np.dot(np.ascontiguousarray(g_left[1:M]), np.ascontiguousarray(np.sin(beta*x_left_vec[1:M])))
        - np.dot(np.ascontiguousarray(g_left[0:M-1]), np.ascontiguousarray(np.sin(beta*x_left_vec[0:M-1])))
    ) # 1xn x [1xM-1 x M-1xn] = 1xn
    
    # vn3
    ######
    vn3: NDArray[np.float64] = (1/beta)*(np.sin(beta*(d+width_micrstr)) - np.sin(beta*d))
    
    # vn4
    ######
    vn4: NDArray[np.float64] = (1/beta**2)*(
        np.dot(np.ascontiguousarray(m_prime), np.ascontiguousarray(np.cos(beta*x_right_vec[1:N]) - np.cos(beta*x_right_vec[0:N-1])))
    ) # 1xn x [1xN-1 x N-1xn] = 1xn
    
    # vn5
    ######
    vn5: NDArray[np.float64] = (1/beta)*(
        np.dot(np.ascontiguousarray(g_right[1:N]), np.ascontiguousarray(np.sin(beta*x_right_vec[1:N])))
        - np.dot(np.ascontiguousarray(g_right[0:N-1]), np.ascontiguousarray(np.sin(beta*x_right_vec[0:N-1])))
    ) # 1xn x [1xN-1 x N-1xn] = 1xn
    
    # vn
    ######
    vn: NDArray[np.float64] = outer_coeff*(vn1+vn2+vn3+vn4+vn5)
    
    return vn.astype(np.float64)

@njit
def calculate_potential(hw_arra: float,
                        vn: NDArray[np.float64],
                        x: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Function to calculate the potential with Fourier reconstruction

    Parameters
    ----------
    hw_arra : float
        half width of the arrangement
    vn : NDArray[np.float64]
        Fourier coefficients
    x : NDArray[np.float64]
        x axis coordinates

    Returns
    -------
    NDArray[np.float64]
        Potential Fourier reconstruction for the given x
    """
    num_fs: int = np.size(vn)
    
    n: NDArray[np.int64] = np.ascontiguousarray(np.arange(0,num_fs))[:, np.newaxis] # nx1
    beta: NDArray[np.float64] = ((2*n+1)*np.pi/(2*hw_arra)).astype(np.float64) # nx1
    cos: NDArray[np.float64] = np.cos(beta*x) # nxm
    VF: NDArray[np.float64] = np.dot(np.ascontiguousarray(vn),np.ascontiguousarray(cos)) # 1xn x nxm = 1xm
    
    return VF.astype(np.float64)
######################################################################################
#                                        ENERGY
######################################################################################
# define logarithmic implementations for cosh and sinh to improve numerical stability
# sinh = (e^x - e^(-x))/2 => ln(sinh(x)) = ln(e^x - e^(-x)) - ln(2)
# cosh = (e^x + e^(-x))/2 => ln(cosh(x)) = ln(e^x + e^(-x)) - ln(2)
@njit
def logsinh(vector: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Function to calculate the logsinh with numerical stability via thresholding

    Parameters
    ----------
    vector : NDArray[np.float64]
        theta in sinh(theta)

    Returns
    -------
    NDArray[np.float64]
        log(sinh)
    """
    absolute_vector: NDArray[np.float64] = np.abs(vector)

    logsinh_result: NDArray[np.float64] = np.where(absolute_vector > 33.0,
                                                   absolute_vector - np.log(2),
                                                   np.log(np.exp(vector) - np.exp(-vector)) - np.log(2))
    return logsinh_result

@njit
def logcosh(vector: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Function to calculate the logcosh with numerical stability via thresholding

    Parameters
    ----------
    vector : NDArray[np.float64]
        theta in cosh(theta)

    Returns
    -------
    NDArray[np.float64]
        log(cosh)
    """
    absolute_vector: NDArray[np.float64] = np.abs(vector)

    logcosh_result: NDArray[np.float64] = np.where(absolute_vector > 33.0,
                                                   absolute_vector - np.log(2),
                                                   np.log(np.exp(vector) + np.exp(-vector)) - np.log(2))
    return logcosh_result

@njit
def calculate_energy(er1: float, er2: float, hw_arra: float, ht_arra: float, ht_subs: float, vn: np.ndarray) -> float:
    """
    Function to calculate the energy of coupled strip system odd mode

    Parameters
    ----------
    er1 : float
        relative permitivity of medium 1
    er2 : float
        relative permitivity of medium 2
    hw_arra : float
        half width of the arrangement
    ht_arra : float
        height of the arrangement
    ht_subs : float
        height of the substrate
    vn : np.ndarray
        Fourier coefficients

    Returns
    -------
    float
        energy of the system
    """
    # Constant terms
    ################
    e0: float = 8.854187817E-12
    e1: float = er1*e0 # 8.54E-12 is permittivity constant e0
    e2: float = er2*e0
    
    num_fs: int = np.size(vn)
    n: NDArray[np.int64] = np.arange(1,num_fs+1)
    
    # Energy Formula Odd-Mode
    #########################
    coeff = (n*np.pi/4)*vn**2 # 1 x n
    
    # w1
    #####
    theta1: NDArray[np.float64] = (n*np.pi*(ht_arra-ht_subs)/hw_arra).astype(np.float64) # 1 x n
    coth1: NDArray[np.float64] = np.exp(logcosh(theta1)-logsinh(theta1)) # log(cosh/sinh) = log(cosh) - log(sinh)
    w1: NDArray[np.float64] = e1*coth1 # 1xn

    # w2
    #####
    theta2: NDArray[np.float64] = (n*np.pi*ht_subs/hw_arra).astype(np.float64) # 1 x n
    coth2: NDArray[np.float64] = np.exp(logcosh(theta2)-logsinh(theta2)) # log(cosh/sinh) = log(cosh) - log(sinh)
    w2: NDArray[np.float64] = e2*coth2 # 1xn


    W: float = np.sum(coeff*(w1+w2))
    
    return W
    
######################################################################################
#                        Capacitance, Impedance, Epsilon Effective
######################################################################################
@njit
def calculate_capacitance(V0: float, W:float) -> float:
    """
    Fucntion to calculate capacitance of the system

    Parameters
    ----------
    V0 : float
        potential at the microstrip
    W : float
        energy of the system

    Returns
    -------
    float
        capacitance of the system
    """
    
    C: float = (2*W)/(V0**2)
    
    return C

@njit
def calculate_impedance(cD: float = 1.0, cL: float = 1.0, env: Literal["caseD", "caseL"] = "caseD") -> float:
    """
    Function to calculate the impedance of the system based on the environment case

    Parameters
    ----------
    cD : float
        capacitance of the caseD system, by default 1.0 (arbitrary no significance)
    cL : float
        capacitance of the caseL system, by default 1.0 (arbitrary no significance)
    env : Literal["caseD", "caseL"]
        case of the environment (caseD/caseL), by default "caseD"

    Returns
    -------
    float
        _description_
    """
    e0: float = 8.854187817E-12
    match env:
        case "caseD":
            return 376.62*e0/((cD*cL)**0.5)
        case "caseL":
            return 376.62*e0/cL
        case _:
            return 0.0 # must not happen as there is a default state(just for the type checker to pass)

@njit
def calculate_epsilonEff(cD: float, cL: float) -> float:
    """
    Function to calculate effective dielectric constant

    Parameters
    ----------
    cD : float
        case D capacitance
    cL : float
        case L capacitance

    Returns
    -------
    eps_eff: float
        effective dielectric constant
    """
    return cD/cL
######################################################################################
#                           Surface Charge Density
######################################################################################