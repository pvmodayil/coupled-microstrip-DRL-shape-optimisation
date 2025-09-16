#################################################
# Author        : Philip Varghese Modayil
# Last Update   : 18-06-2025
# Topic         : Coupled Strip Environment
# Description  : This script sets up the gymnasium environment for coupled strip optimization.
#################################################

#####################################################################################
#                                     Imports
#####################################################################################
from typing import Optional, Literal

import numpy as np
from numpy.typing import NDArray

# import gym
from gymnasium import Env
from gymnasium.spaces import Box

# CSA Lib
import coupledstrip_lib as csa_lib
from coupledstrip_lib import CoupledStripArrangement

import logging
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger: logging.Logger = logging.getLogger(__name__)

#####################################################################################
#                               Gymnasium Environment
#####################################################################################
class CoupledStripEnv(Env):
    """
    # CoupledStripEnv
    Custom environment for coupled strip optimization.
    This environment simulates the dynamics of a coupled strip system.
    """
    
    def __init__(self, CSA: CoupledStripArrangement) -> None:
        
        # Environment paramaters
        self.CSA: CoupledStripArrangement = CSA
        
        # Calculate the baseline energy for scaling reward
        action_left: NDArray = np.zeros(4)
        action_right: NDArray = np.zeros(4)
        x_left: NDArray
        g_left: NDArray
        _control: NDArray
        x_left,g_left,_control = self.get_bezier_curve(action=action_left,side='left')
        x_right: NDArray
        g_right: NDArray
        _control: NDArray
        x_right,g_right,_control = self.get_bezier_curve(action=action_right,side='right')
        
        vn: NDArray = csa_lib.calculate_potential_coeffs(V0=self.CSA.V0,
                                                                 hw_arra=self.CSA.hw_arra,
                                                                 width_micrstr=self.CSA.width_micrstr,
                                                                 space_bw_strps=self.CSA.space_bw_strps,
                                                                 num_fs=self.CSA.num_fs,
                                                                 g_left=g_left,
                                                                 x_left=x_left,
                                                                 g_right=g_right,
                                                                 x_right=x_right)
        
        self.energy_baseline: float = csa_lib.calculate_energy(er1=self.CSA.er1,
                                                    er2=self.CSA.er2,
                                                    hw_arra=self.CSA.hw_arra,
                                                    ht_arra=self.CSA.ht_arra,
                                                    ht_subs=self.CSA.ht_subs,
                                                    vn=vn)
        
        self.minimum_energy: NDArray = np.array([self.energy_baseline])
        
        # Define action and observation space
        """
        Action Space
        -----------------------------------------
        The action space is an array of size four where two pairs of values specify the (x,y) coordinates of the two control points. 
            
        | Action          | Min               | Max                | Size       |   
        |-----------------|-------------------|--------------------|------------|
        | control fcator  | -bound            | bound              | ndarray(4,)|
        
        """
        bound: float = 1
        self.action_space: Box = Box(low=-bound, high=bound, shape=(8,), dtype=np.float32) #type:ignore
        self.action_space_bound: float = bound
        """
        Observation Space
        -----------------------------------------
        The observation space includes width_micrstr, hw_arra, ht_arra, ht_subs and er2
        
        The observation space is an `ndarray` with shape `(5,)` where the elements correspond to the following:
        
        | Num          | Observation           | Min               | Max                |
        |--------------|-----------------------|-------------------|--------------------|
        | 0            | width_micrstr         | 0                 | Inf                |
        | 1            | space_bw_strps        | 0                 | Inf                |
        | 2            | hw_arra               | 0                 | Inf                |
        | 3            | ht_arra               | 0                 | Inf                |
        | 4            | ht_subs               | 0                 | Inf                |
        | 5            | er2                   | 0                 | Inf                |
        """
        self.observation_space: Box = Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32) #type:ignore

    def _get_control_points(self,action:NDArray[np.float64],
                     side: Literal["left","right"]) -> NDArray[np.float64]:
        """
        _get_control_points
        
        Generate control points for a Bézier curve based on the specified side.
        The control points are determined by the side of the microstrip structure
        and the predefined width and spacing.

        Parameters
        ----------
        action : NDArray[np.float64]
            Action array containing parameters for the Bézier curve.
            [x-axis scale doe P1, y-coordiate of P1, x-axis scale for P2, y-coordinate of P2]
        side : Literal["left","right"]
            The side of the microstructure ('left' or 'right').

        Returns
        -------
        NDArray[np.float64]
            Control points for the Bézier curve.
        """
        
        if side == 'left':
            x_end_left: float = self.CSA.space_bw_strps/2
            P0: NDArray[np.float64] = np.array([0, 0])
            
            P3: NDArray[np.float64] = np.array([x_end_left, 1])
            
            P2X: float = x_end_left - action[0]*(x_end_left-0)
            P2Y: float = 1 - action[1]*(1-0)
            P2: NDArray[np.float64] = np.array([P2X, P2Y])
            
            P1X: float = P2X - action[2]*(P2X-0)
            P1Y: float = P2Y - action[3]*(P2Y-0)
            P1: NDArray[np.float64] = np.array([P1X, P1Y])
            
        elif side == 'right':
            x_start_right: float = self.CSA.space_bw_strps/2 + self.CSA.width_micrstr
            P0: NDArray[np.float64] = np.array([x_start_right, 1])
            
            P1X: float = x_start_right + action[0]*(self.CSA.hw_arra-x_start_right)
            P1Y: float = action[1]
            P1: NDArray[np.float64] = np.array([P1X, P1Y])
           
            P2X: float = P1X + action[2]*(self.CSA.hw_arra-P1X)
            P2Y: float = P1Y - action[3]*(P1Y - 0)
            P2: NDArray[np.float64] = np.array([P2X, P2Y])
            
            P3: NDArray[np.float64] = np.array([self.CSA.hw_arra, 0])
            
        else:
            raise ValueError("Invalid side specified. Use 'left' or 'right'.")
        
        return np.array([P0, P1, P2, P3])  # Return control points as a numpy array

    def get_bezier_curve(self, action: NDArray[np.float64], 
                    side: Literal["left","right"]) -> tuple[NDArray[np.float64],NDArray[np.float64],NDArray[np.float64]]:
        """
        get_bezier_curve 
        
        Generate a Bézier curve based on the provided action array.
        Bézier curves are defined by control points, and this function computes
        the curve points using a cubic Bézier formula. 
        The formula: B(t) = (1-t)³P₀ + 3(1-t)²tP₁ + 3(1-t)t²P₂ + t³P₃ 

        Parameters
        ----------
        action : NDArray[np.float64]
            action array given by agent
        side : Literal["left","right"]
            The side of the microstructure ('left' or 'right').

        Returns
        -------
        tuple[NDArray[np.float64],NDArray[np.float64],NDArray[np.float64]]
            x-coordinates, y-coordinates, and the control points of the Bezier curve.
        """
        
        num_pts: int = self.CSA.num_pts
        t_vals: NDArray[np.float64] = (np.linspace(0,1,num_pts)).astype(dtype=np.float64) # 1Xnum_pts
        t: NDArray[np.float64] = t_vals[:, np.newaxis] # num_ptsX1

        # stack the control points to do matrix multiplication
        control_points: NDArray[np.float64] = self._get_control_points(action,side)
        
        B0: NDArray = (1 - t)**3
        B1: NDArray = 3 * t * (1 - t)**2
        B2: NDArray = 3 * t**2 * (1 - t)
        B3: NDArray = t**3
        # Create a matrix of coefficients for the cubic Bézier curve
        curve_coefficients: NDArray[np.float64] = np.column_stack((B0, B1, B2, B3)) # num_ptsX4
        # Calculate the curve points using matrix multiplication
        curve_points: NDArray[np.float64, ] = np.matmul(curve_coefficients, control_points) # num_ptsX2
        
        # Calculate the x and y coordinates using the quadratic Bézier formula
        x_coords: NDArray[np.float64] = curve_points[:, 0]
        y_coords: NDArray[np.float64] = curve_points[:, 1]
        
        return x_coords,y_coords,control_points
    
    def calculate_energy(self,
            g_left: NDArray[np.float64], 
            x_left: NDArray[np.float64],
            g_right: NDArray[np.float64], 
            x_right: NDArray[np.float64]) -> float:
        """
        Function to calculate the energy for the given Bezier curve

        Parameters
        ----------
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
        float
            energy
        """
        vn: NDArray = csa_lib.calculate_potential_coeffs(V0=self.CSA.V0,
                                                        hw_arra=self.CSA.hw_arra,
                                                        width_micrstr=self.CSA.width_micrstr,
                                                        space_bw_strps=self.CSA.space_bw_strps,
                                                        num_fs=self.CSA.num_fs,
                                                        g_left=g_left,
                                                        x_left=x_left,
                                                        g_right=g_right,
                                                        x_right=x_right)
        energy: float = csa_lib.calculate_energy(er1=self.CSA.er1,
                                                er2=self.CSA.er2,
                                                hw_arra=self.CSA.hw_arra,
                                                ht_arra=self.CSA.ht_arra,
                                                ht_subs=self.CSA.ht_subs,
                                                vn=vn)
        return energy
        
    def get_reward(self,
            action: NDArray[np.float64],
            g_left: NDArray[np.float64], 
            x_left: NDArray[np.float64],
            g_right: NDArray[np.float64], 
            x_right: NDArray[np.float64]) -> float:
        """
        Function to calculate the reward for the given action

        Parameters
        ----------
        action : NDArray[np.float64]
            action array from the RL agent
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
        float
            reward value
        """
        # Initialise
        MAX_PENALITY: float = -5
        reward: float
        
        # To promote some change
        if np.all(action == 0):
            # conditon where no chnage happens
            return MAX_PENALITY
        
        # Monotonicity is embedded in the shape drawing, hence it need not be explicitly checked  
        energy: float = self.calculate_energy(g_left=g_left,
                                                x_left=x_left,
                                                g_right=g_right,
                                                x_right=x_right)
        if energy < self.minimum_energy[-1]:
            # logger.info(f"New minimum energy obtained: {energy} VAs\n")
            self.minimum_energy = np.append(self.minimum_energy, energy)
            
        ratio_change: float = self.energy_baseline/energy
        
        # Shift the ratio by -1 as the soft plus function has exponential deviation in that range
        reward = (ratio_change*10E-5)*np.exp(12*ratio_change)
           
        return reward
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[NDArray, dict]:
        """
        Reset the environment to an initial state.
        
        Parameters:
            seed (int, optional): Random seed for reproducibility.
            options (dict, optional): Additional options for reset.
        
        Returns:
            tuple: Initial observation and an empty info dictionary.
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize the state
        initial_state: NDArray[np.float32] = np.array([self.CSA.width_micrstr,
                                                       self.CSA.space_bw_strps,
                                                       self.CSA.hw_arra, 
                                                       self.CSA.ht_arra, 
                                                       self.CSA.ht_subs, 
                                                       self.CSA.er2]).astype(dtype=np.float32)
        return initial_state, {}

    def step(self, action: NDArray) -> tuple[NDArray, float, bool, bool, dict]:
        """
        Execute one time step within the environment.
        
        Parameters:
            action (NDArray): The action to be taken.
        
        Returns:
            tuple: Next observation, reward, done flag, truncated flag, and an info dictionary.
        """
        # Validate action
        if not self.action_space.contains(action):
            raise ValueError("Action is out of bounds.")
        
        terminated: bool = False
        truncated: bool = False
        
        # Take absolute of the action values
        action = np.abs(action) # Bezier curve expects positive values
        
        # Get Bezier curves and get reward
        mid_point: int = self.action_space.shape[0]//2
        action_left: NDArray = action[:mid_point]
        action_right: NDArray = action[mid_point:]
        
        x_left: NDArray
        g_left: NDArray
        x_left,g_left,_control = self.get_bezier_curve(action=action_left,side='left')
        x_right: NDArray
        g_right: NDArray
        x_right,g_right,_control = self.get_bezier_curve(action=action_right,side='right')
        
        reward: float = self.get_reward(action=action,
                                        g_left=g_left,
                                        x_left=x_left,
                                        g_right=g_right,
                                        x_right=x_right)
        
        
        
        # Simulate the environment dynamics, Degenerate DRL (single step-no intermediate states)
        obs_space: NDArray
        info: dict
        obs_space, info = self.reset()
        terminated = True
        
        return obs_space, reward, terminated, truncated, info