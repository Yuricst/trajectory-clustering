"""Distance functions"""
import numpy as np
from numba import njit


@njit
def dist4D(x: np.ndarray, y: np.ndarray, P: float=np.pi*2)->float:
    """Distance function for 4D data. It is the L2 norm. 
    Difference in time is taken as the minimum difference between two time points within one period. 

    Args:
        x (np.ndarray): 1D array; [x, y, z, t]
        y (np.ndarray): 1D array; [x, y, z, t]
        P (float): time period 

    Returns:
        float: LET distance.
    """
    ds = np.linalg.norm(x[:3] - y[:3], ord=2)
    dt = np.abs(x[3] % P - y[3] % P)
    return np.sqrt(ds**2 + dt**2)