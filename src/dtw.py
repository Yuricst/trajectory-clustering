"""Dynamic Time Warping (DTW)"""
import numpy as np
from numba import njit


@njit
def dtw(x, y, dist=lambda x, y: np.linalg.norm(x - y, ord=1)):
    """Dynamic Time Warping (DTW) algorithm.

    Args:
        x (np.ndarray): 1D array.
        y (np.ndarray): 1D array.
        dist (function): distance function.

    Returns:
        float: DTW distance.
    """
    nx = len(x)
    ny = len(y)

    if nx > ny:  # switch inputs for less memory usage
        x, y = y, x
        nx, ny = ny, nx
    
    prev_row = np.full((ny + 1, ), np.inf)
    curr_row = np.empty((ny + 1, ))

    prev_row[0] = 0.0
    for i in range(nx):
        curr_row[0] = np.inf
        for j in range(ny):
            curr_row[j + 1] = dist(x[i], y[j]) + min(curr_row[j], prev_row[j], prev_row[j + 1])
        prev_row, curr_row = curr_row, prev_row  # swap rows for next iteration

    return prev_row[-1]  # return the last element of the last row


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
