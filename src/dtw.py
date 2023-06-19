"""Dynamic Time Warping (DTW)"""
import numpy as np
from numba import njit


def dtw(x, y, dist:callable=lambda x, y: np.linalg.norm(x - y, ord=1), return_path:bool=False, normalize:bool=True):
    """Dynamic Time Warping (DTW) algorithm.

    Args:
        x (np.ndarray): 1D array.
        y (np.ndarray): 1D array.
        dist (function): distance function.
        return_path (bool): return the alignment correspondence.
        normalize (bool): normalize the distance by the length of the path; devide by max(len(x), len(y)).

    Returns:
        float: DTW distance.
        list: alignment correspondence.
    """
    
    # If return_path is True, then return both the DTW distance and the path. More memory usage.
    if return_path:
       return _dtw_path(x, y, dist, normalize)

    # If return_path is False, then return only the DTW distance. Less memory usage.
    else:
        return _dtw(x, y, dist, normalize)  



@njit
def _dtw_path(x, y, dist, normalize):
    """Dynamic Time Warping (DTW) algorithm. Return both the DTW distance and the path.

    Args:
        x (np.ndarray): 1D array.
        y (np.ndarray): 1D array.
        dist (function): distance function.
        return_path (bool): return the alignment correspondence.
        normalize (bool): normalize the distance by the length of the path; devide by max(len(x), len(y)).

    Returns:
        float: DTW distance.
        list: alignment correspondence.
    """

    nx = len(x)
    ny = len(y)

    xyswap = False
    if nx > ny:  # switch inputs for less memory usage
        x, y = y, x
        nx, ny = ny, nx
        xyswap = True

    dtw_matrix = np.full((nx + 1, ny + 1), np.inf)
    dtw_matrix[0, 0] = 0.0

    for i in range(nx):
        for j in range(ny):
            cost = dist(x[i], y[j])
            dtw_matrix[i + 1, j + 1] = cost + min(dtw_matrix[i, j + 1], dtw_matrix[i + 1, j], dtw_matrix[i, j])

    # find the path
    path = []
    i, j = nx, ny
    while (i > 0) and (j > 0):
        path.append((i-1, j-1))  # use Python indexing
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            if dtw_matrix[i-1, j-1] == min(dtw_matrix[i-1, j-1], dtw_matrix[i-1, j], dtw_matrix[i, j-1]):
                i -= 1
                j -= 1
            elif dtw_matrix[i-1, j] == min(dtw_matrix[i-1, j-1], dtw_matrix[i-1, j], dtw_matrix[i, j-1]):
                i -= 1
            else: # dtw_matrix[i, j-1] is the minimum
                j -= 1
    path.append((0, 0))  # don't forget the starting point
    path.reverse()

    if xyswap:
        path = [(y, x) for x, y in path]

    if normalize:
        return dtw_matrix[-1, -1] / max(nx, ny), path
    else:
        return dtw_matrix[-1, -1], path
    

@njit
def _dtw(x, y, dist, normalize):
    """Dynamic Time Warping (DTW) algorithm. Return only the DTW distance.

    Args:
        x (np.ndarray): 1D array.
        y (np.ndarray): 1D array.
        dist (function): distance function.
        return_path (bool): return the alignment correspondence.
        normalize (bool): normalize the distance by the length of the path; devide by max(len(x), len(y)).

    Returns:
        float: DTW distance.
        list: alignment correspondence.
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

    if normalize:
        return prev_row[-1] / max(nx, ny)
    else:
        return prev_row[-1]  # return the last element of the last row