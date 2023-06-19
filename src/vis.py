"""Visualization functions"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Union


def vislet_xy(data):
    """visualize the LET in xy plane"""
    fig, ax = plt.subplots(1,1,figsize=(10,8))

    for key in data.keys():
        trajectory = np.array(data[key]['states'])
        x = trajectory[:, 0]
        y = trajectory[:, 1]
        ax.plot(x, y, linewidth=0.3)

    ax.set_aspect('equal')
    ax.set(xlabel="x, DU", ylabel="y, DU")
    plt.show()


def vislet_xyz(data):
    """visualize the LET in xyz space"""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for key in data.keys():
        trajectory = np.array(data[key]['states'])
        x = trajectory[:, 0]
        y = trajectory[:, 1]
        z = trajectory[:, 2]

        ax.plot(x, y, z, linewidth=0.3)

    ax.set_xlabel('X, DU')
    ax.set_ylabel('Y, DU')
    ax.set_zlabel('Z, DU')

    plt.show()


def vislet_xyt(data):
    """visualize the LET in xy and time space"""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for key in data.keys():
        trajectory = np.array(data[key]['states'])
        x = trajectory[:, 0]
        y = trajectory[:, 1]
        time = np.array(data[key]['Î¸s'])

        ax.plot(x, y, time, linewidth=0.3)

    ax.set_xlabel('X, DU')
    ax.set_ylabel('Y, DU')
    ax.set_zlabel('Time, TU')

    plt.show()


def vislet_xyt_cycle(data, P:float=5.0):
    """visualize the LET in xy and time space, but time has period P"""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for key in data.keys():
        trajectory = np.array(data[key]['states'])
        x = trajectory[:, 0]
        y = trajectory[:, 1]
        time = np.array(data[key]['Î¸s'])

        # Convert to cylindrical space
        theta = 2 * np.pi * time / P
        ax.plot(x * np.cos(theta), x * np.sin(theta), y, linewidth=1.0)

    ax.set_zlabel('Y')

    # Add a circle on x=1, y=0
    theta_grid = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(theta_grid), np.sin(theta_grid), 0, 'k--', linewidth=1.0)

    # Add a grid plane on time=0
    x_grid = np.arange(0, ax.get_xlim()[1], 0.1)
    z_grid = np.arange(ax.get_zlim()[0], ax.get_zlim()[1], 0.1)
    nx = len(x_grid)
    nz = len(z_grid)
    for x in x_grid:
        ax.plot([x] * nz, [0] * nz, z_grid, 'k--', linewidth=0.5)
    for z in z_grid:
        ax.plot(x_grid, [0] * nx, [z] * nx, 'k--', linewidth=0.5)

    plt.show()



def vis_xyt_trajpair(x: np.ndarray, y: np.ndarray, alignment: List[Tuple[int, int]], skip: int=1):
    """Generate a 3D plot of two trajectories with lines between corresponding points

    Args:
        x (np.ndarray): 2D array; [n, 4]
        y (np.ndarray): 2D array; [n, 4]
        alignment (List[Tuple[int, int]]): a list of tuples of indices of corresponding points
        skip (int, optional): skip every `skip` points. Defaults to 1.
    """

    debug = False
    if debug:
        print(f"Number of points in x: {x.shape[0]}")
        print(f"Number of points in y: {y.shape[0]}")
        print(f"Number of corresponding points: {len(alignment)}")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the two trajectories
    ax.plot(x[:, 0], x[:, 1], x[:, 3], linewidth=0.3)
    ax.plot(y[:, 0], y[:, 1], y[:, 3], linewidth=0.3)

    # Plot the lines between corresponding points
    for n, (i, j) in enumerate(alignment):
        if n % skip == 0:
            ax.plot([x[i, 0], y[j, 0]], [x[i, 1], y[j, 1]], [x[i, 3], y[j, 3]], 'k--', linewidth=0.5)

    ax.set_xlabel('X, DU')
    ax.set_ylabel('Y, DU')
    ax.set_zlabel('Time, TU')

    plt.show()


def vis_xyt_cycle_trajpair(x: np.ndarray, y: np.ndarray, alignment: List[Tuple[int, int]], skip: int=1, P:float=2*np.pi):
    """Generate a 3D plot of two trajectories with lines between corresponding points

    Args:
        x (np.ndarray): 2D array; [n, 4]
        y (np.ndarray): 2D array; [n, 4]
        alignment (List[Tuple[int, int]]): a list of tuples of indices of corresponding points
        skip (int, optional): skip every `skip` points. Defaults to 1.
        P (float, optional): period of time. Defaults to 2*np.pi.
    """
    debug = False
    if debug:
        print(f"Number of points in x: {x.shape[0]}")
        print(f"Number of points in y: {y.shape[0]}")
        print(f"Number of corresponding points: {len(alignment)}")
   

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the two trajectories
    for s in [x, y]:
        # Convert to cylindrical space
        theta = 2 * np.pi * s[:, 3] / P
        ax.plot(s[:, 0] * np.cos(theta), s[:, 0] * np.sin(theta), s[:, 1], linewidth=1.0)

    # Plot the lines between corresponding points
    for n, (i, j) in enumerate(alignment):
        if n % skip == 0:
            xline = [x[i, 0] * np.cos(2 * np.pi * x[i, 3] / P), y[j, 0] * np.cos(2 * np.pi * y[j, 3] / P)]
            yline = [x[i, 0] * np.sin(2 * np.pi * x[i, 3] / P), y[j, 0] * np.sin(2 * np.pi * y[j, 3] / P)]
            zline = [x[i, 1], y[j, 1]]
            ax.plot(xline, yline, zline, 'k--', linewidth=0.5)


    # Add a circle on x=1, y=0
    theta_grid = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(theta_grid), np.sin(theta_grid), 0, 'k--', linewidth=1.0)


    ax.set_zlabel('Y, DU')

    plt.show()




