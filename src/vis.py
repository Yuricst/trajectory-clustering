"""Visualization functions"""

import matplotlib.pyplot as plt
import numpy as np


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