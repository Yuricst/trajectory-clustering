"""
Load data
"""
import json
import os
import numpy as np
import matplotlib.pyplot as plt


def visualize_LET(data, P=5, r0=.0):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for key in data.keys():
        print(key)
        trajectory = np.array(data[key]['states'])
        x = trajectory[:, 0]
        y = trajectory[:, 1]
        time = np.array(data[key]['Î¸s'])

        # Convert to cylindrical space
        r = x + r0
        theta = 2 * np.pi * time / P
        ax.plot(r * np.cos(theta), r * np.sin(theta), y, linewidth=1.0)

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




if __name__=="__main__":
    # load file
    data_dir = "../data"
    filepath = os.path.join(data_dir, "lets_pre_clustering_epoch_100km_target_idx1_1757_polar8595.json")
    with open(filepath, 'r') as file:
        data = json.load(file)

    print("data keys: ", data.keys())
    print(data['1'].keys())
    print(len(data['1']['Î¸s']))
    print(len(np.array(data['1']['states'])[:,0]))

    visualize_LET(data)



