
import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.dist import dist4D

if __name__=="__main__":
    x = np.array([1, 2, 3, 1])
    y = np.array([1, 2, 3, 1 + np.pi * 2])

    print("x: ", x)
    print("y: ", y)
    print("dist4D(x, y): ", dist4D(x, y))


    x = np.array([1, 2, 3, 1])
    y = np.array([1, 2, 3, 1 + np.pi * 1])

    print("x: ", x)
    print("y: ", y)
    print("dist4D(x, y): ", dist4D(x, y))


    x = np.array([1, 2, 3, 1])
    y = np.array([1, 2, 2, 1 + np.pi * 2])

    print("x: ", x)
    print("y: ", y)
    print("dist4D(x, y): ", dist4D(x, y))


    x = np.array([1, 2, 3, 1])
    y = np.array([1, 2, 2, 1 + np.pi * 1])

    print("x: ", x)
    print("y: ", y)
    print("dist4D(x, y): ", dist4D(x, y))