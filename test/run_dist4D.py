
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.dtw import *

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