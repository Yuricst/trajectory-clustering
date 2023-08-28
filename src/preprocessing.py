"""Data preprocessing for LETS data."""

import os
import numpy as np


def get4ddata(data: dict):
    """Concatenate x, y, z, t into one array."""
    n = len(data)
    out = {}
    for i, key in enumerate(data.keys()):
        try:
            trajectory = np.array(data[key]['states'])
            time = np.array(data[key]['θs'])

            out[key] = np.array([trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], time]).T
        except:
            print(f"Skipped key {key}")
            print(data[key].keys())

    return out
