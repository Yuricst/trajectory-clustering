"""Test script for DTW distance calculation. DTW alignment path is also visualized."""
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import random

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.dtw import dtw
from src.dist import dist4D
from src.preprocessing import get4ddata
from src.vis import *


if __name__=="__main__":
    # load file
    filepath = "../data/lets_pre_clustering_epoch_100km_target_idx1_1757_polar8595.json"
    with open(filepath, 'r') as file:
        data = json.load(file)

    # Pick pairs of LET data to be visualized
    randomize = False
    if randomize:
        pairs = []
        for i in range(10):
            keys = random.sample(list(data.keys()), 2)
            pairs.append(keys)
        print(pairs)
    else:
        pairs = [
            ('32', '29'),  
            ('32', '78'),
            ('32', '92'),
            ('32', '88'),
            ('32', '91'),
        ]

    # Visualize the DTW distance and alignment
    for keys in pairs:
        sampled_data = {key: data[key] for key in keys}
        sampled_data = get4ddata(sampled_data)
        print(sampled_data.keys())

        # Distance between two trajectories, memory efficient
        dlet_1 = dtw(sampled_data[keys[0]], sampled_data[keys[1]], dist=dist4D)
        # Distance between two trajectories, memory inefficient
        dlet_2, path = dtw(sampled_data[keys[0]], sampled_data[keys[1]], dist=dist4D, return_path=True)
        assert dlet_1 == dlet_2
        print("DTW distance: ", dlet_1)

        # Visualize the DTW alignment
        vis_xyt_trajpair(sampled_data[keys[0]], sampled_data[keys[1]], path, skip=10)
        vis_xyt_cycle_trajpair(sampled_data[keys[0]], sampled_data[keys[1]], path, skip=10)


    


