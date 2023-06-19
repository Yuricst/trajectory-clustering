"""Visualizing clustered LET data"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.vis import *

if __name__=="__main__":
    # load LET data
    filepath = "../data/lets_pre_clustering_epoch_100km_target_idx1_1757_polar8595.json"
    with open(filepath, 'r') as file:
        data = json.load(file)

    # load clustering result
    with open("../out/cluster_dict.json", 'r') as file:
        cluster_dict = json.load(file)

    # Output result
    vislet_xyt_group(data, cluster_dict)
    vislet_xyt_cycle_group(data, cluster_dict)

