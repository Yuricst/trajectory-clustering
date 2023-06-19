"""Clustering LET data using DBSCAN"""

import json
import numpy as np
import os
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.vis import *


if __name__=="__main__":
    # load distance matrix and index dictionary
    distmat = np.load("../out/dist_mat.npy")
    with open("../out/idx_dict.json", 'r') as file:
        idx_dict = json.load(file)

    # clustering
    db = DBSCAN(eps=0.5, min_samples=1, metric='precomputed')
    labels = db.fit_predict(distmat)
    cluster_dict = {key: int(labels[idx_dict[key]]) for key in idx_dict.keys()}

    # Output result
    print("Number of clusters: ", len(set(labels)))
    print("Cluster Breakdown: ", cluster_dict)

    # Save clustering result
    with open("../out/cluster_dict.json", 'w') as file:
        json.dump(cluster_dict, file)

    # Visualize clustering result

