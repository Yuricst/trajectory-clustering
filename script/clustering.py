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
    # load file
    distmat = np.load("../out/dist_mat.npy")
    with open("../out/idx_dict.json", 'r') as file:
        idx_dict = json.load(file)

    # clustering
    db = DBSCAN(eps=0.5, min_samples=1, metric='precomputed')

    labels = db.fit_predict(distmat)
    print("Number of clusters: ", len(set(labels)))
    print(labels)

    # Save clustering result
    with open("../out/cluster_labels.json", 'w') as file:
        json.dump(labels.tolist(), file)


    # Visualize clustering result
    data_dir = "../data"
    filepath = os.path.join(data_dir, "lets_pre_clustering_epoch_100km_target_idx1_1757_polar8595.json")
    with open(filepath, 'r') as file:
        data = json.load(file)

    vislet_xyt_group(data, labels, idx_dict)
    vislet_xyt_cycle_group(data, labels, idx_dict)

    