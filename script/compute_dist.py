"""Compute and save distance matrix of the LETs in the dataset."""

import json
import numpy as np
import os
import sys
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.preprocessing import get4ddata
from src.dist import dist4D
from src.dtw import dtw


if __name__=="__main__":
    # Create output directory
    out_dir = "../out"
    os.makedirs(out_dir, exist_ok=True)

    # load file
    #filepath = r"D:\khrg_data\2023_amos\lets_pre_clustering_epoch_100km_target_idx1_1757_polar8595.json"
    #filepath = r"D:\khrg_data\2023_amos\lets_pre_clustering_full_epoch_100km_target_idx1_1757_polar8595.json"
    filepath = r"/home/yshimane3/Documents/data/amos2023/lets_pre_clustering_epoch_100km_target_idx1_1757_polar8595.json"
    
    #"../data/lets_pre_clustering_epoch_100km_target_idx1_1757_polar8595.json"
    with open(filepath, 'r') as file:
        data = json.load(file)

    # convert to 4d data
    data_4d = get4ddata(data)

    # Compute distance matrix using DTW
    keys = list(data_4d.keys())
    idx_dict = {key: i for i, key in enumerate(keys)}
    n = len(keys)
    dist_mat = np.zeros((n, n))

    pbar = tqdm(total=n*(n+1)//2)
    for i in range(n):
        for j in range(i, n):
            dist_mat[i, j] = dtw(data_4d[keys[i]], data_4d[keys[j]], dist=dist4D)
            dist_mat[j, i] = dist_mat[i, j]
            pbar.update(1)

    # Save distance matrix
    np.save(os.path.join(out_dir, "dist_mat.npy"), dist_mat)

    # Save index dictionary
    with open(os.path.join(out_dir, "idx_dict.json"), 'w') as file:
        json.dump(idx_dict, file)
