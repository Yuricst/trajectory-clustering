"""Test script for basic visualization of the data."""
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.vis import *


if __name__=="__main__":
    # load file
    data_dir = "../data"
    filepath = os.path.join(data_dir, "lets_pre_clustering_epoch_100km_target_idx1_1757_polar8595.json")
    with open(filepath, 'r') as file:
        data = json.load(file)


    vislet_xy(data)

    vislet_xyz(data)

    vislet_xyt(data)

    vislet_xyt_cycle(data, P=5.0)

