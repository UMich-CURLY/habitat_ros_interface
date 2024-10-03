import os 
import argparse
import numpy as np
from IPython import embed
import cv2
PARSER = argparse.ArgumentParser(description=None)

PARSER.add_argument('-d', '--dataset', default="mp3d", type=str, help='dataset')
ARGS = PARSER.parse_args()
dataset = ARGS.dataset

DATA_PATH = "./data/datasets/irl_may_7_12"
invalid_scenes = []

def check_neighbor(first, second):
    actions = np.array([[0,0], [1,0], [-1,0], [0,1], [0,-1]])
    possible_neighbors = list(first) + actions
    for neighbor in possible_neighbors:
        if (second==neighbor).all():
            return True
    return False
    
def is_valid_traj(traj):
    i = 0
    while i < len(traj)-1:
        if not check_neighbor(traj[i], traj[i+1]):
            return False
        i = i+1
    return True

def traj_interp(c):
    d = c.astype(int)
    iter = len(d) - 1
    added = 0
    i = 0
    while i < iter:
        while np.sqrt((d[i+added,0]-d[i+1+added,0])**2 + (d[i+added,1]-d[i+1+added,1])**2) > np.sqrt(1):
            d = np.insert(d, i+added+1, [0, 0], axis=0)
            if d[i+added+2, 0] - d[i+added, 0] > 0:
                d[i+added+1, 0] = d[i+added, 0] + 1
                d[i+added+1, 1] = d[i+added, 1]
            elif d[i+added+2, 0] - d[i+added, 0] < 0:
                d[i+added+1, 0] = d[i+added, 0] - 1
                d[i+added+1, 1] = d[i+added, 1]
            else:
                d[i+added+1, 0] = d[i+added, 0]
                if d[i+added+2, 1] - d[i+added, 1] > 0:
                    d[i+added+1, 1] = d[i+added, 1] + 1
                elif d[i+added+2, 1] - d[i+added, 1] < 0:
                    d[i+added+1, 1] = d[i+added, 1] - 1
                else:
                    d[i+added+1, 1] = d[i+added, 1]
            added += 1
        i += 1
    
    return d
traj_meshes = []
for foldername in os.listdir(DATA_PATH):
    folder_path = DATA_PATH+"/"+foldername
    if (folder_path[-3:]) == "npy":
        continue
    traj_data = np.load(folder_path+"/traj.npy", allow_pickle=True)
    traj_fixed_data = traj_interp(traj_data)
    
    traj_mesh = np.zeros([60,60])
    traj_mesh[traj_fixed_data[:,0], traj_fixed_data[:,1]] = 1
    print(traj_mesh)
    traj_meshes.append(traj_mesh)

with open(DATA_PATH+"/traj_meshes.npy", 'wb') as f:
        np.save(f, np.array(traj_meshes))
