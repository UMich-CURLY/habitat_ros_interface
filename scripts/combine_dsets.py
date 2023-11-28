#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import glob
import gzip
import json
import multiprocessing
import os
import os.path as osp

import tqdm
from habitat.tasks.nav.nav import (
    NavigationEpisode,
    NavigationGoal,
    ShortestPathPoint,
)
import habitat
import habitat_sim
from habitat.datasets.pointnav.pointnav_generator import generate_pointnav_episode
import argparse
'''
Reads everything from the dataset directory and creates a single json file with all the episodes
'''

class pointnav_data():
    def __init__(self):
        self.episodes = []
        self.episode_num = 0
        # cfg = habitat.get_config()
        # sim = habitat.sims.make_sim("Sim-v0", config=cfg.SIMULATOR)
        self.dset = habitat.datasets.make_dataset("PointNav-v1")
        

    def from_json(self, json_str: str):
        deserialized = json.loads(json_str)
        for episode in deserialized["episodes"]:
            episode = NavigationEpisode(**episode)
            episode.episode_id = self.episode_num
            self.episode_num+=1
            for g_index, goal in enumerate(episode.goals):
                episode.goals[g_index] = NavigationGoal(**goal)
            if episode.shortest_paths is not None:
                for path in episode.shortest_paths:
                    for p_index, point in enumerate(path):
                        path[p_index] = ShortestPathPoint(**point)
            self.episodes.append(episode)

    def get_json_files(self,folder):
        for filename in os.listdir(folder):
            with gzip.open(folder+"/"+filename, "rb") as f:
                try:
                    self.from_json(f.read())
                except:
                    print("Bad episode", filename)
        self.dset.episodes = self.episodes


# scenes = glob.glob("./data/scene_datasets/mp3d/17DRP5sb8fy.glb")
# with multiprocessing.Pool(8) as pool, tqdm.tqdm(total=len(scenes)) as pbar:
#     for _ in pool.imap_unordered(_generate_fn, scenes):
#         pbar.update()

# with gzip.open(f"./data/datasets/pointnav/mp3d/v1/all/all.json.gz", "wt") as f:
#     json.dump(dict(episodes=[]), f)

if __name__ == "__main__":
    data = pointnav_data()
    data.get_json_files("/home/catkin_ws/src/habitat_ros_interface/data/datasets/pointnav/mp3d/v1/test/content")
    outfile = "/home/catkin_ws/src/habitat_ros_interface/data/datasets/pointnav/mp3d/v1/full_data.json.gz"
    with gzip.open(outfile, "wt") as f:
        f.write(data.dset.to_json())