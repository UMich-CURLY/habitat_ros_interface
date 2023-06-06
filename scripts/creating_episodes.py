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

import habitat
import habitat_sim
from habitat.datasets.pointnav.pointnav_generator import generate_pointnav_episode
import argparse
PARSER = argparse.ArgumentParser(description=None)
PARSER.add_argument('-s', '--scene', default="17DRP5sb8fy", type=str, help='scene')
PARSER.add_argument('-d', '--dataset', default="mp3d", type=str, help='dataset')

ARGS = PARSER.parse_args()
scene = ARGS.scene
dataset = ARGS.dataset
num_episodes_per_scene = 1

def _generate_fn():
    cfg = habitat.get_config()
    cfg.defrost()
    if (dataset == "mp3d"):
        cfg.SIMULATOR.SCENE = "data/scene_datasets/mp3d/"+scene+"/"+scene+".glb"
    elif(dataset == "gibson"):
        cfg.SIMULATOR.SCENE = "/home/catkin_ws/src/habitat_ros_interface/data/scene_datasets/gibson/"+scene+".glb"
    elif (dataset == "habitat"):
        cfg.SIMULATOR.SCENE = "/home/catkin_ws/src/habitat_ros_interface/data/scene_datasets/habitat-test-scenes/"+scene+".glb"
    else:
        print("No dataset found")
        exit(0)
    cfg.SIMULATOR.AGENT_0.SENSORS = []
    cfg.freeze()

    sim = habitat.sims.make_sim("Sim-v0", config=cfg.SIMULATOR)

    dset = habitat.datasets.make_dataset("PointNav-v1")
    dset.episodes = list(
        generate_pointnav_episode(
            sim, num_episodes_per_scene, is_gen_shortest_path=False
        )
    )
    count_episodes = 0
    if (dataset == "mp3d"):
        for ep in dset.episodes:
            ep.scene_id = "data/scene_datasets/mp3d/"+scene+"/"+scene+".glb"
        print(dset.episodes)
        scene_key = osp.basename(osp.dirname(osp.dirname(scene)))
        out_file = f"./data/datasets/pointnav/mp3d/v1/test/content/"+scene + str(count_episodes)+".json.gz"
        os.makedirs(osp.dirname(out_file), exist_ok=True)
        with gzip.open(out_file, "wt") as f:
            f.write(dset.to_json())
    elif (dataset == "gibson"):
        for ep in dset.episodes:
            ep.scene_id = "/home/catkin_ws/src/habitat_ros_interface/data/scene_datasets/gibson/"+scene+".glb"
        print(dset.episodes)
        scene_key = osp.basename(osp.dirname(osp.dirname(scene)))
        out_file = f"./data/datasets/pointnav/gibson/v1/test/content/"+scene + str(count_episodes)+".json.gz"
        os.makedirs(osp.dirname(out_file), exist_ok=True)
        with gzip.open(out_file, "wt") as f:
            f.write(dset.to_json())
    if (dataset == "habitat"):
        for ep in dset.episodes:
            ep.scene_id = "data/scene_datasets/habitat-test-scenes/"+scene+".glb"
        print(dset.episodes)
        scene_key = osp.basename(osp.dirname(osp.dirname(scene)))
        out_file = f"./data/datasets/pointnav/habitat-test-scenes/v1/test/content/"+scene + str(count_episodes)+".json.gz"
        os.makedirs(osp.dirname(out_file), exist_ok=True)
        with gzip.open(out_file, "wt") as f:
            f.write(dset.to_json())
    else:
        print("No dataset found")
        exit(0)


# scenes = glob.glob("./data/scene_datasets/mp3d/17DRP5sb8fy.glb")
# with multiprocessing.Pool(8) as pool, tqdm.tqdm(total=len(scenes)) as pbar:
#     for _ in pool.imap_unordered(_generate_fn, scenes):
#         pbar.update()

# with gzip.open(f"./data/datasets/pointnav/mp3d/v1/all/all.json.gz", "wt") as f:
#     json.dump(dict(episodes=[]), f)

if __name__ == "__main__":
	_generate_fn()