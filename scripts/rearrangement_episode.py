#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import habitat
from habitat.datasets.rearrange.rearrange_dataset import *
from IPython import embed
import os
import os.path as osp
import glob
import gzip
import json
import multiprocessing
from habitat.tasks.nav.nav import (
    NavigationEpisode,
    NavigationGoal,
    ShortestPathPoint,
)
import argparse
PARSER = argparse.ArgumentParser(description=None)
PARSER.add_argument('-s', '--scene', default="17DRP5sb8fy", type=str, help='scene')
PARSER.add_argument('-d', '--dataset', default="mp3d", type=str, help='dataset')

ARGS = PARSER.parse_args()
scene = ARGS.scene
dataset = ARGS.dataset

def from_json(json_str: str):
    deserialized = json.loads(json_str)
    for episode in deserialized["episodes"]:
        episode = NavigationEpisode(**episode)
        episode.episode_id = 0
        for g_index, goal in enumerate(episode.goals):
            episode.goals[g_index] = NavigationGoal(**goal)
        if episode.shortest_paths is not None:
            for path in episode.shortest_paths:
                for p_index, point in enumerate(path):
                    path[p_index] = ShortestPathPoint(**point)
        return episode

def example():
    # Note: Use with for the example testing, doesn't need to be like this on the README

    with habitat.Env(
        config=habitat.get_config(
            "configs/tasks/rearrange/play.yaml"
        )
    ) as env:
        print("Environment creation successful")
        observations = env.reset()  # noqa: F841

        print("Agent acting inside environment.")
        count_steps = 0
        print(env.episodes)
        pointnav_dataset_path = "/home/catkin_ws/src/habitat_ros_interface/data/datasets/pointnav/mp3d/v1/test/content/"+scene+"0.json.gz"
        with gzip.open(pointnav_dataset_path, "rb") as f:
            episode = from_json(f.read())
        # for i in range(len(env.episodes)):
        #     for j in range(len(env.episodes[i].static_objs)):
        #         strings = env.episodes[i].static_objs[j][0] 
        #         new_string = "/habitat-lab/"+strings
        #         print(new_string)
        #     #     env.episodes[i].static_objs[j][0] = new_string
        #     #     if (j>=4):
        #     #         env.episodes[i].static_objs[j] = []
        #     # env.episodes[i].static_objs = env.episodes[i].static_objs[0:4]
        # # env.episodes[0].static_objs = []
        # # env.episodes[0].art_objs = []
        env.episodes[0].markers = []
        env.episodes[0].ao_states = {}
        if (dataset == "mp3d"):
            env.episodes[0].scene_id = "/home/catkin_ws/src/habitat_ros_interface/data/scene_datasets/mp3d/"+scene+"/"+scene+".glb"
            env.episodes[0].scene_dataset_config = "/home/catkin_ws/src/habitat_ros_interface/data/scene_datasets/mp3d/mp3d.scene_dataset_config.json"
            out_file = f"/home/catkin_ws/src/habitat_ros_interface/data/datasets/rearrange/mp3d/v1/test/content/"+scene+"0.json.gz"
        elif (dataset == "gibson"):
            env.episodes[0].scene_id = "/home/catkin_ws/src/habitat_ros_interface/data/scene_datasets/gibson/"+scene+".glb"
            # env.episodes[0].scene_dataset_config = "/home/catkin_ws/src/habitat_ros_interface/data/scene_datasets/mp3d/mp3d.scene_dataset_config.json"
            out_file = f"/home/catkin_ws/src/habitat_ros_interface/data/datasets/rearrange/gibson/v1/test/content/"+scene+"0.json.gz"        
        elif (dataset == "habitat"):
            env.episodes[0].scene_id = "/home/catkin_ws/src/habitat_ros_interface/data/scene_datasets/habitat-test-scenes/"+scene+".glb"
            # env.episodes[0].scene_dataset_config = "/home/catkin_ws/src/habitat_ros_interface/data/scene_datasets/mp3d/mp3d.scene_dataset_config.json"
            out_file = f"/home/catkin_ws/src/habitat_ros_interface/data/datasets/rearrange/habitat-test-scenes/v1/test/content/"+scene+"0.json.gz"        
        env.episodes[0].start_position = episode.start_position
        env.episodes[0].start_rotation = episode.start_rotation
        env.episodes[0].start_position = episode.start_position
        env.episodes[0].info.update(episode.info)
        env.episodes[0].goals = episode.goals
        # env.episodes[0].targets = []
        rearrange_dataset = RearrangeDatasetV0()
        rearrange_dataset.episodes = [env.episodes[0]]
        
        os.makedirs(osp.dirname(out_file), exist_ok=True)
        with gzip.open(out_file, "wt") as f:
            f.write(rearrange_dataset.to_json())
        
        # while not env.episode_over:
        #     observations = env.step(env.action_space.sample())  # noqa: F841
        #     count_steps += 1
        # print("Episode finished after {} steps.".format(count_steps))


if __name__ == "__main__":
    example()
