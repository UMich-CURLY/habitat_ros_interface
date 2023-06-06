#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#This script geneartes a ros/rviz compatible map based on the specified Habitat scene's top-down map

import os

import imageio
import numpy as np
import yaml
import habitat
from habitat.tasks.nav.nav import NavigationEpisode, NavigationGoal
from habitat.utils.visualizations import maps
from IPython import embed
from typing import TYPE_CHECKING, Union, cast

# from habitat.utils.visualizations.maps import COORDINATE_MIN, COORDINATE_MAX
import argparse
PARSER = argparse.ArgumentParser(description=None)
PARSER.add_argument('-s', '--scene', default="17DRP5sb8fy", type=str, help='scene')
PARSER.add_argument('-mps', '--mps', default=0.025, type=float, help='mps')
PARSER.add_argument('-d', '--dataset', default="mp3d", type=str, help='dataset')

ARGS = PARSER.parse_args()
scene = ARGS.scene
meters_per_pixel = ARGS.mps
dataset = ARGS.dataset
MAP_DIR = "/home/catkin_ws/src/habitat_ros_interface/maps"
if not os.path.exists(MAP_DIR):
    print("Didi not find maps directory")
    os.makedirs(MAP_DIR)

def get_topdown_map(config_paths, map_name):

    config = habitat.get_config(config_paths=config_paths)
    dataset = habitat.make_dataset(
        id_dataset=config.DATASET.TYPE, config=config.DATASET
    )
    env = habitat.Env(config=config, dataset=dataset)
    env.reset()
    

    
    hablab_topdown_map = maps.get_topdown_map_from_sim(
            cast("HabitatSim", env.sim), meters_per_pixel = meters_per_pixel, height = 0.0
        )
    recolor_map = np.array(
        [[128, 128, 128], [255, 255, 255], [0, 0, 0]], dtype=np.uint8
    )
    hablab_topdown_map = recolor_map[hablab_topdown_map]
    square_map_resolution = 5000
    map_resolution = [5000,5000]
    top_down_map = hablab_topdown_map
    grid_dimensions = (top_down_map.shape[0]*meters_per_pixel, top_down_map.shape[1]*meters_per_pixel)
    
    # top_down_map = maps.get_topdown_map(pathfinder = env._sim.pathfinder, map_resolution=(square_map_resolution,square_map_resolution), height = 0.0)

    # # Image containing 0 if occupied, 1 if unoccupied, and 2 if border (if
    # # the flag is set)
    # top_down_map[np.where(top_down_map == 0)] = 125
    # top_down_map[np.where(top_down_map == 1)] = 255
    # top_down_map[np.where(top_down_map == 2)] = 0
    imageio.imsave(os.path.join(MAP_DIR, map_name + ".pgm"), hablab_topdown_map)
    print("writing Yaml file! ")
    complete_name = os.path.join(MAP_DIR, map_name + ".yaml")
    f = open(complete_name, "w+")

    f.write("image: " + map_name + ".pgm\n")
    f.write("resolution: " + str(meters_per_pixel) + "\n")
    f.write("origin: [" + str(-1) + "," + str(-grid_dimensions[0]+1) + ", 0.000000]\n")
    f.write("negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196")
    f.close()


def main():
    config = {}
    with open("configs/tasks/nav_to_obj_copy.yml",'r') as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        config = yaml.load(file, Loader=yaml.FullLoader)
        if (dataset == "mo3d"):
            config['DATASET']['DATA_PATH'] = "./data/datasets/rearrange/mp3d/v1/test/content/"+scene+"0.json.gz"
        elif (dataset == "gibson"):
            config['DATASET']['DATA_PATH'] = "./data/datasets/rearrange/gibson/v1/test/content/"+scene+"0.json.gz"
        elif (dataset == "habitat"):
            config['DATASET']['DATA_PATH'] = "./data/datasets/rearrange/habitat-test-scenes/v1/test/content/"+scene+"0.json.gz"
    with open("configs/tasks/custom_rearrange.yml",'w') as file:
        print("Replacing the data config to the new scene ", scene)
        documents = yaml.dump(config, file)
    #first parameter is config path, second parameter is map name
    # if (not os.path.isfile("./maps/resolution_"+scene+"_"+str(meters_per_pixel)+".pgm")): 
    #     get_topdown_map("configs/tasks/custom_rearrange.yml", "resolution_"+scene+"_"+str(meters_per_pixel))

if __name__ == "__main__":
    main()