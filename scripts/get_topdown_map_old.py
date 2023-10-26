import os

import imageio
import numpy as np
import yaml
import habitat
from habitat.tasks.nav.nav import NavigationEpisode, NavigationGoal
from habitat.utils.visualizations import maps
# from habitat.utils.visualizations.maps import COORDINATE_MIN, COORDINATE_MAX
from typing import TYPE_CHECKING, Union, cast
from habitat_sim.utils.common import d3_40_colors_rgb
from PIL import Image
import cv2
import argparse
from IPython import embed

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
    observations = env.reset()
        
    semantic_scene = env.sim.semantic_annotations()
    instance_id_to_label_id = {int(obj.id.split("_")[-1]): obj.category.index() for obj in semantic_scene.objects}
    names = {int(obj.id.split("_")[-1]): obj.category.name() for obj in semantic_scene.objects}
    instance_label_mapping = np.array([ instance_id_to_label_id[i] for i in range(len(instance_id_to_label_id)) ])
    instance_names = np.array([names[i] for i in range(len(names))])
    candidate_doors_index = np.where(instance_names == 'door')[0]
    chosen_object = semantic_scene.objects[np.random.choice(candidate_doors_index)]    
    temp_position = env._sim.pathfinder.get_random_navigable_point_near(chosen_object.aabb.center,2)
    agent_state = env.sim.get_agent_state()
    env.sim.set_agent_state(temp_position, agent_state.rotation)
    observations = env.sim.get_sensor_observations()
    observations_semantic = np.take(instance_label_mapping, observations['semantic'])
    semantic_img = Image.new("P", (observations_semantic.shape[1], observations_semantic.shape[0]))
    semantic_img.putpalette(d3_40_colors_rgb.flatten())
    semantic_img.putdata((observations_semantic.flatten()%40).astype(np.uint8))
    semantic_img = semantic_img.convert("RGBA")
    semantic_img = np.asarray(semantic_img)
    cv2.imwrite("semantic_img.png", semantic_img)
    
    # observations_rgb = np.take(instance_label_mapping, observations['rgb'])
    # rgb_img = Image.new("P", (observations_rgb.shape[0], observations_rgb.shape[1],3))
    # rgb_img.pudata((observations_rgb))
    cv2.imwrite("rgb_img.png", np.asarray(observations['rgb']))
    embed()
    meters_per_pixel =0.025
    hablab_topdown_map = maps.get_topdown_map_from_sim(
            cast("HabitatSim", env.sim), meters_per_pixel= meters_per_pixel
        )
    recolor_map = np.array(
        [[128, 128, 128], [255, 255, 255], [0, 0, 0]], dtype=np.uint8
    )
    hablab_topdown_map = recolor_map[hablab_topdown_map]
    square_map_resolution = 5000
    map_resolution = [5000,5000]
    top_down_map = hablab_topdown_map
    grid_dimensions = (top_down_map.shape[0]*meters_per_pixel, top_down_map.shape[1]*meters_per_pixel)
    # Image containing 0 if occupied, 1 if unoccupied, and 2 if border (if
    # the flag is set)
    top_down_map[np.where(top_down_map == 0)] = 125
    top_down_map[np.where(top_down_map == 1)] = 255
    top_down_map[np.where(top_down_map == 2)] = 0
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
    with open("configs/tasks/pointnav_rgbd.yaml",'r') as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        config = yaml.load(file, Loader=yaml.FullLoader)
        if (dataset == "mp3d"):
            config['DATASET']['DATA_PATH'] = "./data/datasets/pointnav/mp3d/v1/test/content/"+scene+"0.json.gz"
        elif (dataset == "gibson"):
            config['DATASET']['DATA_PATH'] = "./data/datasets/pointnav/gibson/v1/test/content/"+scene+"0.json.gz"
        elif (dataset == "habitat"):
            config['DATASET']['DATA_PATH'] = "./data/datasets/pointnav/habitat-test-scenes/v1/test/content/"+scene+"0.json.gz"
    with open("configs/tasks/pointnav_rgbd.yaml",'w') as file:
        print("Replacing the data config to the new scene ", scene)
        documents = yaml.dump(config, file)
    #first parameter is config path, second parameter is map name

    # if (not os.path.isfile("./maps/resolution_"+scene+"_"+str(meters_per_pixel)+".pgm")): 
    get_topdown_map("configs/tasks/pointnav_rgbd.yaml", "resolution_"+scene+"_"+str(meters_per_pixel))



if __name__ == "__main__":
    main()