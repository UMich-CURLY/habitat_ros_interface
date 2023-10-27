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
from habitat_sim.utils import common as utils
import magnum as mn
import math
import quaternion as qt
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

def draw_agent_in_top_down(env, map_path = "agent_pos.png", line = None):
    agent_state = env.sim.get_agent_state()
    agent_pos = agent_state.position
    meters_per_pixel =0.025
    
    top_down_map = maps.get_topdown_map(
        env.sim.pathfinder, height=0.06, meters_per_pixel=meters_per_pixel
    )
    recolor_map = np.array(
        [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
    )
    top_down_map = recolor_map[top_down_map]
    grid_dimensions = (top_down_map.shape[0], top_down_map.shape[1])
    agent_grid_pos = maps.to_grid(
        agent_pos[2], agent_pos[0], grid_dimensions, pathfinder=env.sim.pathfinder
    )
    agent_forward = utils.quat_to_magnum(
        env.sim.agents[0].get_state().rotation
    ).transform_vector(mn.Vector3(0, 0, -1.0))
    agent_orientation = math.atan2(agent_forward[0], agent_forward[2])
    # draw the agent and trajectory on the map
    maps.draw_agent(
        top_down_map, agent_grid_pos, agent_orientation, agent_radius_px=8
    )
    door_points_2d = []
    if line is not None:
        for i in range(line.shape[0]):
            door_points_2d.append(maps.to_grid(
                line[i,2], line[i,0], grid_dimensions, pathfinder=env.sim.pathfinder
            ))
            try:
                top_down_map[door_points_2d[i][0], door_points_2d[i][1]] = [0,255,0]
            except:
                pass
    cv2.imwrite(map_path, top_down_map)

def get_topdown_map(config_paths, map_name):

    config = habitat.get_config(config_paths=config_paths)
    dataset = habitat.make_dataset(
        id_dataset=config.DATASET.TYPE, config=config.DATASET
    )
    env = habitat.Env(config=config, dataset=dataset)
    env.reset()
    observations = env.reset()
    np.random.seed(520)    
    semantic_scene = env.sim.semantic_annotations()
    instance_id_to_label_id = {int(obj.id.split("_")[-1]): obj.category.index() for obj in semantic_scene.objects}
    names = {int(obj.id.split("_")[-1]): obj.category.name() for obj in semantic_scene.objects}
    instance_label_mapping = np.array([ instance_id_to_label_id[i] for i in range(len(instance_id_to_label_id)) ])
    instance_names = np.array([names[i] for i in range(len(names))])
    candidate_doors_index = np.where(instance_names == 'door')[0]
    chosen_object = semantic_scene.objects[np.random.choice(candidate_doors_index)]    
    temp_position = env._sim.pathfinder.get_random_navigable_point_near(chosen_object.aabb.center,1.5)
    agent_state = env.sim.get_agent_state()
    env.sim.set_agent_state(temp_position,np.array([0,0,0,1]))
    agent_door = chosen_object.aabb.center - agent_state.position
    agent_forward = utils.quat_to_magnum(
            env.sim.agents[0].get_state().rotation
        ).transform_vector(mn.Vector3(0.0,0.0,-1.0))
    agent_door[2] = -agent_door[2]
    agent_door[1] =0
    agent_forward = utils.quat_to_magnum(
            env.sim.agents[0].get_state().rotation
        ).transform_vector(mn.Vector3(agent_door[0], agent_door[1], agent_door[2]))

    diff_quat = utils.quat_from_two_vectors(np.array(agent_forward), agent_door)
    # diff_quat = diff_quat*qt.quaternion(0,0,-0.7071,0.7071)
    diff_quat = qt.from_rotation_vector(np.array(agent_forward))
    env.sim.set_agent_state(temp_position, diff_quat)
    render_camera = env._sim.get_agent(0).scene_node.node_sensor_suite.get_sensors()['rgb']
    # render_camera.zoom(2)
    render_camera = env._sim.get_agent(0).scene_node.node_sensor_suite.get_sensors()['semantic']
    render_camera.zoom(2)
    observations = env.sim.get_observations_at(chosen_object.aabb.center)
    observations_semantic = np.take(instance_label_mapping, observations['semantic'])
    semantic_img = Image.new("P", (observations_semantic.shape[1], observations_semantic.shape[0]))
    semantic_img.putpalette(d3_40_colors_rgb.flatten())
    semantic_img.putdata((observations_semantic.flatten()%40).astype(np.uint8))
    semantic_img = semantic_img.convert("RGBA")
    semantic_img = np.asarray(semantic_img)
    cv2.imwrite("semantic_img.png", semantic_img)
    embed()
    # observations_rgb = np.take(instance_label_mapping, observations['rgb'])
    # rgb_img = Image.new("P", (observations_rgb.shape[0], observations_rgb.shape[1],3))
    # rgb_img.pudata((observations_rgb))
    observations = env.sim.get_sensor_observations()
    cv2.imwrite("rgb_img.png", np.asarray(observations['rgb']))
    
    agent_state = env.sim.get_agent_state()
    agent_pos = agent_state.position
    meters_per_pixel =0.025
    
    ### draw door in topdown map ###
    a = np.append(chosen_object.aabb.center+chosen_object.aabb.sizes/10, 1.0)
    b = np.append(chosen_object.aabb.center-chosen_object.aabb.sizes/10, 1.0)
    a[1] = chosen_object.aabb.center[1]
    b[1] = chosen_object.aabb.center[1]
    line = np.linspace(a,b,50)
    draw_agent_in_top_down(env, map_path = "agent_pos.png", line = line)
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
    hablab_topdown_map = maps.get_topdown_map_from_sim(
            cast("HabitatSim", env.sim), meters_per_pixel= meters_per_pixel
        )
    recolor_map = np.array(
        [[128, 128, 128], [255, 255, 255], [0, 0, 0]], dtype=np.uint8
    )
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