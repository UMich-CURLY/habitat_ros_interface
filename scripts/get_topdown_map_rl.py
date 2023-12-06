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
import creating_episodes as ep
import gzip
import json
from habitat.tasks.nav.nav import (
    NavigationEpisode,
    NavigationGoal,
    ShortestPathPoint,
)
PARSER = argparse.ArgumentParser(description=None)

PARSER.add_argument('-s', '--scene', default="17DRP5sb8fy", type=str, help='scene')
PARSER.add_argument('-mps', '--mps', default=0.025, type=float, help='mps')
PARSER.add_argument('-d', '--dataset', default="mp3d", type=str, help='dataset')
ARGS = PARSER.parse_args()
print(ARGS)
scene = ARGS.scene
meters_per_pixel = ARGS.mps
dataset = ARGS.dataset
MAP_DIR = "/home/catkin_ws/src/habitat_ros_interface/maps"
IMAGE_DIR = "/home/catkin_ws/src/habitat_ros_interface/images/current_scene"
if not os.path.exists(MAP_DIR):
    print("Didi not find maps directory")
    os.makedirs(MAP_DIR)

if not os.path.exists(IMAGE_DIR):
    print("Didi not find maps directory")
    os.makedirs(IMAGE_DIR)

def draw_agent_in_top_down(sim, map_path = "agent_pos.png", line = None, goal = None, points_3d = None):
    agent_state = sim.get_agent_state()
    agent_pos = agent_state.position
    meters_per_pixel =0.025
    
    top_down_map = maps.get_topdown_map(
        sim.pathfinder, height=agent_pos[1], meters_per_pixel=meters_per_pixel
    )
    recolor_map = np.array(
        [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
    )
    top_down_map = recolor_map[top_down_map]
    grid_dimensions = (top_down_map.shape[0], top_down_map.shape[1])
    agent_grid_pos = maps.to_grid(
        agent_pos[2], agent_pos[0], grid_dimensions, pathfinder=sim.pathfinder
    )
    agent_forward = utils.quat_to_magnum(
        sim.agents[0].get_state().rotation
    ).transform_vector(mn.Vector3(0, 0, -1.0))
    agent_orientation = math.atan2(agent_forward[0], agent_forward[2])
    # draw the agent and trajectory on the map
    maps.draw_agent(
        top_down_map, agent_grid_pos, agent_orientation, agent_radius_px=8
    )
    if goal is not None:
        goal_grid_pos = maps.to_grid(
            goal[2], goal[0], grid_dimensions, pathfinder=sim.pathfinder
        )
        maps.draw_agent(
            top_down_map, goal_grid_pos, agent_orientation, agent_radius_px=8
        )
    door_points_2d = []
    if points_3d is not None:
        for point_3d in points_3d:
            point_2d = maps.to_grid(
                point_3d[2], point_3d[0], grid_dimensions, pathfinder=sim.pathfinder
            )
            top_down_map[point_2d] = [255,0,0]
    if line is not None:
        for i in range(line.shape[0]):
            door_points_2d.append(maps.to_grid(
                line[i,2], line[i,0], grid_dimensions, pathfinder=sim.pathfinder
            ))
            try:
                top_down_map[door_points_2d[i][0], door_points_2d[i][1]] = [0,255,0]
            except:
                pass
    cv2.imwrite(map_path, top_down_map)

def sem_img_to_world(proj, cam, W,H, u, v, debug = False):
    K = proj
    T_world_camera = cam
    rotation_0 = T_world_camera[0:3,0:3]
    translation_0 = T_world_camera[0:3,3]
    uv_1=np.array([[u,v,1]], dtype=np.float32)
    uv_1=np.array([[2*u/W -1,-2*v/H +1,1]], dtype=np.float32)
    uv_1=np.array([[2*v/H -1,-2*u/W +1,1]], dtype=np.float32)
    uv_1=uv_1.T
    assert(W == H)
    if (debug):
        embed()
    inv_rot = np.linalg.inv(rotation_0)
    A = np.matmul(np.linalg.inv(K[0:3,0:3]), uv_1)
    A[2] = 1
    t = np.array([translation_0])
    c = (A-t.T)
    d = inv_rot.dot(c)
    return d

def get_topdown_map(sim, map_name, selected_door_number = None, select_min= False):

    
    np.random.seed(1000)    
    semantic_scene = sim.semantic_annotations()
    instance_id_to_label_id = {int(obj.id.split("_")[-1]): obj.category.index() for obj in semantic_scene.objects}
    names = {int(obj.id.split("_")[-1]): obj.category.name() for obj in semantic_scene.objects}
    instance_label_mapping = np.array([ instance_id_to_label_id[i] for i in range(len(instance_id_to_label_id)) ])
    instance_names = np.array([names[i] for i in range(len(names))])
    candidate_doors_index = np.where(instance_names == 'door')[0]
    non_obs_candidate_doors = []
    sdfs = []
    for candidate_door in candidate_doors_index:
        chosen_object = semantic_scene.objects[candidate_door]    
        if sim.distance_to_closest_obstacle(chosen_object.aabb.center) <0.25 or sim.distance_to_closest_obstacle(chosen_object.aabb.center) >0.6:
            continue
        else:
            non_obs_candidate_doors.append(candidate_door)
            sdfs.append(sim.distance_to_closest_obstacle(chosen_object.aabb.center))
        temp_position_agent = sim.pathfinder.get_random_navigable_point_near(chosen_object.aabb.center,1.5)
        temp_position = chosen_object.aabb.center
        temp_position[1] = temp_position_agent[1]
        temp_rot = chosen_object.obb.rotation
        quat_rot =  qt.quaternion(temp_rot[3], temp_rot[0], temp_rot[1], temp_rot[2])
        quat_rot = quat_rot  *qt.quaternion(0.7071,0.0,0.0,-0.7071)
        new_quat = np.array([quat_rot.x, quat_rot.y, quat_rot.z, quat_rot.w])
        if (not np.allclose(new_quat, np.array([0,0,0,1]), atol = 0.1)) or (not np.allclose(new_quat, np.array([-0.499,0.499,0.499,0.499]), atol = 0.1)):
            continue
    sdfs = np.array(sdfs)
    candidate_doors_index = np.array(non_obs_candidate_doors)
    door_number = selected_door_number
    if select_min:
        arg_min = np.ndarray.argmin(sdfs)
        door_number = candidate_doors_index[int(arg_min)]
    
        
    chosen_object= semantic_scene.objects[door_number] 
    # target_quat = mn.Quaternion(mn.Vector3([1.0,0,0]), 0.0)
    # obj_rot = chosen_object_unrotated.obb.rotation
    # obj_quat = mn.Quaternion(mn.Vector3(obj_rot[0], obj_rot[1], obj_rot[2]), obj_rot[3])
    # obj_quat_inv = obj_quat.inverted()
    # x = target_quat*obj_quat_inv
    # chosen_object.obb = chosen_object.obb.rotate(x)
    # chosen_object.aabb = chosen_object.obb.to_aabb()
    print("sdf at chosen door is ", sim.distance_to_closest_obstacle(chosen_object.aabb.center))
    temp_position_agent = sim.pathfinder.get_random_navigable_point_near(chosen_object.aabb.center,1.5)
    temp_position = chosen_object.aabb.center
    temp_position[1] = temp_position_agent[1]
    temp_rot = chosen_object.obb.rotation
    quat_rot =  qt.quaternion(temp_rot[3], temp_rot[0], temp_rot[1], temp_rot[2])
    a =  qt.as_euler_angles(quat_rot)
    a[0] = a[2] = 0
    quat_rot = qt.from_euler_angles(a)
    # temp_rot = [-temp_rot[3], 0.0, temp_rot[1], 0.0]
    # temp_rot = temp_rot/np.linalg.norm(temp_rot)
    # quat_rot =  qt.quaternion(temp_rot[0], temp_rot[1], temp_rot[2], temp_rot[3])
    # embed()
    # quat_rot = qt.quaternion(0.7071,0.7071,0.0,0.0) * quat_rot
    new_quat = np.array([quat_rot.x, quat_rot.y, quat_rot.z, quat_rot.w])
    # new_quat = np.array([1,0,0,0])
    sim.set_agent_state(temp_position,new_quat)
    print("Door state is ", temp_position, new_quat)
    # agent_state = env.sim.get_agent_state()
    # observations = env.sim.get_sensor_observations()
    # cv2.imwrite("rgb_img_no_rot.png", np.asarray(observations['rgb']))
    # draw_agent_in_top_down(env, map_path = "agent_pos_no_rot.png")
    # agent_door = chosen_object.aabb.center - agent_state.position
    # agent_forward = utils.quat_to_magnum(
    #         env.sim.agents[0].get_state().rotation
    #     ).transform_vector(mn.Vector3(0.0,0.0,-1.0))
    # agent_door[2] = -agent_door[2]
    # agent_door[1] = 0
    # agent_forward = utils.quat_to_magnum(
    #         env.sim.agents[0].get_state().rotation
    #     ).transform_vector(mn.Vector3(agent_door[0], agent_door[1], agent_door[2]))

    # diff_quat = utils.quat_from_two_vectors(np.array(agent_forward), agent_door)
    
    # diff_quat = qt.from_rotation_vector(np.array(agent_forward))
    # diff_euler = qt.as_euler_angles(diff_quat)
    # diff_euler[0] = diff_euler[2] = 0
    # diff_quat = qt.from_euler_angles(diff_euler)
    # new_quat = np.array([diff_quat.x, diff_quat.y, diff_quat.z, diff_quat.w])
    # env.sim.set_agent_state(temp_position, new_quat)
    render_camera = sim.get_agent(0).scene_node.node_sensor_suite.get_sensors()['semantic']
    render_camera.zoom(2)
    observations = sim.get_observations_at(chosen_object.aabb.center)
    observations_semantic = np.take(instance_label_mapping, observations['semantic'])
    semantic_img = Image.new("P", (observations_semantic.shape[1], observations_semantic.shape[0]))
    semantic_img.putpalette(d3_40_colors_rgb.flatten())
    semantic_img.putdata((observations_semantic.flatten()%40).astype(np.uint8))
    semantic_img = semantic_img.convert("RGBA")
    semantic_img = np.asarray(semantic_img)
    cv2.imwrite(IMAGE_DIR+"/semantic_img.png", semantic_img)
    cv2.imwrite(IMAGE_DIR+"rgb_img.png", np.asarray(observations['rgb']))
    semantic_img = cv2.imread(IMAGE_DIR+"/semantic_img.png")
    hablab_topdown_map = maps.get_topdown_map_from_sim(
                cast("HabitatSim", sim), meters_per_pixel= 0.025
            )
    recolor_map = np.array(
        [[128, 128, 128], [255, 255, 255], [0, 0, 0]], dtype=np.uint8
    )
    hablab_topdown_map = recolor_map[hablab_topdown_map]
    new_top_down_map = hablab_topdown_map.copy()
    small_top_down_map = 255*np.ones(hablab_topdown_map.shape)
    semantic_img_camera_mat = np.array(render_camera.render_camera.camera_matrix)
    semantic_img_proj_mat = np.array(render_camera.render_camera.projection_matrix)
    grid_dimensions = (hablab_topdown_map.shape[0], hablab_topdown_map.shape[1])
    grid_points = np.ones([semantic_img.shape[0],semantic_img.shape[1],2])
    goal_points = np.ones([semantic_img.shape[0],semantic_img.shape[1],2])
    empty_image = hablab_topdown_map
    center_gt = list(maps.to_grid(chosen_object.aabb.center[2], chosen_object.aabb.center[0] , grid_dimensions, pathfinder = sim.pathfinder))
    for i in range(0,semantic_img.shape[0]):
        for j in range(0,semantic_img.shape[1]):
            world_coordinates = sem_img_to_world(semantic_img_proj_mat, semantic_img_camera_mat, semantic_img.shape[0], semantic_img.shape[1], i, j)
            [x,y] = list(maps.to_grid(world_coordinates[2], world_coordinates[0], grid_dimensions, pathfinder = sim.pathfinder))
            # print([i,j])
            # x = x - 1
            world_coordinates[1] = 0.04
            dist = np.linalg.norm(np.array(center_gt)*0.025-np.array([x,y])*0.025)
            if (dist >1.0 and dist<1.5):
                if(sim.pathfinder.is_navigable(world_coordinates)):
                    # empty_image[x,y] = [0,0,0]
                    goal_points[i,j,0] = x
                    goal_points[i,j,1] = y
            grid_points[i,j,0] = x
            grid_points[i,j,1] = y
            hablab_topdown_map[x,y] = semantic_img[i, j, 0:3]
            # if (i ==j == 360):
            #     center_gt = list(maps.to_grid(chosen_object.aabb.center[2], chosen_object.aabb.center[0] , grid_dimensions, pathfinder = sim.pathfinder,i,j))
            #     print(center_gt[0] - x, center_gt[1]-y)
    # grid_points = np.array(grid_points)
    min_x = int(np.min(grid_points[:,:,0]))
    min_y = int(np.min(grid_points[:,:,1]))
    max_x = int(np.max(grid_points[:,:,0]))
    max_y = int(np.max(grid_points[:,:,1]))
    print(min_x, min_y, max_x, max_y)
    resolution_semantic = (max_x - min_x)*0.025/semantic_img.shape[0]
    for i in np.arange(-20,semantic_img.shape[0]+20,resolution_semantic):
        for j in np.arange(-20,semantic_img.shape[1]+20,resolution_semantic):
            world_coordinates = sem_img_to_world(semantic_img_proj_mat, semantic_img_camera_mat, semantic_img.shape[0], semantic_img.shape[1], i, j)
            [x,y] = list(maps.to_grid(world_coordinates[2], world_coordinates[0], grid_dimensions, pathfinder = sim.pathfinder))
            # if (i ==j == 360):
            #     center_gt = list(maps.to_grid(chosen_object.aabb.center[2], chosen_object.aabb.center[0] , grid_dimensions, pathfinder = sim.pathfinder,))
            #     print(center_gt[0] - x, center_gt[1]-y)
            try:
                if((new_top_down_map[x,y] == [0,0,0]).all()):
                    small_top_down_map[x,y] = [0,0,0]
                # hablab_topdown_map[x,y] = semantic_img[i, j, 0:3]
            except:
                print("Not found in sem image")
                
    # max_x += int(1/resolution_semantic)
    # max_y += int(1/resolution_semantic)
    # # min_x -= int(1/resolution_semantic)
    # # min_y -= int(1/resolution_semantic)
    # range_x = np.arange(min_x, max_x)
    # range_y = np.arange(min_y, max_y)
    # line_1 = np.column_stack((np.tile(min_x, range_y.size), range_y))
    # line_2 = np.column_stack((np.tile(max_x, range_y.size), range_y))
    # line_3 = np.column_stack((range_x, np.tile(min_y, range_x.size)))
    # line_4 = np.column_stack((range_x, np.tile(max_y, range_x.size)))
    # square = np.concatenate((line_1, line_2, line_3, line_4))
    # print(np.max(square))
    # small_top_down_map[square[:,0], square[:,1],:] = [0,0,0]
    # print(min_x, min_y, max_x, max_y)
    cv2.imwrite(IMAGE_DIR+"/top_down_with_semantic_overlay.png", hablab_topdown_map)
    cv2.imwrite(IMAGE_DIR+"/small_top_down.png", small_top_down_map)
    cv2.imwrite(IMAGE_DIR+"/goal_sink.png", empty_image)
    # observations_rgb = np.take(instance_label_mapping, observations['rgb'])
    # rgb_img = Image.new("P", (observations_rgb.shape[0], observations_rgb.shape[1],3))
    # rgb_img.pudata((observations_rgb))
    observations = sim.get_sensor_observations()
    cv2.imwrite(IMAGE_DIR+"/rgb_img.png", np.asarray(observations['rgb']))
    complete_name = os.path.join(IMAGE_DIR, "image_config" + ".yaml")
    with open(IMAGE_DIR+"/cam_mat.npy", 'wb') as f:
        np.save(f,np.array(render_camera.render_camera.camera_matrix))
    with open(IMAGE_DIR+"/proj_mat.npy", 'wb') as f:
        np.save(f, np.array(render_camera.render_camera.projection_matrix))
    with open(IMAGE_DIR+"/world_to_door.npy", 'wb') as f:
        np.save(f, np.array(chosen_object.obb.world_to_local))
    f = open(complete_name, "w+")
    center = " \n- " + str(chosen_object.aabb.center[0]) + "\n- " + str(chosen_object.aabb.center[1]) + "\n- "+ str(chosen_object.aabb.center[2])
    f.write("H: " + str(semantic_img.shape[0]) + "\n")
    f.write("W: " + str(semantic_img.shape[1]) + "\n")
    f.write("camera_matrix: " + IMAGE_DIR+"/cam_mat.npy" +"\n")
    f.write("projection_matrix: " + IMAGE_DIR+"/proj_mat.npy" +"\n")
    f.write("object_id: " + str(door_number) + "\n")
    f.write("resolution: "+ str(resolution_semantic)+ "\n")
    f.write("door_center: "+ center+ "\n")
    f.write("world_to_door: " + IMAGE_DIR+"/world_to_door.npy" + "\n")
    f.close()
    agent_state = sim.get_agent_state()
    agent_pos = agent_state.position
    meters_per_pixel =0.025
    
    ### draw door in topdown map ###
    a = np.append(chosen_object.aabb.center+chosen_object.aabb.sizes/10, 1.0)
    b = np.append(chosen_object.aabb.center-chosen_object.aabb.sizes/10, 1.0)
    a[1] = chosen_object.aabb.center[1]
    b[1] = chosen_object.aabb.center[1]
    line = np.linspace(a,b,50)
    draw_agent_in_top_down(sim, map_path = "agent_pos.png", line = line)
    if (not os.path.isfile("./maps/resolution_"+scene+"_"+str(meters_per_pixel)+".pgm")): 
        meters_per_pixel =0.025
        hablab_topdown_map = maps.get_topdown_map_from_sim(
                cast("HabitatSim", sim), meters_per_pixel= meters_per_pixel
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
    return len(candidate_doors_index), door_number

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

def main():
    with open("configs/tasks/pointnav_rgbd.yaml",'r') as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        config = yaml.load(file, Loader=yaml.FullLoader)
        if (dataset == "mp3d"):
            config['DATASET']['DATA_PATH'] = "/home/catkin_ws/src/habitat_ros_interface/data/datasets/pointnav/mp3d/v1/test/content/"+scene+"0.json.gz"
        elif (dataset == "gibson"):
            config['DATASET']['DATA_PATH'] = "/home/catkin_ws/src/habitat_ros_interface/data/datasets/pointnav/gibson/v1/test/content/"+scene+"0.json.gz"
        elif (dataset == "habitat"):
            config['DATASET']['DATA_PATH'] = "/home/catkin_ws/src/habitat_ros_interface/data/datasets/pointnav/habitat-test-scenes/v1/test/content/"+scene+"0.json.gz"
    with open("configs/tasks/pointnav_rgbd.yaml",'w') as file:
        print("Replacing the data config to the new scene ", scene)
        documents = yaml.dump(config, file)
    with open("configs/tasks/pointnav_mp3d.yaml",'r') as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        config = yaml.load(file, Loader=yaml.FullLoader)
        if (dataset == "mp3d"):
            config['DATASET']['DATA_PATH'] = "/home/catkin_ws/src/habitat_ros_interface/data/datasets/pointnav/mp3d/v1/test/content/"+scene+"0.json.gz"
        elif (dataset == "gibson"):
            config['DATASET']['DATA_PATH'] = "/home/catkin_ws/src/habitat_ros_interface/data/datasets/pointnav/gibson/v1/test/content/"+scene+"0.json.gz"
        elif (dataset == "habitat"):
            config['DATASET']['DATA_PATH'] = "/home/catkin_ws/src/habitat_ros_interface/data/datasets/pointnav/habitat-test-scenes/v1/test/content/"+scene+"0.json.gz"
    
    with open("configs/tasks/pointnav_mp3d.yaml",'w') as file:
        print("Replacing the data config to the new scene ", scene)
        documents = yaml.dump(config, file)
    #first parameter is config path, second parameter is map name
    config_paths = "configs/tasks/pointnav_rgbd.yaml"
    config = habitat.get_config(config_paths=config_paths)
    eps = habitat.make_dataset(
        id_dataset=config.DATASET.TYPE, config=config.DATASET
    )
    with gzip.open(config['DATASET']['DATA_PATH'], "rb") as f:
        episode = from_json(f.read())
    eps.episodes = [episode]
    env = habitat.Env(config=config, dataset=eps)
    door_num = int(env.episodes[0].info['door_number'])
    print("Using door number")
    semantic_scene = env.sim.semantic_annotations()
    chosen_object = semantic_scene.objects[door_num] 
    a = np.append(chosen_object.aabb.center+chosen_object.aabb.sizes/10, 1.0)
    b = np.append(chosen_object.aabb.center-chosen_object.aabb.sizes/10, 1.0)
    a[1] = chosen_object.aabb.center[1]
    b[1] = chosen_object.aabb.center[1]
    line = np.linspace(a,b,50)
    temp_position, rot = env.episodes[0].start_position, env.episodes[0].start_rotation
    env.sim.set_agent_state(temp_position, rot)
    draw_agent_in_top_down(env.sim, map_path = "agent_pos_in_topdown.png", line = line)
    # if (not os.path.isfile("./maps/resolution_"+scene+"_"+str(meters_per_pixel)+".pgm")): 
    total_doors, current_door = get_topdown_map(env.sim, "resolution_"+scene+"_"+str(meters_per_pixel), selected_door_number=door_num)
   
    print("Chosen gate %d from %d doors ", current_door, total_doors)


if __name__ == "__main__":
    main()