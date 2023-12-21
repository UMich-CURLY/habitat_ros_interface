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
import habitat_sim.nav as habitat_path

import tqdm
from habitat_sim.utils import common as utils
import magnum as mn
import habitat
import habitat_sim
from habitat.datasets.pointnav.pointnav_generator import generate_pointnav_episode
from habitat.tasks.nav.nav import (
    NavigationEpisode,
    NavigationGoal,
    ShortestPathPoint,
)
import argparse
import numpy as np
import quaternion as qt
from IPython import embed
from get_topdown_map_rl import draw_agent_in_top_down
from get_trajectory import *
from get_trajectory_rvo import *
PARSER = argparse.ArgumentParser(description=None)
PARSER.add_argument('-s', '--scene', default="17DRP5sb8fy", type=str, help='scene')
PARSER.add_argument('-d', '--dataset', default="mp3d", type=str, help='dataset')

ARGS = PARSER.parse_args()
scene = ARGS.scene
dataset = ARGS.dataset
num_episodes_per_scene = 1
GOAL_BAND = (1.0, 1.5)
IMG_OUT_PATH = "data/datasets/pointnav/mp3d/v1/test/images/"+scene
if not os.path.exists(IMG_OUT_PATH):
    print("Didi not find maps directory")
    os.makedirs(IMG_OUT_PATH)
'''
Create N Number of episodes for each scene. Specify the correct scene in arguments
'''
class pointnav_data():
    def __init__(self):
        cfg = habitat.get_config()
        cfg.defrost()
        if (dataset == "mp3d"):
            cfg.SIMULATOR.SCENE = "/home/catkin_ws/src/habitat_ros_interface/data/scene_datasets/mp3d/"+scene+"/"+scene+".glb"
        elif(dataset == "gibson"):
            cfg.SIMULATOR.SCENE = "/home/catkin_ws/src/habitat_ros_interface/data/scene_datasets/gibson/"+scene+".glb"
        elif (dataset == "habitat"):
            cfg.SIMULATOR.SCENE = "/home/catkin_ws/src/habitat_ros_interface/data/scene_datasets/habitat-test-scenes/"+scene+".glb"
        else:
            print("No dataset found")
            exit(0)
        cfg.SIMULATOR.AGENT_0.SENSORS = []
        cfg.freeze()

        self.sim = habitat.sims.make_sim("Sim-v0", config=cfg.SIMULATOR)

        self.dset = habitat.datasets.make_dataset("PointNav-v1")
        self.dset.episodes = list(
            generate_pointnav_episode(
                self.sim, num_episodes_per_scene, is_gen_shortest_path=False
            )
        )

    def get_start_goal(self, selected_door_number = None, select_min= False):
        semantic_scene = self.sim.semantic_annotations()
        instance_id_to_label_id = {int(obj.id.split("_")[-1]): obj.category.index() for obj in semantic_scene.objects}
        names = {int(obj.id.split("_")[-1]): obj.category.name() for obj in semantic_scene.objects}
        instance_label_mapping = np.array([ instance_id_to_label_id[i] for i in range(len(instance_id_to_label_id)) ])
        instance_names = np.array([names[i] for i in range(len(names))])
        candidate_doors_index = np.where(instance_names == 'door')[0]
        non_obs_candidate_doors = []
        sdfs = []
        for candidate_door in candidate_doors_index:
            chosen_object = semantic_scene.objects[candidate_door]
            dist = self.sim.distance_to_closest_obstacle(chosen_object.aabb.center) 
            if dist <0.3 or dist>0.8:
                continue
            # if (not np.allclose(goal_rot, np.array([0,0,0,1]), atol = 0.1)) or (not np.allclose(goal_rot, np.array([-0.499,0.499,0.499,0.499]), atol = 0.1)):
            #     embed()
            temp_rot = chosen_object.obb.rotation
            quat_rot =  qt.quaternion(temp_rot[3], temp_rot[0], temp_rot[1], temp_rot[2])
            a =  qt.as_euler_angles(quat_rot)
            if (int(abs(a[1]*180/3.1415))%90 > 3):
                continue
            else:
                non_obs_candidate_doors.append(candidate_door)
                sdfs.append(self.sim.distance_to_closest_obstacle(chosen_object.aabb.center))
        sdfs = np.array(sdfs)
        candidate_doors_index = np.array(non_obs_candidate_doors)
        if not selected_door_number:
            door_number = np.random.choice(candidate_doors_index)
        else:
            door_number = candidate_doors_index[selected_door_number]
        if select_min:
            arg_min = np.ndarray.argmin(sdfs)
            door_number = candidate_doors_index[int(arg_min)]
        
        self.chosen_object = semantic_scene.objects[door_number]    
        # target_quat = mn.Quaternion(mn.Vector3([1.0,0,0]), 0.0)
        # obj_rot = chosen_object.obb.rotation
        # obj_quat = mn.Quaternion(mn.Vector3(obj_rot[0], obj_rot[1], obj_rot[2]), obj_rot[3])
        # obj_quat_inv = obj_quat.inverted()
        # x = target_quat*obj_quat_inv
        # self.chosen_object = {'obb': None, 'aabb':None}
        # embed()
        # self.chosen_object.obb = chosen_object.obb.rotate(x)
        # self.chosen_object.aabb = chosen_object.obb.to_aabb()
        # temp_rot = chosen_object.obb.rotation
        # quat_rot =  qt.quaternion(temp_rot[3], temp_rot[0], temp_rot[1], temp_rot[2])
        # quat_rot = quat_rot  *qt.quaternion(0.7071, 0, 0, -0.7071)
        # new_quat = np.array([quat_rot.x, quat_rot.y, quat_rot.z, quat_rot.w])
        agent_state = self.sim.get_agent_state(0)
        chosen_object = semantic_scene.objects[door_number]   
        goes_through_door = False
        path_dist = 0
        max_tries = 200
        try_num = 0
        is_same_floor = False
        while (not goes_through_door or path_dist<1.0 or not is_same_floor):
            try_num+=1
            temp_position, rot = self.get_in_band_around_door(agent_state.rotation)
            self.sim.set_agent_state(temp_position, rot)
            agent_pos = self.sim.get_agent_state(0).position
            agent_state = self.sim.get_agent_state(0)
            start_pos = [agent_pos[0], agent_pos[1], agent_pos[2]]
            
            ## Asume the agent goal is always the goal of the 0th agent
            path = habitat_path.ShortestPath()
            path.requested_start = np.array(start_pos)
            agent_goal_pos_3d, goal_rot = self.get_in_band_around_door(agent_state.rotation)
            if (try_num > max_tries):
                path.requested_end = agent_goal_pos_3d
                if(not self.sim.pathfinder.find_path(path)):
                    print("Didn't find path")
                path_dist = path.geodesic_distance
                print("No solution found, try a different door ")
                door_number = -1
                break
            if (not self.is_point_on_other_side(agent_goal_pos_3d, agent_state.position)):
                continue
            else:
                path.requested_end = agent_goal_pos_3d
            
            if(not self.sim.pathfinder.find_path(path)):
                print("Didn't find path")
                embed()
                continue       
            goes_through_door = self.check_path_goes_through_door(path)
            path_dist = path.geodesic_distance
            is_same_floor = self.is_same_floor(agent_goal_pos_3d, agent_state.position)
        print(path.points)
        return start_pos, path.points[-1], path.geodesic_distance, door_number

    def get_in_band_around_door(self, agent_rotation = None):
        temp_position = self.sim.pathfinder.get_random_navigable_point_near(self.chosen_object.aabb.center,5)
        diff_vec = temp_position - self.chosen_object.aabb.center
        diff_vec[1] = 0
        temp_dist = np.linalg.norm(diff_vec)
        while (temp_dist < 1.0 or temp_dist >1.5):
            temp_position = self.sim.pathfinder.get_random_navigable_point_near(self.chosen_object.aabb.center,5)
            diff_vec = temp_position - self.chosen_object.aabb.center
            diff_vec[1] = 0
            temp_dist = np.linalg.norm(diff_vec)
        
        if (agent_rotation is not None):
            agent_door = self.chosen_object.aabb.center - temp_position
            agent_door[2] = -agent_door[2]
            agent_door[1] = 0
            agent_forward = utils.quat_to_magnum(
                    agent_rotation
                ).transform_vector(mn.Vector3(agent_door[0], agent_door[1], agent_door[2]))        
            diff_quat = qt.from_rotation_vector(np.array(agent_forward))
            diff_euler = qt.as_euler_angles(diff_quat)
            diff_euler[0] = diff_euler[2] = 0
            diff_quat = qt.from_euler_angles(diff_euler)
            new_quat = np.array([diff_quat.x, diff_quat.y, diff_quat.z, diff_quat.w])
            return temp_position, new_quat
        else:
            return temp_position, None
    
    def check_path_goes_through_door(self, path):
        diff = path.points[0]-self.chosen_object.aabb.center
        diff[1] = 0
        dist = np.linalg.norm(diff)
        initial_dist = dist
        dist_from_door = 100
        for point in path.points:
            diff = point-self.chosen_object.aabb.center
            diff[1] = 0
            dist = np.linalg.norm(diff)
            if dist <dist_from_door:
                dist_from_door = dist
        if (dist_from_door) < initial_dist:
            print("Path dist from door and starting is, All good!", dist_from_door, initial_dist)
            return True
        else:
            print("Path dist from door and starting is", dist_from_door, initial_dist)
            return False
    def is_point_on_other_side(self, p1, p2):
        transform = self.chosen_object.obb.world_to_local
        size = np.array(self.chosen_object.aabb.sizes)
        p1_local = np.matmul(transform, np.append(p1,1.0).T)
        p2_local = np.matmul(transform, np.append(p2,1.0).T)
        p_size = np.matmul(transform, np.append(size,1.0).T)
        y1 = p1_local[2]
        y2 = p2_local[2]
        x1 = p1_local[1]
        x2 = p2_local[1]
        print(size)
        x_size = np.max([size[0], size[2]])

        if (np.sign(y1) == np.sign(y2) or abs(y1)<5 or abs(y2) <5 or abs(x1)>x_size or abs(x2)>x_size):
            return False
        else:
            return True
    def is_in_same_region(self, pos):
        region = self.chosen_object.region
        agent_loc = pos

        ''' Check if agent is in region 0 '''
        center = region.aabb.center
        sizes = region.aabb.sizes
        if ((center - 1/2*sizes < agent_loc).all()  and (agent_loc < center + 1/2*sizes).all()):
            return True
        else:
            return False

    def is_same_floor(self, p1, p2):
        if abs(p1[1]-p2[1])>1.0:
            return False
        else:
            return True
    def _generate_fn(self, scene, dataset, start = None, goal = None):
        
        count_episodes = 0
        for ep in self.dset.episodes:
            door = -1 
            max_tries = 20
            try_num = 0
            
            while (door == -1):
                start, goal, dist, door = self.get_start_goal()
                try_num +=1
                if (try_num>max_tries):
                    print("This scene does not have the right setup for the door scenario")
                    return False
            ep.start_position = np.float64(start)
            ep.goals = [NavigationGoal(position = goal, radius = 0.2)]
            ep.info={"geodesic_distance": dist, "door_number": np.float64(door)}
            semantic_scene = self.sim.semantic_annotations()
            chosen_object = semantic_scene.objects[int(door)] 
            a = np.append(chosen_object.aabb.center+chosen_object.aabb.sizes/2, 1.0)
            b = np.append(chosen_object.aabb.center-chosen_object.aabb.sizes/2, 1.0)
            a[1] = chosen_object.aabb.center[1]
            b[1] = chosen_object.aabb.center[1]
            line = np.linspace(a,b,50)
            # ep.extra_info = {}
        if (dataset == "mp3d"):
            for ep in self.dset.episodes:
                ep.scene_id = "/home/catkin_ws/src/habitat_ros_interface/data/scene_datasets/mp3d/"+scene+"/"+scene+".glb"
                ep.scene_dataset_config = "/home/catkin_ws/src/habitat_ros_interface/data/scene_datasets/mp3d/mp3d.scene_dataset_config.json"
            print(self.dset.episodes)
            scene_key = osp.basename(osp.dirname(osp.dirname(scene)))
            out_file = f"/home/catkin_ws/src/habitat_ros_interface/data/datasets/pointnav/mp3d/v1/test/content/"+scene + str(count_episodes)+".json.gz"
            os.makedirs(osp.dirname(out_file), exist_ok=True)
            with gzip.open(out_file, "wt") as f:
                f.write(self.dset.to_json())
        elif (dataset == "gibson"):
            for ep in self.dset.episodes:
                ep.scene_id = "/home/catkin_ws/src/habitat_ros_interface/data/scene_datasets/gibson/"+scene+".glb"
            print(self.dset.episodes)
            scene_key = osp.basename(osp.dirname(osp.dirname(scene)))
            out_file = f"./data/datasets/pointnav/gibson/v1/test/content/"+scene + str(count_episodes)+".json.gz"
            os.makedirs(osp.dirname(out_file), exist_ok=True)
            with gzip.open(out_file, "wt") as f:
                f.write(self.dset.to_json())
        elif (dataset == "habitat"):
            for ep in self.dset.episodes:
                ep.scene_id = "data/scene_datasets/habitat-test-scenes/"+scene+".glb"
            print(self.dset.episodes)
            scene_key = osp.basename(osp.dirname(osp.dirname(scene)))
            out_file = f"./data/datasets/pointnav/habitat-test-scenes/v1/test/content/"+scene + str(count_episodes)+".json.gz"
            os.makedirs(osp.dirname(out_file), exist_ok=True)
            with gzip.open(out_file, "wt") as f:
                f.write(self.dset.to_json())
        else:
            print("No dataset found")
            return False
        
        draw_agent_in_top_down(self.sim, map_path = IMG_OUT_PATH+"/"+scene+"_ep.png", line = line, goal = goal)
        # points = []
        # for i in range(1000):
        #     point, a = self.get_in_band_around_door()
        #     if (self.is_point_on_other_side(point, start)):
        #         points.append(point)
        # print("Length of points is ", len(points))
        # draw_agent_in_top_down(self.sim, map_path = "door_area.png", line = line, goal = goal, points_3d = points)
        return True



# scenes = glob.glob("./data/scene_datasets/mp3d/17DRP5sb8fy.glb")
# with multiprocessing.Pool(8) as pool, tqdm.tqdm(total=len(scenes)) as pbar:
#     for _ in pool.imap_unordered(_generate_fn, scenes):
#         pbar.update()

# with gzip.open(f"./data/datasets/pointnav/mp3d/v1/all/all.json.gz", "wt") as f:
#     json.dump(dict(episodes=[]), f)

if __name__ == "__main__":
    scene_data = pointnav_data()
    success = scene_data._generate_fn(scene=scene, dataset = dataset)
