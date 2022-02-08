# [START import]
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
# [END import]

import math
import os
import random
import sys

import imageio
import magnum as mn
import numpy as np
from matplotlib import pyplot as plt


import habitat_sim
from habitat_sim.utils import common as utils
from habitat_sim.utils import viz_utils as vut
import csv
display = True

if display:
    from habitat.utils.visualizations import maps

test_scene = "/home/trippy/Documents/Matterport_dataset/v1/tasks/mp3d/17DRP5sb8fy/17DRP5sb8fy.glb"

rgb_sensor = True  # @param {type:"boolean"}
depth_sensor = True  # @param {type:"boolean"}
semantic_sensor = True  # @param {type:"boolean"}

def display_map(topdown_map, key_points= None, region_edge_points = None, object_edge_points = None):
    plt.figure(figsize=(12, 8))
    ax = plt.subplot(1, 1, 1)
    ax.axis("off")
    plt.imshow(topdown_map)
    # plot points on map
    
    if key_points is not None:
        disp_points = key_points['points']
        navigability = key_points['navigable']
        counter = 0
        for points in disp_points:
            if navigability[counter]:
                plt.plot(points[0], points[1], marker="o", markersize=5, alpha=0.8, color = "green")
            else:
                plt.plot(points[0], points[1], marker="o", markersize=5, alpha=0.8, color = "red")
            counter = counter+1
    if region_edge_points is not None:
        x = []
        y = []
        counter = 0
        for point in region_edge_points:
            x.append(point[0])
            y.append(point[1])
            counter = counter+1
            if (counter%4 == 0):
                x.append(x[0])
                y.append(y[0])
                plt.plot(x, y, linewidth = 2, markersize = 3, color = "red")
                x = []
                y = []
    if object_edge_points is not None:
        x = []
        y = []
        counter = 0
        for point in object_edge_points:
            x.append(point[0])
            y.append(point[1])
            counter = counter+1
            if (counter%4 == 0):
                x.append(x[0])
                y.append(y[0])
                plt.plot(x, y, linewidth = 2, markersize = 3, color = "blue")
                x = []
                y = []
    plt.show(block=False)

def convert_points_to_topdown(pathfinder, points, meters_per_pixel):
    points_topdown = []
    bounds = pathfinder.get_bounds()
    for point in points:
        # convert 3D x,z to topdown x,y
        px = (point[0] - bounds[0][0]) / meters_per_pixel
        py = (point[2] - bounds[0][2]) / meters_per_pixel
        points_topdown.append(np.array([px, py]))
    return points_topdown

sim_settings = {
    "width": 256,  # Spatial resolution of the observations
    "height": 256,
    "scene": test_scene,  # Scene path
    "default_agent": 0,
    "sensor_height": 1.5,  # Height of sensors in meters
    "color_sensor": rgb_sensor,  # RGB sensor
    "depth_sensor": depth_sensor,  # Depth sensor
    "semantic_sensor": semantic_sensor,  # Semantic sensor
    "seed": 1,  # used in the random navigation
    "enable_physics": True,  # kinematics only
}



def make_cfg(settings):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    sim_cfg.scene_id = settings["scene"]
    sim_cfg.enable_physics = settings["enable_physics"]

    # Note: all sensors must have the same resolution
    sensor_specs = []

    color_sensor_spec = habitat_sim.CameraSensorSpec()
    color_sensor_spec.uuid = "color_sensor"
    color_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    color_sensor_spec.resolution = [settings["height"], settings["width"]]
    color_sensor_spec.postition = [0.0, settings["sensor_height"], 0.0]
    color_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(color_sensor_spec)

    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [settings["height"], settings["width"]]
    depth_sensor_spec.postition = [0.0, settings["sensor_height"], 0.0]
    depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(depth_sensor_spec)

    semantic_sensor_spec = habitat_sim.CameraSensorSpec()
    semantic_sensor_spec.uuid = "semantic_sensor"
    semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    semantic_sensor_spec.resolution = [settings["height"], settings["width"]]
    semantic_sensor_spec.postition = [0.0, settings["sensor_height"], 0.0]
    semantic_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(semantic_sensor_spec)

    # Here you can specify the amount of displacement in a forward action and the turn angle
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=30.0)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=30.0)
        ),
    }

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])

cfg = make_cfg(sim_settings)
# Needed to handle out of order cell run in Colab
try:  # Got to make initialization idiot proof
    sim.close()
except NameError:
    pass
sim = habitat_sim.Simulator(cfg)

chosen_points = []  
map_points_3d = [] 
final_map_points_3d = []     
navigable = [] 
xy_vis_points = {}
scene = sim.semantic_scene
floor_y = 0.0
print(floor_y)
top_down_map = maps.get_topdown_map(
    sim.pathfinder, height=floor_y, meters_per_pixel=0.05
)
recolor_map = np.array(
    [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
)

top_down_map = recolor_map[top_down_map]
grid_dimensions = (top_down_map.shape[0], top_down_map.shape[1])
print(grid_dimensions)
all_points = []
# for i in range(1,grid_dimensions[0]):
#     for j in range(1,grid_dimensions[1]):
#         all_points.append([i,j])

# with open('my_src/all_points.csv',  mode='w') as csvfile:
#     csv_writer = csv.writer(csvfile, delimiter=',')
#     for points in all_points:
#         csv_writer.writerow(points)
    # csvfile.close()
# agent = sim.initialize_agent(sim_settings["default_agent"])
# agent_state = habitat_sim.AgentState()
with open('my_src/handpicked_points.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            map_points = maps.from_grid(
                        int(float(row[1])),
                        int(float(row[0])),
                        grid_dimensions,
                        pathfinder=sim.pathfinder,
                    )
            map_points_3d = np.array([map_points[1], floor_y, map_points[0]])
            # # agent_state.position = np.array(map_points_3d)  # in world space
            # # agent.set_state(agent_state)
            # map_points_3d = sim.pathfinder.snap_point(map_points_3d)
            
            
            if(sim.pathfinder.is_navigable(map_points_3d)):
            #     print(row)
            #     chosen_points.append(list(map(float,row)))
            #     final_map_points_3d.append(map_points_3d)
            #     navigable.append(sim.pathfinder.is_navigable(map_points_3d))
                final_map_points_3d.append(map_points_3d)
                navigable.append(sim.pathfinder.is_navigable(map_points_3d))
grid_points = []
for points in final_map_points_3d:
    grid_points_back = maps.to_grid(
                        points[2],
                        points[0],
                        grid_dimensions,
                        pathfinder=sim.pathfinder,
                    )
    grid_points.append([grid_points_back[1], grid_points_back[0]])

distance_matrix = []
for start_point in final_map_points_3d:
    distances = []
    for end_point in final_map_points_3d:
        path = habitat_sim.ShortestPath()
        path.requested_start = start_point
        path.requested_end = end_point
        found_path = False
        print(start_point, end_point)
        counter = 0
        while(found_path==False and counter<30):
            counter = counter+1
            found_path = sim.pathfinder.find_path(path)
        geodesic_distance = path.geodesic_distance
        distances.append(geodesic_distance)
    distance_matrix.append(distances)
print(distance_matrix)

with open('my_src/distance_matrix.csv', mode='w') as csvfile:
    csv_writer = csv.writer(csvfile, delimiter=',')
    for distances in distance_matrix:
            csv_writer.writerow(distances)

# with open('my_src/handpicked_points_3d.csv', mode='w') as csvfile:
#     csv_writer = csv.writer(csvfile, delimiter=',')
#     for points in final_map_points_3d:
#             csv_writer.writerow(points)

xy_vis_points['points'] = convert_points_to_topdown(
                sim.pathfinder, final_map_points_3d, meters_per_pixel=0.05
            )

# for points in chosen_points:
#     xy_vis_points['points'].append(points)
#     navigable.append(False)
# xy_vis_points['points'] = grid_points
xy_vis_points['navigable'] = navigable

display_map(top_down_map, key_points = xy_vis_points)
input("Press Enter to exit")