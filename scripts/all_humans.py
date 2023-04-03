 #!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.



from habitat.utils.geometry_utils import quaternion_rotate_vector
from habitat.tasks.utils import cartesian_to_polar
from habitat_sim.utils import common as utils
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
import habitat_sim.nav as habitat_path
import habitat
import habitat_sim.bindings as hsim
import magnum as mn
import quaternion as qt
from habitat.utils.visualizations import maps
import habitat_sim
import numpy as np
import time
import random
# import cv2
import sys
sys.path.append("/root/miniconda3/envs/robostackenv/lib/python3.9/site-packages")
sys.path.append("/opt/conda/envs/robostackenv/lib/python3.9/site-packages")
import rospy
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
from std_msgs.msg import Bool
from geometry_msgs.msg import Twist, TransformStamped
from geometry_msgs.msg import PointStamped, PoseStamped, PoseWithCovarianceStamped
from visualization_msgs.msg import Marker, MarkerArray
import threading
import tf
from tour_planner_dropped import tour_planner
import csv
from move_base_msgs.msg import MoveBaseActionResult
from matplotlib import pyplot as plt
from IPython import embed
from nav_msgs.srv import GetPlan
from get_trajectory import *
from get_trajectory_rvo import *

import tf2_ros
lock = threading.Lock()
rospy.init_node("robot_1", anonymous=False)
import argparse
PARSER = argparse.ArgumentParser(description=None)
PARSER.add_argument('-s', '--scene', default="17DRP5sb8fy", type=str, help='scene')
ARGS = PARSER.parse_args()
scene = ARGS.scene
AGENT_GOAL_POS_2d = [800,500]
AGENT_START_POS_2d = [800,100]
FOLLOWER_OFFSET = [1.5,-1.0,0.0]
AGENTS_SPEED = 0.5
USE_RVO = False
MPP = 0.001
def convert_points_to_topdown(pathfinder, points, meters_per_pixel = 0.025):
    points_topdown = []
    bounds = pathfinder.get_bounds()
    for point in points:
        # convert 3D x,z to topdown x,y
        px = (point[0] - bounds[0][0]) / meters_per_pixel
        py = (point[2] - bounds[0][2]) / meters_per_pixel
        points_topdown.append(np.array([px, py]))
    return points_topdown

def quat_to_coeff(quat):
        quaternion = [quat.x, quat.y, quat.z, quat.w]
        return quaternion

def to_grid(pathfinder, points, grid_dimensions):
    map_points = maps.to_grid(
                        points[2],
                        points[0],
                        grid_dimensions,
                        pathfinder=pathfinder,
                    )
    return ([map_points[1], map_points[0]])

def from_grid(pathfinder, points, grid_dimensions):
    floor_y = 0.0
    map_points = maps.from_grid(
                        points[1],
                        points[0],
                        grid_dimensions,
                        pathfinder=pathfinder,
                    )
    map_points_3d = np.array([map_points[1], floor_y, map_points[0]])
    # # agent_state.position = np.array(map_points_3d)  # in world space
    # # agent.set_state(agent_state)
    map_points_3d = pathfinder.snap_point(map_points_3d)
    return map_points_3d

def quat_from_two_vectors(v0: np.ndarray, v1: np.ndarray) -> qt.quaternion:
    r"""Creates a quaternion that rotates the first vector onto the second vector

    :param v0: The starting vector, does not need to be a unit vector
    :param v1: The end vector, does not need to be a unit vector
    :return: The quaternion

    Calculates the quaternion q such that

    .. code:: py

        v1 = quat_rotate_vector(q, v0)
    """

    v0 = v0 / np.linalg.norm(v0)
    v1 = v1 / np.linalg.norm(v1)
    c = v0[0]*v1[0]+v0[1]*v1[1]+v0[2]*v1[2]
    if c < (-1 + 1e-8):
        c = max(c, -1)
        m = np.stack([v0, v1], 0)
        _, _, vh = np.linalg.svd(m, full_matrices=True)
        axis = vh[2]
        w2 = (1 + c) * 0.5
        w = np.sqrt(w2)
        axis = axis * np.sqrt(1 - w2)
        return qt.quaternion(w, *axis)

    axis = np.cross(v0, v1)
    s = np.sqrt((1 + c) * 2)
    return qt.quaternion(s * 0.5, *(axis / s))

def draw_axes(sim, translation, axis_len=1.0):
    lr = sim.get_debug_line_render()
    # draw axes with x+ = red, y+ = green, z+ = blue
    opacity = 1.0
    red = mn.Color4(1.0, 0.0, 0.0, opacity)
    green = mn.Color4(0.0, 1.0, 0.0, opacity)
    blue = mn.Color4(0.0, 0.0, 1.0, opacity)
    white = mn.Color4(1.0, 1.0, 1.0, opacity)
    lr.draw_transformed_line(translation, mn.Vector3(axis_len, 0, 0), red)
    lr.draw_transformed_line(translation, mn.Vector3(0, axis_len, 0), green)
    lr.draw_transformed_line(translation, mn.Vector3(0, 0, axis_len), blue)
# Set an object transform relative to the agent state
def set_object_state_from_agent(
    sim,
    obj,
    offset=np.array([0, 2.0, -1.5]),
    orientation=mn.Quaternion(((0, 0, 0), 1)),
    ):
    agent_transform = sim.agents[0].scene_node.transformation_matrix()
    ob_translation = agent_transform.transform_point(offset)
    obj.translation = ob_translation
    obj.rotation = orientation

def map_to_base_link(msg, my_env):
    grid_x,grid_y = my_env.grid_dimensions
    theta = msg['theta']
    my_env.br.sendTransform((-my_env.initial_state[0][0]+1, -my_env.initial_state[0][1]+1,0.0),
                    tf.transformations.quaternion_from_euler(0, 0, 0.0),
                    rospy.Time.now(),
                    "decision_frame",
                    "interim_link"
    )
    my_env.br.sendTransform((0.0,0.0,0.0),
                    tf.transformations.quaternion_from_euler(0, 0, -theta),
                    rospy.Time.now(),
                    "interim_link",
                    "base_link"
    )
    poseMsg = PoseStamped()
    poseMsg.header.stamp = rospy.Time.now()
    poseMsg.header.frame_id = "base_link"
    quat = tf.transformations.quaternion_from_euler(0, 0, 0.0)
    poseMsg.pose.orientation.x = quat[0]
    poseMsg.pose.orientation.y = quat[1]
    poseMsg.pose.orientation.z = quat[2]
    poseMsg.pose.orientation.w = quat[3]
    poseMsg.header.stamp = rospy.Time.now()
    poseMsg.pose.position.x = 0.0
    poseMsg.pose.position.y = 0.0
    poseMsg.pose.position.z = 0.0
    my_env._robot_pose.publish(poseMsg)
    follower_pos = my_env.follower.rigid_state.translation
    follower_pose_2d = to_grid(my_env.env._sim.pathfinder, follower_pos, my_env.grid_dimensions)
    follower_pose_2d = follower_pose_2d*(0.025*np.ones([1,2]))[0]
    poseMsg.header.frame_id = "decision_frame"
    poseMsg.pose.orientation.x = 0.0
    poseMsg.pose.orientation.y = 0.0
    poseMsg.pose.orientation.z = 0.0
    poseMsg.pose.orientation.w = 1.0
    poseMsg.header.stamp = rospy.Time.now()
    poseMsg.pose.position.x = my_env.initial_state[1][0]-1
    poseMsg.pose.position.y = my_env.initial_state[1][1]-1
    poseMsg.pose.position.z = 0.0
    my_env._pub_follower.publish(poseMsg)
    goal_marker = Marker()
    goal_marker.header.frame_id = "decision_frame"
    goal_marker.type = 2
    goal_marker.pose.position.x = my_env.initial_state[0][4]-1
    goal_marker.pose.position.y = my_env.initial_state[0][5]-1
    goal_marker.pose.position.z = 0.0
    goal_marker.pose.orientation.x = 0.0
    goal_marker.pose.orientation.y = 0.0
    goal_marker.pose.orientation.z = 0.0
    goal_marker.pose.orientation.w = 1.0
    goal_marker.scale.x = 0.5
    goal_marker.scale.y = 0.5
    goal_marker.scale.z = 0.5
    goal_marker.color.a = 1.0 
    goal_marker.color.r = 0.0
    goal_marker.color.g = 1.0
    goal_marker.color.b = 0.0
    my_env._pub_goal_marker.publish(goal_marker)

class sim_env(threading.Thread):
    _x_axis = 0
    _y_axis = 1
    _z_axis = 2
    _dt = 0.00478
    _sensor_rate = 50  # hz
    _r_sensor = rospy.Rate(_sensor_rate)
    _current_episode = 0
    sensor_time_step = 1/_sensor_rate
    _total_number_of_episodes = 0
    _nodes = []
    _global_plan_published = False
    current_goal = []
    current_position = []
    current_orientation = []
    flag_action_uncertainty = True
    flag_wait_time_uncertainty = True
    action_uncertainty_rate = 0.9
    follower = []
    new_goal = False
    control_frequency = 10
    time_step = 1.0 / (control_frequency)
    _r_control = rospy.Rate(control_frequency)
    human_control_frequency = 10
    human_time_step = 1/human_control_frequency
    linear_velocity = np.array([0.0,0.0,0.0])
    angular_velocity = np.array([0.0,0.0,0.0])
    received_vel = False
    goal_reached = False
    start_time = []
    tour_planning_time = []
    all_points = []
    rtab_pose = []
    goal_time = []
    obs = []
    update_counter = 0
    human_update_counter = 0 
    agent_update_counter = 0
    update_multiple = human_time_step/time_step
    def __init__(self, env_config_file):
        ##### Checking the git branch stuff
        threading.Thread.__init__(self)
        self.env_config_file = env_config_file
        self.env = habitat.Env(config=habitat.get_config(self.env_config_file))
        self.env._sim.robot.params.arm_init_params = [1.32, 1.40, -0.2, 1.72, 0.0, 1.66, 0.0]
        floor_y = 0.0
        top_down_map = maps.get_topdown_map(
            self.env._sim.pathfinder, height=floor_y, meters_per_pixel=0.025
        )
        self.grid_dimensions = (top_down_map.shape[0], top_down_map.shape[1])
        print("Grid size is ", self.grid_dimensions)
        print("Initializeed environment")
        # always assume height equals width
        # self.env._sim.agents[0].move_filter_fn = self.env._sim.step_filter
        agent_state = self.env.sim.get_agent_state(0)
        self.observations = self.env.reset()
        arm_joint_positions  = [1.32, 1.40, -0.2, 1.72, 0.0, 1.66, 0.0]
        self.env._sim.robot.arm_joint_pos = arm_joint_positions
        temp_position = self.env._sim.pathfinder.get_random_navigable_point()
        island_radius = 2.0
        temp_island_radius = 2.0
        for i in range(50):
            new_temp_position = self.env._sim.pathfinder.get_random_navigable_point()
            [y,x] = np.array(to_grid(self.env._sim.pathfinder, new_temp_position, self.grid_dimensions))
            if top_down_map[x][y] == 0:
                continue
            temp_island_radius = self.env._sim.pathfinder.island_radius(new_temp_position)
            if island_radius < temp_island_radius:
                temp_position = new_temp_position
                island_radius = temp_island_radius
                print("Found better one")
        if island_radius<5.0:
            print("Island radius is ", island_radius)
            embed()
        
        agent_state.position = temp_position
        self.lr = self.env._sim.get_debug_line_render()
        self.lr.set_line_width(3)

        self.env.sim.set_agent_state(agent_state.position, agent_state.rotation)
        self.env._sim.robot.base_pos = mn.Vector3(agent_state.position)
        print(self.env.sim)
        config = self.env._sim.config
        print(self.env._sim.active_dataset)
        self._sensor_resolution = {
            "RGB": 720,  
            "DEPTH": 720,
        }
        print(self.env._sim.pathfinder.get_bounds())
        
        # agent_init_pos = np.array(from_grid(self.env._sim.pathfinder, AGENT_START_POS_2d, self.grid_dimensions))
        # agent_state.position = agent_init_pos
        self.env.sim.set_agent_state(agent_state.position, agent_state.rotation)
        self.env._sim.robot.base_pos = mn.Vector3(agent_state.position)
        self._pub_rgb = rospy.Publisher("~rgb", numpy_msg(Floats), queue_size=1)
        self._pub_depth = rospy.Publisher("~depth", numpy_msg(Floats), queue_size=1)
        self._robot_pose = rospy.Publisher("~robot_pose", PoseStamped, queue_size = 1)
        self._pub_follower = rospy.Publisher("~follower_pose", PoseStamped, queue_size = 1)
        self._pub_goal_marker = rospy.Publisher("~goal", Marker, queue_size = 1)
        
        self.br = tf.TransformBroadcaster()
        # self._pub_pose = rospy.Publisher("~pose", PoseStamped, queue_size=1)
        rospy.Subscriber("~plan_3d", numpy_msg(Floats),self.plan_callback, queue_size=1)
        rospy.Subscriber("/rtabmap/goal_reached", Bool,self.update_goal_status, queue_size=1)
        rospy.Subscriber("/move_base/result", MoveBaseActionResult ,self.update_move_base_goal_status, queue_size=1)
        rospy.Subscriber("/rtabmap/localization_pose", PoseWithCovarianceStamped,self.rtabpose_callback,queue_size=1)    
        # rospy.Subscriber("/move_base_simple/goal", PoseStamped,self.landmark_callback,queue_size=1)    
        # self.pub_goal = rospy.Publisher("~current_goal", PoseStamped, queue_size=1)
        self.pub_goal = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=1)
        rospy.Subscriber("/clicked_point", PointStamped,self.point_callback, queue_size=1)
        # goal_radius = self.env.episodes[0].goals[0].radius
        # if goal_radius is None:
        #     goal_radius = config.SIMULATOR.FORWARD_STEP_SIZE
        # self.follower = ShortestPathFollower(
        #     self.env.sim, goal_radius, False
        # )
        self.env._sim.enable_physics = True
        # self.tour_plan = tour_planner()
        print("before initialized object")
        
            
        global rigid_obj_mgr
        rigid_obj_mgr = self.env._sim.get_rigid_object_manager()
        global obj_template_mgr
        obj_template_mgr = self.env._sim.get_object_template_manager()
        rigid_obj_mgr.remove_all_objects()

        config=habitat.get_config(self.env_config_file)
        
        robot_pos_in_2d = to_grid(self.env._sim.pathfinder, self.env._sim.robot.base_pos, self.grid_dimensions)
        print(robot_pos_in_2d)
        ### Add human objects and groups here! 
        ### N has the total number of extra humans, besides the robot and the two followers
        self.N = 5
        
        self.groups = [[0,1,2], [3], [4,5],[6,7]]


        ##### Initiating objects for other humans #####

        self.human_template_ids = []
        self.objs = []
        self.vel_control_objs = []
        self.initial_state = []
        humans_initial_pos_3d = []
        humans_goal_pos_3d = []
        humans_initial_velocity = []
        ##### Final 3d goals for the agent and the extra agents, the leader and followers just follow the agent 
        self.final_goals_3d = np.zeros([self.N+1,3])
        self.goal_dist = np.zeros(self.N+3)


        #### Initiating robot in the esfm state ####
        agent_pos = self.env.sim.robot.base_pos
        start_pos = [agent_pos[0], agent_pos[1], agent_pos[2]]
        ## Asume the agent goal is always the goal of the 0th agent
        agent_goal_pos_3d = np.array(from_grid(self.env._sim.pathfinder, AGENT_GOAL_POS_2d, self.grid_dimensions))
        goal_distance = 0.5
        path = habitat_path.ShortestPath()
        path.requested_start = np.array(start_pos)
        for i in range(50):
            goal = self.env._sim.pathfinder.get_random_navigable_point()

            temp_goal_dist = np.sqrt((goal[0]-start_pos[0])**2 + (goal[2]-start_pos[2])**2)
            path.requested_end = goal
            if(not self.env._sim.pathfinder.find_path(path)):
                continue
            if temp_goal_dist > goal_distance:
                agent_goal_pos_3d = goal
                goal_distance = temp_goal_dist
        if (goal_distance<10):
            print("chose another scene maybe!", goal_distance)
            embed()
        path.requested_end = agent_goal_pos_3d
        self.final_goals_3d[0,:] = agent_goal_pos_3d
        if(not self.env._sim.pathfinder.find_path(path)):
            print("Watch this one Tribhi!!!!")
            embed()
        agents_initial_pos_3d =[]
        agents_goal_pos_3d = []
        agents_initial_pos_3d.append(path.points[0])
        agents_goal_pos_3d.append(path.points[1])
        sphere_template_id = obj_template_mgr.load_configs('./scripts/sphere')[0]
        file_obj = rigid_obj_mgr.add_object_by_template_id(sphere_template_id)
        # self.objs.append(obj)
        
        # obj_template_handle = './scripts/sphere.object_config.json'
        # obj_template = obj_template_mgr.get_template_by_handle(obj_template_handle)
        # print(obj_template)
        # file_obj = rigid_obj_mgr.add_object_by_template_handle(obj_template_handle) 
        # print(file_obj)
        file_obj.motion_type = habitat_sim.physics.MotionType.STATIC
        sphere_pos = agents_goal_pos_3d[-1]
        file_obj.translation = mn.Vector3(sphere_pos[0],sphere_pos[1], sphere_pos[2])
        # sphere_offset = file_obj.translation - agent_state.position
        # set_object_state_from_agent(self.env._sim, file_obj, np.array(sphere_offset - sphere), orientation = object_orientation2)
        agents_initial_velocity = [0.5,0.0]
        initial_pos = list(to_grid(self.env._sim.pathfinder, agents_initial_pos_3d[0], self.grid_dimensions))
        initial_pos = [pos*0.025 for pos in initial_pos]
        
        goal_pos = list(to_grid(self.env._sim.pathfinder, agents_goal_pos_3d[0], self.grid_dimensions))
        goal_pos = [pos*0.025 for pos in goal_pos]
        self.initial_state.append(initial_pos+agents_initial_velocity+goal_pos)
        self.goal_dist[0] = np.linalg.norm((np.array(self.initial_state[0][0:2])-np.array(self.initial_state[0][4:6])))
        ##### Human 1 being initiated #####
        human_template_id = obj_template_mgr.load_configs('./scripts/humantwo')[0]
        self.leader_id = human_template_id
        file_obj = rigid_obj_mgr.add_object_by_template_id(human_template_id)
        # self.objs.append(obj)
        
        # obj_template_handle = './scripts/humantwo.object_config.json'
        # obj_template = obj_template_mgr.get_template_by_handle(obj_template_handle)
        # print(obj_template)
        # file_obj = rigid_obj_mgr.add_object_by_template_handle(obj_template_handle) 
        # print(file_obj)
        file_obj.motion_type = habitat_sim.physics.MotionType.KINEMATIC
        self.leader = file_obj
        orientation_x = 90  # @param {type:"slider", min:-180, max:180, step:1}
        orientation_y = (np.pi/2-0.97)*180/np.pi #+angle_diff[1]*180/np.pi# @param {type:"slider", min:-180, max:180, step:1}
        orientation_z = 180  # @param {type:"slider", min:-180, max:180, step:1}@param {type:"slider", min:-180, max:180, step:1}
        rotation_x = mn.Quaternion.rotation(mn.Deg(orientation_x), mn.Vector3(1.0, 0, 0))
        rotation_y = mn.Quaternion.rotation(mn.Deg(orientation_y), mn.Vector3(0.0, 1.0, 0))
        rotation_z = mn.Quaternion.rotation(mn.Deg(orientation_z), mn.Vector3(0.0, 0, 1.0))
        object_orientation2 = rotation_z * rotation_y * rotation_x
        leader_pos_3d = self.env._sim.pathfinder.get_random_navigable_point_near(self.env._sim.robot.base_pos, 2)
        leader_offset = leader_pos_3d - agent_state.position
        leader_offset[1] += 1.0
        set_object_state_from_agent(self.env._sim, self.leader, np.array(leader_offset), orientation = object_orientation2)
        self.leader_velocity_control = self.leader.velocity_control
        self.leader_velocity_control.controlling_lin_vel = True
        self.leader_velocity_control.controlling_ang_vel = True
        self.leader_velocity_control.ang_vel_is_local = False
        self.leader_velocity_control.lin_vel_is_local = False
        self.leader_velocity_control.linear_velocity = np.array([0.0,0.0,0.0])
        self.leader_velocity_control.angular_velocity = np.array([0.0,0.0,0.0])
        object_state = self.leader.rigid_state.translation
        current_initial_pos_2d = to_grid(self.env._sim.pathfinder, object_state, self.grid_dimensions)
        current_initial_pos_2d = [pos*0.025 for pos in current_initial_pos_2d]
        self.initial_state.append(current_initial_pos_2d+agents_initial_velocity+initial_pos)
        self.goal_dist[1] = np.linalg.norm((np.array(self.initial_state[1][0:2])-np.array(self.initial_state[1][4:6])))
        ##### Human 2 being initiated #####
        human_template_id = obj_template_mgr.load_configs('./scripts/humantwo')[0]
        self.follower_id = human_template_id
        file_obj = rigid_obj_mgr.add_object_by_template_id(human_template_id)
        # self.objs.append(obj)
        
        # obj_template_handle = './scripts/humantwo.object_config.json'
        # obj_template = obj_template_mgr.get_template_by_handle(obj_template_handle)
        # print(obj_template)
        # file_obj = rigid_obj_mgr.add_object_by_template_handle(obj_template_handle) 
        # print(file_obj)
        file_obj.motion_type = habitat_sim.physics.MotionType.KINEMATIC
        self.follower = file_obj
        orientation_x = 90  # @param {type:"slider", min:-180, max:180, step:1}
        orientation_y = (np.pi/2-0.97)*180/np.pi #+angle_diff[1]*180/np.pi# @param {type:"slider", min:-180, max:180, step:1}
        orientation_z = 180  # @param {type:"slider", min:-180, max:180, step:1}@param {type:"slider", min:-180, max:180, step:1}
        rotation_x = mn.Quaternion.rotation(mn.Deg(orientation_x), mn.Vector3(1.0, 0, 0))
        rotation_y = mn.Quaternion.rotation(mn.Deg(orientation_y), mn.Vector3(0.0, 1.0, 0))
        rotation_z = mn.Quaternion.rotation(mn.Deg(orientation_z), mn.Vector3(0.0, 0, 1.0))
        object_orientation2 = rotation_z * rotation_y * rotation_x
        follower_pos_3d = self.env._sim.pathfinder.get_random_navigable_point_near(self.env._sim.robot.base_pos, 2)
        follower_offset = follower_pos_3d - agent_state.position
        follower_offset[1] += 1.0
        set_object_state_from_agent(self.env._sim, self.follower, np.array(follower_offset), orientation = object_orientation2)
        self.follower_velocity_control = self.follower.velocity_control
        self.follower_velocity_control.controlling_lin_vel = True
        self.follower_velocity_control.controlling_ang_vel = True
        self.follower_velocity_control.ang_vel_is_local = False
        self.follower_velocity_control.lin_vel_is_local = False
        self.follower_velocity_control.linear_velocity = np.array([0.0,0.0,0.0])
        self.follower_velocity_control.angular_velocity = np.array([0.0,0.0,0.0])
        object_state = self.follower.rigid_state.translation
        current_initial_pos_2d = to_grid(self.env._sim.pathfinder, object_state, self.grid_dimensions)
        current_initial_pos_2d = [pos*0.025 for pos in current_initial_pos_2d]
        self.initial_state.append(current_initial_pos_2d+agents_initial_velocity+initial_pos)
        self.goal_dist[2] = np.linalg.norm((np.array(self.initial_state[2][0:2])-np.array(self.initial_state[2][4:6])))

        #### Add the rest of the people with random goal assigned ####
        self.human_template_ids = []
        self.objs = []
        self.vel_control_objs = []
        
        for k in range(self.N):
            human_template_id = obj_template_mgr.load_configs('./scripts/human')[0]
            self.human_template_ids.append(human_template_id)
            file_obj = rigid_obj_mgr.add_object_by_template_id(human_template_id)
            # self.objs.append(obj)
            
            # obj_template_handle = './scripts/humantwo.object_config.json'
            # obj_template = obj_template_mgr.get_template_by_handle(obj_template_handle)
            # print(obj_template)
            # file_obj = rigid_obj_mgr.add_object_by_template_handle(obj_template_handle) 
            # print(file_obj)
            file_obj.motion_type = habitat_sim.physics.MotionType.KINEMATIC
            self.objs.append(file_obj)
            
            #### Pick a random start location for this agent ####
            start_pos_3d = self.env._sim.pathfinder.get_random_navigable_point_near(self.env._sim.robot.base_pos,10)
            # start_pos = from_grid(self.env._sim.pathfinder, start_pos_3d, self.grid_dimensions)
            goal_pos_3d = self.env._sim.pathfinder.get_random_navigable_point_near(start_pos_3d, 10)
            self.final_goals_3d[k+1,:] = goal_pos_3d
            path = habitat_path.ShortestPath()
            path.requested_start = np.array(start_pos_3d)
            path.requested_end = goal_pos_3d
            if(not self.env._sim.pathfinder.find_path(path)):
                print("Watch this one Tribhi!!!!",i)
                continue
            humans_initial_pos_3d.append(path.points[0])
            humans_goal_pos_3d.append(path.points[1])
            human_initial_pos = list(to_grid(self.env._sim.pathfinder, humans_initial_pos_3d[-1], self.grid_dimensions))
            human_initial_pos = [pos*0.025 for pos in human_initial_pos]
            goal_pos = list(to_grid(self.env._sim.pathfinder, humans_goal_pos_3d[-1], self.grid_dimensions))
            goal_pos = [pos*0.025 for pos in goal_pos]
            self.initial_state.append(human_initial_pos+agents_initial_velocity+goal_pos)
            agent_state = self.env.sim.get_agent_state(0)
            offset = humans_initial_pos_3d[k]-agent_state.position
            offset[1] += 1.0
            orientation_x = 90   # @param {type:"slider", min:-180, max:180, step:1}
            orientation_y = (np.pi/2-0.97)*180/np.pi  # @param {type:"slider", min:-180, max:180, step:1}
            orientation_z = 180  # @param {type:"slider", min:-180, max:180, step:1}
            rotation_x = mn.Quaternion.rotation(mn.Deg(orientation_x), mn.Vector3(1.0, 0, 0))
            rotation_y = mn.Quaternion.rotation(mn.Deg(orientation_y), mn.Vector3(0.0, 1.0, 0))
            rotation_z = mn.Quaternion.rotation(mn.Deg(orientation_z), mn.Vector3(0.0, 0, 1.0))
            object_orientation = rotation_z * rotation_y * rotation_x
            set_object_state_from_agent(self.env._sim, file_obj, offset=offset, orientation = object_orientation)
            vel_control_obj = file_obj.velocity_control
            vel_control_obj.controlling_lin_vel = True
            vel_control_obj.controlling_ang_vel = True
            vel_control_obj.ang_vel_is_local = False
            vel_control_obj.lin_vel_is_local = False
            self.vel_control_objs.append(vel_control_obj)
            # linear_vel_in_map_frame = np.array([0.1,0.0,0.0])
            # angular_velocity_in_map_frame = np.array([0.0,0.0,0.0])
            
            self.vel_control_objs[k].linear_velocity = np.array([0.0,0.0,0.0])
            self.vel_control_objs[k].angular_velocity = np.array([0.0,0.0,0.0])
            self.goal_dist[k+3] = np.linalg.norm((np.array(self.initial_state[k+3][0:2])-np.array(self.initial_state[k+3][4:6])))
        
        if USE_RVO:
            self.sfm = ped_rvo(self, map_path = "./maps/resolution_"+scene+"_0.025.pgm", resolution = 0.025)
            print("Initialized rvo2 sim")
        else:
            self.sfm = social_force(self, map_path = "./maps/resolution_"+scene+"_0.025.pgm", resolution = 0.025, groups = self.groups)
            print("Initialized ESFM sim")
        print(self.initial_state)
        # self.initial_state.append(robot_pos_in_2d+humans_initial_velocity[0]+humans_goal_pos_2d[2])
        # self.groups.append([self.N])
        agent_state = self.env.sim.get_agent_state(0)
        map_to_base_link({'x': initial_pos[0], 'y': initial_pos[1], 'theta': self.env.sim.robot.base_rot}, self)
        self.initial_pos = initial_pos
        print("created habitat_plant succsefully")

    def __del__(self):
        if (len(self.all_points)>1):
            with open('./scripts/all_points.csv',  mode='w') as csvfile:
                csv_writer = csv.writer(csvfile, delimiter=',')
                for points in self.all_points:
                    print(points)
                    csv_writer.writerow(points)
            csvfile.close()

    def _render(self):
        self.observations.update(self.env._task._sim.get_observations_at()) 


    def update_agent_pos_vel(self): 
        if(self.agent_update_counter/self.update_multiple == self.human_update_counter):
            self.update_pos_vel()
            agent_pos = self.env.sim.robot.base_pos
            start_pos = [agent_pos[0], agent_pos[1], agent_pos[2]]
            initial_pos = list(to_grid(self.env._sim.pathfinder, start_pos, self.grid_dimensions))
            initial_pos = [pos*0.025 for pos in initial_pos]
            vel = [(self.initial_pos[0]-initial_pos[0])/self.human_time_step,(self.initial_pos[1]-initial_pos[1])/self.human_time_step]
            self.initial_state[0][2:4] = vel
        self.agent_update_counter +=1
        lin_vel = self.linear_velocity[2]
        ang_vel = self.angular_velocity[1]
        base_vel = [lin_vel, ang_vel]
        # self.observations.update(self.env.step({"action":"BASE_VELOCITY", "action_args":{"base_vel":base_vel}}))
        self.env.step({"action":"BASE_VELOCITY", "action_args":{"base_vel":base_vel}})
        ##### For teleop human/follower 
        # self.follower_velocity_control.linear_velocity = self.linear_velocity
        # self.follower_velocity_control.angular_velocity = self.angular_velocity
        self.lr.push_transform(self.env._sim.robot.base_transformation)
        origin = mn.Vector3(0.0, 0.0, 0.0)
        draw_axes(self.env._sim,origin, axis_len=0.4)
        self.lr.pop_transform()
        #### Forward simulate
        self.env.sim.step_physics(self.time_step)
        self.observations.update(self.env._task._sim.get_sensor_observations())

    def update_pos_vel(self):
        #### Update agent state 
        agent_pos = self.env.sim.robot.base_pos
        start_pos = [agent_pos[0], agent_pos[1], agent_pos[2]]
        initial_pos = list(to_grid(self.env._sim.pathfinder, start_pos, self.grid_dimensions))
        initial_pos = [pos*0.025 for pos in initial_pos]
        self.initial_state[0][0:2] = initial_pos
        self.goal_dist[0] = np.linalg.norm((np.array(self.initial_state[0][0:2])-np.array(self.initial_state[0][4:6])))
        ####  Update Leader state in ESFM 
        object_state = self.leader.rigid_state.translation
        current_initial_pos_2d = to_grid(self.env._sim.pathfinder, object_state, self.grid_dimensions)
        current_initial_pos_2d = [pos*0.025 for pos in current_initial_pos_2d]
        self.initial_state[1][0:2] = current_initial_pos_2d
        self.initial_state[1][4:6] = initial_pos
        self.goal_dist[1] = np.linalg.norm((np.array(self.initial_state[1][0:2])-np.array(self.initial_state[1][4:6])))
        #### Update Follower state in ESFM
        object_state = self.follower.rigid_state.translation
        current_initial_pos_2d = to_grid(self.env._sim.pathfinder, object_state, self.grid_dimensions)
        current_initial_pos_2d = [pos*0.025 for pos in current_initial_pos_2d]
        self.initial_state[2][4:6] = initial_pos
        self.initial_state[2][0:2] = current_initial_pos_2d
        self.goal_dist[2] = np.linalg.norm((np.array(self.initial_state[2][0:2])-np.array(self.initial_state[2][4:6])))
        #### Update other humans state in ESFM, sample new goal if reached 
        for k in range(self.N):
            human_state = self.objs[k].rigid_state.translation
            current_initial_pos_2d = to_grid(self.env._sim.pathfinder, human_state, self.grid_dimensions)
            current_initial_pos_2d = [pos*0.025 for pos in current_initial_pos_2d]
            self.initial_state[k+3][0:2] = current_initial_pos_2d
            self.goal_dist[+3] = np.linalg.norm((np.array(self.initial_state[k+3][0:2])-np.array(self.initial_state[k+3][4:6])))
        #### Calculate new velocity
        
        computed_velocity = self.sfm.get_velocity(np.array(self.initial_state), groups = self.groups, filename = "result_counter"+str(self.update_counter))
        
        #### Set new velocity for the follower
        human_state = self.follower.rigid_state
        computed_velocity[2,:] = [computed_velocity[2,0], computed_velocity[2,1]]
        next_vel_control = mn.Vector3(computed_velocity[2,0], computed_velocity[2,1], 0.0)
        diff_angle = quat_from_two_vectors(mn.Vector3(1,0,0), next_vel_control)
        diff_list = [diff_angle.x, diff_angle.y, diff_angle.z, diff_angle.w]
        angle= tf.transformations.euler_from_quaternion(diff_list)
        orientation_x = 90  # @param {type:"slider", min:-180, max:180, step:1}
        orientation_y = (np.pi/2-0.97)*180/np.pi+angle[2]*180/np.pi#+angle_diff[1]*180/np.pi# @param {type:"slider", min:-180, max:180, step:1}
        orientation_z = 180  # @param {type:"slider", min:-180, max:180, step:1}@param {type:"slider", min:-180, max:180, step:1}
        rotation_x = mn.Quaternion.rotation(mn.Deg(orientation_x), mn.Vector3(1.0, 0, 0))
        rotation_y = mn.Quaternion.rotation(mn.Deg(orientation_y), mn.Vector3(0.0, 1.0, 0))
        rotation_z = mn.Quaternion.rotation(mn.Deg(orientation_z), mn.Vector3(0.0, 0, 1.0))
        object_orientation2 = rotation_z * rotation_y * rotation_x
        agent_state = self.env._sim.get_agent_state(0)
        if(not np.isnan(angle).any()):
            set_object_state_from_agent(self.env._sim, self.follower, offset= human_state.translation - agent_state.position, orientation = object_orientation2)
            # if(self.goal_dist[1]>self.goal_dist[0]):
            self.follower_velocity_control.linear_velocity = [computed_velocity[2,0], 0.0,  computed_velocity[2,1]]
            print("setting linear velocity for follower")
            # else:
                # self.follower_velocity_control.linear_velocity = [0.0,0.0,0.0]
        else:
            self.follower_velocity_control.linear_velocity = [0.0,0.0,0.0]
            self.follower_velocity_control.angular_velocity = [0.0,0.0,0.0]
            computed_velocity[2,:] = [computed_velocity[2,0], computed_velocity[2,1]]

        #### Setting leader velocity ####
        human_state = self.leader.rigid_state
        next_vel_control = mn.Vector3(computed_velocity[1,0], computed_velocity[1,1], 0.0)
        diff_angle = quat_from_two_vectors(mn.Vector3(1,0,0), next_vel_control)
        diff_list = [diff_angle.x, diff_angle.y, diff_angle.z, diff_angle.w]
        angle= tf.transformations.euler_from_quaternion(diff_list)
        orientation_x = 90  # @param {type:"slider", min:-180, max:180, step:1}
        orientation_y = (np.pi/2-0.97)*180/np.pi+angle[2]*180/np.pi#+angle_diff[1]*180/np.pi# @param {type:"slider", min:-180, max:180, step:1}
        orientation_z = 180  # @param {type:"slider", min:-180, max:180, step:1}@param {type:"slider", min:-180, max:180, step:1}
        rotation_x = mn.Quaternion.rotation(mn.Deg(orientation_x), mn.Vector3(1.0, 0, 0))
        rotation_y = mn.Quaternion.rotation(mn.Deg(orientation_y), mn.Vector3(0.0, 1.0, 0))
        rotation_z = mn.Quaternion.rotation(mn.Deg(orientation_z), mn.Vector3(0.0, 0, 1.0))
        object_orientation2 = rotation_z * rotation_y * rotation_x
        agent_state = self.env._sim.get_agent_state(0)
        if(not np.isnan(angle).any()):
            set_object_state_from_agent(self.env._sim, self.leader, offset= human_state.translation - agent_state.position, orientation = object_orientation2)
            self.leader_velocity_control.linear_velocity = [computed_velocity[1,0], 0.0,  computed_velocity[1,1]]
            print("setting linear velocity for leader")
        else:
            self.leader_velocity_control.linear_velocity = [0.0,0.0,0.0]
            self.leader_velocity_control.angular_velocity = [0.0,0.0,0.0]
            computed_velocity[1,:] = [computed_velocity[1,0], computed_velocity[1,1]]
        # print(computed_velocity, self.follower_velocity_control.linear_velocity)
        
        self.initial_state[1][2:4] = [computed_velocity[1,0], computed_velocity[1,1]]
        self.initial_state[2][2:4] = [computed_velocity[2,0], computed_velocity[2,1]]

        #### Setting velocity for the other humans 
        for k in range(self.N):
            human_state = self.objs[k].rigid_state
            next_vel_control = mn.Vector3(computed_velocity[k+3,0], computed_velocity[k+3,1], 0.0)
            diff_angle = quat_from_two_vectors(mn.Vector3(1,0,0), next_vel_control)
            diff_list = [diff_angle.x, diff_angle.y, diff_angle.z, diff_angle.w]
            angle= tf.transformations.euler_from_quaternion(diff_list)
            orientation_x = 90  # @param {type:"slider", min:-180, max:180, step:1}
            orientation_y = (np.pi/2-0.97)*180/np.pi+angle[2]*180/np.pi#+angle_diff[1]*180/np.pi# @param {type:"slider", min:-180, max:180, step:1}
            orientation_z = 180  # @param {type:"slider", min:-180, max:180, step:1}@param {type:"slider", min:-180, max:180, step:1}
            rotation_x = mn.Quaternion.rotation(mn.Deg(orientation_x), mn.Vector3(1.0, 0, 0))
            rotation_y = mn.Quaternion.rotation(mn.Deg(orientation_y), mn.Vector3(0.0, 1.0, 0))
            rotation_z = mn.Quaternion.rotation(mn.Deg(orientation_z), mn.Vector3(0.0, 0, 1.0))
            object_orientation2 = rotation_z * rotation_y * rotation_x
            agent_state = self.env._sim.get_agent_state(0)
            if(not np.isnan(angle).any()):
                set_object_state_from_agent(self.env._sim, self.objs[k], offset= human_state.translation - agent_state.position, orientation = object_orientation2)
                self.vel_control_objs[k].linear_velocity = [computed_velocity[k+3,0], 0.0,  computed_velocity[k+3,1]]
                print("setting linear velocity for extra agent")
            else:
                self.leader_velocity_control.linear_velocity = [0.0,0.0,0.0]
                self.leader_velocity_control.angular_velocity = [0.0,0.0,0.0]
            # print(computed_velocity, self.follower_velocity_control.linear_velocity)
            self.initial_state[k+3][2:4] = [computed_velocity[k+3,0], computed_velocity[k+3,1]]
            
            #### Update to next topogoal if reached the first one 
            GOAL_THRESHOLD = 0.2
            if (self.goal_dist[k]<= GOAL_THRESHOLD):
                final_goal_grid = list(to_grid(self.env._sim.pathfinder, self.final_goals_3d[k+1,:], self.grid_dimensions))
                goal_pos = [pos*0.025 for pos in final_goal_grid]
                dist = np.linalg(goal_pos - self.initial_state[k+3][4:6])
                path = habitat_path.ShortestPath()
                path.requested_start = np.array(human_state.translation)
                #### If it isn't a intermediate goal, sample a new goal for the agent 
                if dist<=GOAL_THRESHOLD:
                    new_goal_pos_3d = self.env._sim.pathfinder.get_random_navigable_point_near(human_state.translation, 10)
                    path.requested_end = new_goal_pos_3d 
                    if(not self.env._sim.pathfinder.find_path(path)):
                        continue
                    self.final_goals_3d[k+1,:] = new_goal_pos_3d 
                    
                #### Update to next intermediate goal 
                else:
                    path.requested_end = self.final_goals_3d[k+1,:]
                    if(not self.env._sim.pathfinder.find_path(path)):
                        continue
                humans_goal_pos_3d = path.points[1]
                goal_pos = list(to_grid(self.env._sim.pathfinder, humans_goal_pos_3d, self.grid_dimensions))
                goal_pos = [pos*0.025 for pos in goal_pos]
                self.initial_state[k+3][4:6] = goal_pos
                self.goal_dist[k+3] = np.linalg((np.array(self.initial_state[k+3][0:2])-np.array(self.initial_state[k+3][4:6])))



        # map_to_base_link({'x': initial_pos[0], 'y': initial_pos[1], 'theta': mn.Rad(np.arctan2(vel[1], vel[0]))},self)

        self.update_counter+=1
        self.human_update_counter +=1
        
        
        


    def run(self):
        """Publish sensor readings through ROS on a different thread.
            This method defines what the thread does when the start() method
            of the threading class is called
        """
        while not rospy.is_shutdown():
            lock.acquire()
            rgb_with_res = np.concatenate(
                (
                    np.float32(self.observations["robot_third_rgb"][:,:,0:3].ravel()),
                    np.array(
                        [512,512]
                    ),
                )
            )
            # multiply by 10 to get distance in meters
            depth_with_res = np.concatenate(
                (
                    np.float32(self.observations["robot_head_depth"].ravel() ),
                    np.array(
                        [
                            128,
                            128
                        ]
                    ),
                )
            )       

            self._pub_rgb.publish(np.float32(rgb_with_res))
            self._pub_depth.publish(np.float32(depth_with_res))

            #### Publish pose and transform
            agent_pos = self.env.sim.robot.base_pos
            start_pos = [agent_pos[0], agent_pos[1], agent_pos[2]]
            initial_pos = list(to_grid(self.env._sim.pathfinder, start_pos, self.grid_dimensions))
            initial_pos = [pos*0.025 for pos in initial_pos]
            self.initial_state[0][0:2] = initial_pos
            # self.initial_state[0][2:4] = vel
            self.goal_dist[0] = np.linalg.norm((np.array(self.initial_state[0][0:2])-np.array(self.initial_state[0][4:6])))
            map_to_base_link({'x': initial_pos[0], 'y': initial_pos[1], 'theta': self.get_object_heading(self.env._sim.robot.base_transformation)},self)
            lock.release()
            self._r_sensor.sleep()
            

    def update_orientation(self):
        # if self.received_vel:
            # self.received_vel = False
            # self.vel_control_objs[0].linear_velocity = self.linear_velocity
            # self.vel_control_objs[0].angular_velocity = self.angular_velocity
            # self.vel_control.linear_velocity = self.linear_velocity
            # self.vel_control.angular_velocity = self.angular_velocity
        self.update_agent_pos_vel()
        if(self._global_plan_published):
            if(self.new_goal and self._current_episode<self._total_number_of_episodes):
                print("Executing goal number ", self._current_episode, self._total_number_of_episodes)
                self.new_goal= False
                poseMsg = PoseStamped()
                poseMsg.header.frame_id = "map"
                poseMsg.pose.orientation.x = self.current_goal[3]
                poseMsg.pose.orientation.y = self.current_goal[4]
                poseMsg.pose.orientation.z = self.current_goal[5]
                poseMsg.pose.orientation.w = self.current_goal[6]
                poseMsg.header.stamp = rospy.Time.now()
                poseMsg.pose.position.x = self.current_goal[0]
                poseMsg.pose.position.y = self.current_goal[1]
                poseMsg.pose.position.z = self.current_goal[2]
                self.pub_goal.publish(poseMsg)
            
                    

        rospy.sleep(self.time_step)
        # self.linear_velocity = np.array([0.0,0.0,0.0])
        # self.angular_velocity = np.array([0.0,0.0,0.0])
        # self.vel_control.linear_velocity = self.linear_velocity
        # self.vel_control.angular_velocity = self.angular_velocity
        # self._render()

    def set_dt(self, dt):
        self._dt = dt
      
    def get_object_heading(self,obj_transform):
        a = obj_transform
        b = a.transform_point([0.5,0.0,0.0])
        d = a.transform_point([0.0,0.0,0.0])
        c = np.array(to_grid(self.env._sim.pathfinder, [b[0],b[1],b[2]], self.grid_dimensions))
        e = np.array(to_grid(self.env._sim.pathfinder, [d[0],d[1],d[2]], self.grid_dimensions))
        vel = (c-e)*(0.5/np.linalg.norm(c-e)*np.ones([1,2]))[0]
        return mn.Rad(np.arctan2(vel[1], vel[0]))

    def plan_callback(self,msg):
        print("In plan_callback ", self._global_plan_published)
        # if(self._global_plan_published == False):
        if (True):
            self._global_plan_published = True
            length = len(msg.data)
            self._nodes = msg.data.reshape(int(length/7),7)   
            self._total_number_of_episodes = self._nodes.shape[0]
            self.current_goal = self._nodes[self._current_episode+1]
            print("Exiting plan_callback")
            self.tour_planning_time = rospy.get_time() - self.start_time
            print("Calculated a tour plan in ", self.tour_planning_time)
            self.new_goal = True
            self.goal_time = rospy.get_time()
        # else:
        #     if(self.goal_reached):
        #         self.current_goal = self._nodes[self._current_episode+1]
        #         self._current_episode = self._current_episode+1
        #         self.new_goal=True
        # if(self.new_goal):
        #     self.new_goal= False
        #     self.poseMsg = PoseStamped()
        #     # self.poseMsg.header.frame_id = "world"
        #     self.poseMsg.pose.orientation.x = 0
        #     self.poseMsg.pose.orientation.y = 0
        #     self.poseMsg.pose.orientation.z = 0
        #     self.poseMsg.pose.orientation.w = 1
        #     self.poseMsg.header.stamp = rospy.Time.now()
        #     self.poseMsg.pose.position.x = self.current_goal[0]
        #     self.poseMsg.pose.position.y = self.current_goal[1]
        #     self.poseMsg.pose.position.z = self.current_goal[2]
        #     self.pub_goal(poseMsg)

    def landmark_callback(self,point):
        self.all_points.append([point.pose.position.x,point.pose.position.y,point.pose.position.z,point.pose.orientation.x,point.pose.orientation.y,point.pose.orientation.z,point.pose.orientation.w])        
        print(self._current_episode+1, self.all_points[-1])
    #     map_points_3d = np.array([map_points[1], floor_y, map_points[0]])
    #     # self.current_goal = [map_points[1], floor_y, map_points[0]]
    #     # goal_position = np.array([map_points[1], floor_y, map_points[0]], dtype=np.float32)
    #     self._current_episode = self._current_episode+1
    #     # self.new_goal = True   
    def point_callback(self,point):
        # depot = self.rtab_pose
        # self.start_time = rospy.get_time()
        # self.tour_plan.plan(depot)
        computed_velocity = self.sfm.get_velocity(np.array(self.initial_state), groups = self.groups, filename = "requested save", save_anim = True)


    def rtabpose_callback(self,point):
        pose = point.pose
        self.rtab_pose = [pose.pose.position.x,pose.pose.position.y,pose.pose.position.z,pose.pose.orientation.x,pose.pose.orientation.y,pose.pose.orientation.z,pose.pose.orientation.w]

    def update_goal_status(self,msg):
        self.goal_reached=msg.data
        print("Rtabmap goal reached? ", self.goal_reached)

    def update_move_base_goal_status(self,msg):
        self.goal_reached = (msg.status.status ==3)
        print("Move base goal reached? ", self.goal_reached)
        self.goal_time = rospy.get_time()-self.goal_time
        print("Time to execute this tour is", self.goal_time)
        self.goal_time = rospy.get_time()
        if(self._current_episode==self._total_number_of_episodes-1):
                print("Tour plan executed in ", rospy.get_time()-self.start_time)
        else:
            self.current_goal = self._nodes[self._current_episode+1]
            self._current_episode = self._current_episode+1
            self.new_goal=True

def callback(vel, my_env):
    #### Robot Control ####
    my_env.linear_velocity = np.array([(1.0 * vel.linear.y), 0.0, (1.0 * vel.linear.x)])
    my_env.angular_velocity = np.array([0, vel.angular.z, 0])
    #### Follower control #####
    # my_env.linear_velocity = np.array([-vel.linear.x*np.sin(0.97), -vel.linear.x*np.cos(0.97),0.0])
    # my_env.angular_velocity = np.array([0, 0, vel.angular.z])
    # my_env.received_vel = True
    # my_env.update_orientation()

def main():

    my_env = sim_env(env_config_file="configs/tasks/custom_rearrange.yml")
    # start the thread that publishes sensor readings
    my_env.start()

    rospy.Subscriber("/cmd_vel", Twist, callback, (my_env), queue_size=1)

    # # Old code
    while not rospy.is_shutdown():
   
        my_env.update_orientation()
        # rospy.spin()
        my_env._r_control.sleep()

if __name__ == "__main__":
    main()