 #!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.



from habitat.utils.geometry_utils import quaternion_rotate_vector
from habitat.tasks.utils import cartesian_to_polar
from habitat_sim.utils import common as utils
from habitat_sim.utils.common import d3_40_colors_rgb
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
import cv2
import sys
sys.path.append("/root/miniconda3/envs/robostackenv/lib/python3.9/site-packages")
sys.path.append("/opt/conda/envs/robostackenv/lib/python3.9/site-packages")
import rospy
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
from std_msgs.msg import Bool
from geometry_msgs.msg import Twist, TransformStamped
from geometry_msgs.msg import PointStamped, PoseStamped, PoseWithCovarianceStamped, PoseArray, Pose
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
import struct
import geometry_msgs
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import tf2_ros
import yaml

lock = threading.Lock()
rospy.init_node("sim", anonymous=False)
import argparse
PARSER = argparse.ArgumentParser(description=None)
PARSER.add_argument('-s', '--scene', default="17DRP5sb8fy", type=str, help='scene')
ARGS = PARSER.parse_args()
scene = ARGS.scene
USE_RVO = False
IMAGE_DIR = "/home/catkin_ws/src/habitat_ros_interface/images/current_scene"

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
    follower = []
    new_goal = False
    control_frequency = 20
    time_step = 1.0 / (control_frequency)
    _r_control = rospy.Rate(control_frequency)
    human_control_frequency = 5
    human_time_step = 1/human_control_frequency
    linear_velocity = np.array([0.0,0.0,0.0])
    angular_velocity = np.array([0.0,0.0,0.0])
    received_vel = False
    goal_reached = False
    start_time = []
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
        config=habitat.get_config(self.env_config_file)
        self.env = habitat.Env(config)
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
        
        with open(IMAGE_DIR+"/image_config.yaml", "r") as stream:
            try:
                image_config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
                raise
        semantic_scene = self.env.sim.semantic_annotations()
        self.chosen_object = semantic_scene.objects[image_config["object_id"]]
        self.semantic_img_H = image_config["H"]
        self.semantic_img_W = image_config["W"]
        with open(image_config["projection_matrix"], 'rb') as f:
            self.semantic_img_proj_mat = np.load(f)
        with open(image_config["camera_matrix"], 'rb') as f:
            self.semantic_img_camera_mat = np.load(f)
        self.semantic_img = cv2.imread(IMAGE_DIR+"/semantic_img.png")   

        #### Initialize agent near the door     
        temp_position, rot = self.get_in_band_around_door(agent_state.rotation)
        self.env.sim.set_agent_state(temp_position, rot)
        self.env._sim.robot.base_pos = mn.Vector3(temp_position)
        agent_state = self.env.sim.get_agent_state(0)
        print(self.env.sim)
        config = self.env._sim.config
        print(self.env._sim.active_dataset)
        self._pub_rgb = rospy.Publisher("~rgb", numpy_msg(Floats), queue_size=1)
        self._pub_semantic = rospy.Publisher("~semantic", numpy_msg(Floats), queue_size=1)
        self._pub_depth = rospy.Publisher("~depth", numpy_msg(Floats), queue_size=1)
        self._pub_third_rgb = rospy.Publisher("~third_rgb", numpy_msg(Floats), queue_size=1)
        self._robot_pose = rospy.Publisher("~robot_pose", PoseStamped, queue_size = 1)
        self.cloud_pub = rospy.Publisher("semantic_cloud", PointCloud2, queue_size=2)
        self._pub_all_agents = rospy.Publisher("~agent_poses", PoseArray, queue_size = 1)
        self._pub_goal_marker = rospy.Publisher("~goal", Marker, queue_size = 1)
        
        self.br = tf.TransformBroadcaster()
        self.br_tf_2 = tf2_ros.TransformBroadcaster()
        # self._pub_pose = rospy.Publisher("~pose", PoseStamped, queue_size=1)
        # rospy.Subscriber("~plan_3d", numpy_msg(Floats),self.plan_callback, queue_size=1)
        # rospy.Subscriber("/rtabmap/goal_reached", Bool,self.update_goal_status, queue_size=1)
        rospy.Subscriber("/move_base/result", MoveBaseActionResult ,self.update_move_base_goal_status, queue_size=1)
        # rospy.Subscriber("/rtabmap/localization_pose", PoseWithCovarianceStamped,self.rtabpose_callback,queue_size=1)    
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
        self.N = 1
        
        # self.groups = [[0,1,2], [3], [4],[5],[6],[7]]
        self.groups = [[0], [1]]

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
        self.goal_dist = np.zeros(self.N+1)


        #### Initiating robot in the esfm state ####
        agent_pos = self.env.sim.robot.base_pos
        start_pos = [agent_pos[0], agent_pos[1], agent_pos[2]]
        
        ## Asume the agent goal is always the goal of the 0th agent
        path = habitat_path.ShortestPath()
        path.requested_start = np.array(start_pos)
        for i in range(50):
            agent_goal_pos_3d, goal_rot = self.get_in_band_around_door(agent_state.rotation)
            if (self.is_point_on_other_side(agent_goal_pos_3d, agent_state.position)):
                break
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
        
        #### Add the rest of the people with random goal assigned ####
        self.human_template_ids = []
        self.objs = []
        self.vel_control_objs = []
        
        for k in range(self.N):
            human_template_id = obj_template_mgr.load_configs('./scripts/human')[0]
            self.human_template_ids.append(human_template_id)
            file_obj = rigid_obj_mgr.add_object_by_template_id(human_template_id)
            file_obj.motion_type = habitat_sim.physics.MotionType.KINEMATIC
            self.objs.append(file_obj)
            
            #### Pick a random start location for this agent ####
            start_pos_3d, a = self.get_in_band_around_door()
            # start_pos = from_grid(self.env._sim.pathfinder, start_pos_3d, self.grid_dimensions)
            temp_goal_pos_3d, a = self.get_in_band_around_door()
            while(not self.is_point_on_other_side(start_pos_3d, temp_goal_pos_3d)):
                temp_goal_pos_3d, a = self.get_in_band_around_door()
            goal_pos_3d = temp_goal_pos_3d
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
            self.goal_dist[k+1] = np.linalg.norm((np.array(self.initial_state[k+1][0:2])-np.array(self.initial_state[k+1][4:6])))
        
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
        self.map_to_base_link({'x': initial_pos[0], 'y': initial_pos[1], 'theta': self.env.sim.robot.base_rot})
        self.initial_pos = initial_pos
        print("created habitat_plant succsefully")

    def get_in_band_around_door(self, agent_rotation = None):
        temp_position = self.env.sim.pathfinder.get_random_navigable_point_near(self.chosen_object.aabb.center,5)
        diff_vec = temp_position - self.chosen_object.aabb.center
        diff_vec[1] = 0
        temp_dist = np.linalg.norm(diff_vec)
        print("Diff vector ", diff_vec)
        while (temp_dist < 1.5 or temp_dist >2.5):
            temp_position = self.env.sim.pathfinder.get_random_navigable_point_near(self.chosen_object.aabb.center,5)
            diff_vec = temp_position - self.chosen_object.aabb.center
            diff_vec[1] = 0
            temp_dist = np.linalg.norm(diff_vec)
        
        if (agent_rotation):
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

    def is_point_on_other_side(self, p1, p2):
        transform = self.chosen_object.obb.world_to_local
        p1_local = np.matmul(transform, np.append(p1,1.0).T)
        p2_local = np.matmul(transform, np.append(p2,1.0).T)
        y1 = p1_local[2]
        y2 = p2_local[2]
        x1 = p1_local[1]
        x2 = p1_local[1]

        if (np.sign(x1) == np.sign(x2) and np.sign(y1) == np.sign(y2)):
            return False
        else:
            return True
        
    def _render(self):
        self.observations.update(self.env._task._sim.get_observations_at()) 

    def map_to_base_link(self, msg):
        theta = msg['theta']
        use_tf_2 = True
        if (not use_tf_2):
            self.br.sendTransform((-self.initial_state[0][0]+1, -self.initial_state[0][1]+1,0.0),
                            tf.transformations.quaternion_from_euler(0, 0, 0.0),
                            rospy.Time(0),
                            "my_map_frame",
                            "interim_link"
            )
            self.br.sendTransform((0.0,0.0,0.0),
                            tf.transformations.quaternion_from_euler(0, 0, -theta),
                            rospy.Time(0),
                            "interim_link",
                            "base_link"
            )
        else:
            t = geometry_msgs.msg.TransformStamped()
            t.header.stamp = rospy.Time.now()
            t.header.frame_id = "interim_link"
            t.child_frame_id = "my_map_frame"
            t.transform.translation.x = -self.initial_state[0][0]+1
            t.transform.translation.y = -self.initial_state[0][1]+1
            t.transform.translation.z = 0.0
            q = tf.transformations.quaternion_from_euler(0, 0, 0.0)
            t.transform.rotation.x = q[0]
            t.transform.rotation.y = q[1]
            t.transform.rotation.z = q[2]
            t.transform.rotation.w = q[3]
            self.br_tf_2.sendTransform(t)

            t = geometry_msgs.msg.TransformStamped()
            t.header.stamp = rospy.Time.now()
            t.header.frame_id = "base_link"
            t.child_frame_id = "interim_link"
            t.transform.translation.x = 0.0
            t.transform.translation.y = 0.0
            t.transform.translation.z = 0.0
            q = tf.transformations.quaternion_from_euler(0, 0, -theta)
            t.transform.rotation.x = q[0]
            t.transform.rotation.y = q[1]
            t.transform.rotation.z = q[2]
            t.transform.rotation.w = q[3]
            self.br_tf_2.sendTransform(t)

        poseMsg = PoseStamped()
        poseMsg.header.stamp = rospy.Time.now()
        poseMsg.header.frame_id = "base_link"
        quat = tf.transformations.quaternion_from_euler(0, 0, 0.0)
        poseMsg.pose.orientation.x = quat[0]
        poseMsg.pose.orientation.y = quat[1]
        poseMsg.pose.orientation.z = quat[2]
        poseMsg.pose.orientation.w = quat[3]
        poseMsg.pose.position.x = 0.0
        poseMsg.pose.position.y = 0.0
        poseMsg.pose.position.z = 0.0
        self._robot_pose.publish(poseMsg)

        ##### Publish other agents 
        poseArrayMsg = PoseArray()
        poseArrayMsg.header.frame_id = "my_map_frame"
        poseArrayMsg.header.stamp = rospy.Time.now()
        # follower_pos = my_env.follower.rigid_state.translation
        # theta = my_env.get_object_heading(my_env.follower.transformation)
        # quat = tf.transformations.quaternion_from_euler(0, 0, theta)
        # follower_pose_2d = to_grid(my_env.env._sim.pathfinder, follower_pos, my_env.grid_dimensions)
        # follower_pose_2d = follower_pose_2d*(0.025*np.ones([1,2]))[0]
        
        for i in range(len(self.initial_state)-1):
            poseMsg = Pose()
            # if (i==0):
            #     theta = my_env.get_object_heading(my_env.leader.transformation) - mn.Rad(np.pi/2-0.97 +np.pi)
            # elif (i==1):
            #     theta = my_env.get_object_heading(my_env.follower.transformation) - mn.Rad(np.pi/2-0.97 + np.pi)
            # else:
            theta = self.get_object_heading(self.objs[i].transformation) - mn.Rad(np.pi/2-0.97 +np.pi)
            quat = tf.transformations.quaternion_from_euler(0, 0, theta)
            poseMsg.orientation.x = quat[0]
            poseMsg.orientation.y = quat[1]
            poseMsg.orientation.z = quat[2]
            poseMsg.orientation.w = quat[3]
            poseMsg.position.x = self.initial_state[i+1][0]-1
            poseMsg.position.y = self.initial_state[i+1][1]-1
            poseMsg.position.z = 0.0
            poseArrayMsg.poses.append(poseMsg)
        self._pub_all_agents.publish(poseArrayMsg)

        goal_marker = Marker()
        goal_marker.header.frame_id = "my_map_frame"
        goal_marker.type = 2
        goal_marker.pose.position.x = self.initial_state[0][4]-1
        goal_marker.pose.position.y = self.initial_state[0][5]-1
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
        self._pub_goal_marker.publish(goal_marker)

    def update_agent_pos_vel(self): 
        if(self.agent_update_counter/self.update_multiple == self.human_update_counter):
            self.update_pos_vel()
            agent_pos = self.env.sim.robot.base_pos
            start_pos = [agent_pos[0], agent_pos[1], agent_pos[2]]
            initial_pos = list(to_grid(self.env._sim.pathfinder, start_pos, self.grid_dimensions))
            initial_pos = [pos*0.025 for pos in initial_pos]
            vel = [(self.initial_pos[0]-initial_pos[0])/self.human_time_step,(self.initial_pos[1]-initial_pos[1])/self.human_time_step]
            self.initial_state[0][2:4] = vel
            self.initial_pos = initial_pos
        self.agent_update_counter +=1
        lin_vel = self.linear_velocity[2]
        ang_vel = self.angular_velocity[1]
        base_vel = [lin_vel, ang_vel]
        # self.observations.update(self.env.step({"action":"BASE_VELOCITY", "action_args":{"base_vel":base_vel}}))
        self.env.step({"action":"BASE_VELOCITY", "action_args":{"base_vel":base_vel}})
        ##### For teleop human/follower 
        # self.follower_velocity_control.linear_velocity = self.linear_velocity
        # self.follower_velocity_control.angular_velocity = self.angular_velocity
        origin = mn.Vector3(0.0, 0.0, 0.0)
        #### Forward simulate
        self.env.sim.step_physics(self.time_step)
        self.observations.update(self.env._task._sim.get_sensor_observations())

    def update_pos_vel(self):
        #### Update agent state 
        agent_pos = self.env.sim.robot.base_pos
        start_pos = [agent_pos[0], agent_pos[1], agent_pos[2]]
        initial_pos = list(to_grid(self.env._sim.pathfinder, start_pos, self.grid_dimensions))
        initial_pos = [pos*0.025 for pos in initial_pos]
        a = self.env.sim.robot.base_transformation
        b = a.transform_point([-0.2,0.0,0.0])
        point_behind = np.array(to_grid(self.env._sim.pathfinder, [b[0],b[1],b[2]], self.grid_dimensions))
        point_behind = [pos*0.025 for pos in point_behind]
        self.initial_state[0][0:2] = initial_pos
        self.goal_dist[0] = np.linalg.norm((np.array(self.initial_state[0][0:2])-np.array(self.initial_state[0][4:6])))
        
        for k in range(self.N):
            human_state = self.objs[k].rigid_state.translation
            current_initial_pos_2d = to_grid(self.env._sim.pathfinder, human_state, self.grid_dimensions)
            current_initial_pos_2d = [pos*0.025 for pos in current_initial_pos_2d]
            self.initial_state[k+1][0:2] = current_initial_pos_2d
            self.goal_dist[k+1] = np.linalg.norm((np.array(self.initial_state[k+1][0:2])-np.array(self.initial_state[k+1][4:6])))
        #### Calculate new velocity
        
        computed_velocity = self.sfm.get_velocity(np.array(self.initial_state), groups = self.groups, filename = "result_counter"+str(self.update_counter))
        
        #### Setting velocity for the other humans 
        for k in range(self.N):
            human_state = self.objs[k].rigid_state
            next_vel_control = mn.Vector3(computed_velocity[k+1,0], computed_velocity[k+1,1], 0.0)
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
                self.vel_control_objs[k].linear_velocity = [computed_velocity[k+1,0], 0.0,  computed_velocity[k+1,1]]
                print("setting linear velocity for extra agent")
            else:
                self.vel_control_objs[k].linear_velocity = [0.0,0.0,0.0]
                self.vel_control_objs[k].angular_velocity = [0.0,0.0,0.0]
            # print(computed_velocity, self.follower_velocity_control.linear_velocity)
            self.initial_state[k+1][2:4] = [computed_velocity[k+1,0], computed_velocity[k+1,1]]
            
            #### Update to next topogoal if reached the first one 
            GOAL_THRESHOLD = 0.5
            if (self.goal_dist[k+1]<= GOAL_THRESHOLD):
                final_goal_grid = list(to_grid(self.env._sim.pathfinder, self.final_goals_3d[k+1,:], self.grid_dimensions))
                goal_pos = [pos*0.025 for pos in final_goal_grid]
                dist = np.linalg.norm(np.array(goal_pos) - np.array(self.initial_state[k+1][4:6]))
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
                self.initial_state[k+1][4:6] = goal_pos
                self.goal_dist[k+1] = np.linalg.norm((np.array(self.initial_state[k+1][0:2])-np.array(self.initial_state[k+1][4:6])))
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
            third_rgb_with_res = np.concatenate(
                (
                    np.float32(self.observations["robot_third_rgb"][:,:,0:3].ravel()),
                    np.array(
                        [128,128]
                    ),
                )
            )
            rgb_with_res = np.concatenate(
                (
                    np.float32(self.observations["robot_head_rgb"][:,:,0:3].ravel()),
                    np.array(
                        [128,128]
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
            
            semantic_img =self.semantic_img
            semantic_with_res = np.concatenate(
                (
                    np.float32(semantic_img[:, :, 0:3].ravel()),
                    np.array(
                        [semantic_img.shape[0], semantic_img.shape[1]]
                    ),
                )
            )
            # points = []
            # for i in range(0,semantic_img.shape[0],5):
            #     for j in range(0,semantic_img.shape[1], 5):
            #         world_coordinates = sem_img_to_world(self.semantic_img_proj_mat, self.semantic_img_camera_mat, self.semantic_img_W, self.semantic_img_H, i, j)
            #         [x,y] = list(maps.to_grid(world_coordinates[2], world_coordinates[0], self.grid_dimensions, pathfinder = self.env._sim.pathfinder))
            #         # print([i,j])
            #         # x = x - 1
            #         if (i ==j == 360):
            #             center_gt = list(to_grid(self.env._sim.pathfinder, self.chosen_object.aabb.center, self.grid_dimensions))
            #             print(center_gt[0] - y, center_gt[1]-x)
            #         [y,x] = [x*0.025, y*0.025]
            #         [r,g,b] = semantic_img[i, j, 0:3]
            #         a = 255
            #         z = 0.1
            #         rgb = struct.unpack('I', struct.pack('BBBB', b, g, r, a))[0]
            #         pt = [x - 1.0, y - 1.0, z, rgb]
            #         points.append(pt)
            # fields = [PointField('x', 0, PointField.FLOAT32, 1),
            # PointField('y', 4, PointField.FLOAT32, 1),
            # PointField('z', 8, PointField.FLOAT32, 1),
            # # PointField('rgb', 12, PointField.UINT32, 1),
            # PointField('rgba', 12, PointField.UINT32, 1),
            # ]
            # header = Header()
            # header.frame_id = "my_map_frame"
            # pc2 = point_cloud2.create_cloud(header, fields, points)
            # pc2.header.stamp = rospy.Time.now()
            # self.cloud_pub.publish(pc2)
            # cv2.imwrite("semantic_image.png", semantic_img)
            self._pub_rgb.publish(np.float32(rgb_with_res))
            self._pub_semantic.publish(np.float32(semantic_with_res))
            self._pub_depth.publish(np.float32(depth_with_res))
            self._pub_third_rgb.publish(np.float32(third_rgb_with_res))
            #### Publish pose and transform
            agent_pos = self.env.sim.robot.base_pos
            start_pos = [agent_pos[0], agent_pos[1], agent_pos[2]]
            initial_pos = list(to_grid(self.env._sim.pathfinder, start_pos, self.grid_dimensions))
            initial_pos = [pos*0.025 for pos in initial_pos]
            self.initial_state[0][0:2] = initial_pos
            # self.initial_state[0][2:4] = vel
            self.goal_dist[0] = np.linalg.norm((np.array(self.initial_state[0][0:2])-np.array(self.initial_state[0][4:6])))
            self.map_to_base_link({'x': initial_pos[0], 'y': initial_pos[1], 'theta': self.get_object_heading(self.env._sim.robot.base_transformation)})
            lock.release()
            self._r_sensor.sleep()
            

    def update_orientation(self):
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

    def point_callback(self,point):
        # depot = self.rtab_pose
        # self.start_time = rospy.get_time()
        # self.tour_plan.plan(depot)
        computed_velocity = self.sfm.get_velocity(np.array(self.initial_state), groups = self.groups, filename = "requested save", save_anim = True)

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