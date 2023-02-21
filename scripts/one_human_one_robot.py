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
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PointStamped, PoseStamped, PoseWithCovarianceStamped
import threading
import tf
# from tour_planner_dropped import tour_planner
import csv
from move_base_msgs.msg import MoveBaseActionResult
from matplotlib import pyplot as plt
from IPython import embed
from nav_msgs.srv import GetPlan
from get_trajectory import *
lock = threading.Lock()
rospy.init_node("robot_1", anonymous=False)

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
    _r = rospy.Rate(_sensor_rate)
    _current_episode = 0
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
    control_frequency = 20
    time_step = 1.0 / (control_frequency)
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
    update_multiple = 2
    def __init__(self, env_config_file):
        threading.Thread.__init__(self)
        self.env_config_file = env_config_file
        self.env = habitat.Env(config=habitat.get_config(self.env_config_file))
        self.env._sim.robot.params.arm_init_params = [1.32, 1.40, -0.2, 1.72, 0.0, 1.66, 0.0]
        
        print("Initializeed environment")
        # always assume height equals width
        # self.env._sim.agents[0].move_filter_fn = self.env._sim.step_filter
        agent_state = self.env.sim.get_agent_state(0)
        self.observations = self.env.reset()
        arm_joint_positions  = [1.32, 1.40, -0.2, 1.72, 0.0, 1.66, 0.0]
        self.env._sim.robot.arm_joint_pos = arm_joint_positions
        agent_state.position = [-2.293175119872487,0.0,-1.2777875958067]
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
        floor_y = 0.0
        top_down_map = maps.get_topdown_map(
            self.env._sim.pathfinder, height=floor_y, meters_per_pixel=0.025
        )
        self.grid_dimensions = (top_down_map.shape[0], top_down_map.shape[1])
        print("Grid size is ", self.grid_dimensions)
        maps.draw_path(top_down_map, ([0,0], [300,300]))
        plt.figure(figsize=(12, 8))
        ax = plt.subplot(1, 1, 1)
        ax.axis("on")
        plt.imshow(top_down_map)
        plt.savefig("./top_down_map.png")
        self._pub_rgb = rospy.Publisher("~rgb", numpy_msg(Floats), queue_size=1)
        self._pub_depth = rospy.Publisher("~depth", numpy_msg(Floats), queue_size=1)
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
        self.sfm = social_force()
        self.sfm.load_obs_from_map()
        
        global rigid_obj_mgr
        rigid_obj_mgr = self.env._sim.get_rigid_object_manager()
        global obj_template_mgr
        obj_template_mgr = self.env._sim.get_object_template_manager()
        rigid_obj_mgr.remove_all_objects()

        config=habitat.get_config(self.env_config_file)
        
        robot_pos_in_2d = to_grid(self.env._sim.pathfinder, self.env._sim.robot.base_pos, self.grid_dimensions)
        print(robot_pos_in_2d)
        ### Add human objects and groups here! 

        self.N = 2
        map_points = []
        with open('./scripts/humans_initial_points.csv', newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            for row in spamreader:
                map_points.append([float(i) for i in row])
        
        self.humans_initial_pos_2d = map_points[:-1]
        goal_idx = np.arange(self.N)
        random.shuffle(goal_idx)
        humans_goal_pos_2d = np.array(self.humans_initial_pos_2d)[goal_idx]
        humans_goal_pos_2d = list(humans_goal_pos_2d)
        self.groups = [[0],[1]]

        #Test with 1 
        # self.N = 1
        # map_points = [[380.0,290.0]]
        
        # humans_initial_pos_2d = map_points
        # humans_goal_pos_2d = [[370,360]]
        # self.groups = [[0]]
        # Use poissons distribution to get the number of groups.
        # Select random navigable points for the different groups and members. 
        
        
        
        self.human_template_ids = []
        self.objs = []
        self.vel_control_objs = []
        self.initial_state = []
        humans_initial_pos_3d = []
        humans_goal_pos_3d = []
        humans_initial_velocity = []
        self.final_goals_3d = np.zeros([self.N, 3])
        self.goal_dist = np.zeros(self.N)
        for i in range(self.N):
            human_template_id = obj_template_mgr.load_configs('./scripts/humantwo')[0]
            self.human_template_ids.append(human_template_id)
            obj = rigid_obj_mgr.add_object_by_template_id(human_template_id)
            # self.objs.append(obj)
            
            obj_template_handle = './scripts/humantwo.object_config.json'
            obj_template = obj_template_mgr.get_template_by_handle(obj_template_handle)
            print(obj_template)
            file_obj = rigid_obj_mgr.add_object_by_template_handle(obj_template_handle) 
            print(file_obj)
            file_obj.motion_type = habitat_sim.physics.MotionType.KINEMATIC
            self.objs.append(file_obj)
            # initial_point = ([358,350])
            # map_point_3d = from_grid(self.env._sim.pathfinder, initial_point, self.grid_dimensions)
            # print("the object position is ", map_point_3d)
            # Set initial state for the SFM 
            
            start_pos = from_grid(self.env._sim.pathfinder, self.humans_initial_pos_2d[i], self.grid_dimensions)
            self.final_goals_3d[i,:] = np.array(from_grid(self.env._sim.pathfinder, humans_goal_pos_2d[i], self.grid_dimensions))
            path = habitat_path.ShortestPath()
            path.requested_start = np.array(start_pos)
            path.requested_end = self.final_goals_3d[i,:]
            if(not self.env._sim.pathfinder.find_path(path)):
                print("Watch this one Tribhi!!!!",i)
                embed()
                continue
            humans_initial_pos_3d.append(path.points[0])
            humans_goal_pos_3d.append(path.points[1])
            humans_initial_velocity.append([0.5,0.0])
            initial_pos = list(to_grid(self.env._sim.pathfinder, humans_initial_pos_3d[-1], self.grid_dimensions))
            initial_pos = [pos*0.025 for pos in initial_pos]
            goal_pos = list(to_grid(self.env._sim.pathfinder, humans_goal_pos_3d[-1], self.grid_dimensions))
            goal_pos = [pos*0.025 for pos in goal_pos]
            self.initial_state.append(initial_pos+humans_initial_velocity[i]+goal_pos)
            agent_state = self.env.sim.get_agent_state(0)
            print(" The agent position", agent_state.position)
            offset = humans_initial_pos_3d[i]-agent_state.position
            offset[1] = 1.0
            print("Here is the offset", offset)
            obj_template.scale *= 0.1
            orientation_x = 90   # @param {type:"slider", min:-180, max:180, step:1}
            orientation_y = (np.pi/2-0.97)*180/np.pi  # @param {type:"slider", min:-180, max:180, step:1}
            orientation_z = 180  # @param {type:"slider", min:-180, max:180, step:1}
            rotation_x = mn.Quaternion.rotation(mn.Deg(orientation_x), mn.Vector3(1.0, 0, 0))
            rotation_y = mn.Quaternion.rotation(mn.Deg(orientation_y), mn.Vector3(0.0, 1.0, 0))
            rotation_z = mn.Quaternion.rotation(mn.Deg(orientation_z), mn.Vector3(0.0, 0, 1.0))
            self.object_orientation = rotation_z * rotation_y * rotation_x
            
            set_object_state_from_agent(self.env._sim, file_obj, offset=offset, orientation = self.object_orientation)
            vel_control_obj = file_obj.velocity_control
            vel_control_obj.controlling_lin_vel = True
            vel_control_obj.controlling_ang_vel = True
            vel_control_obj.ang_vel_is_local = False
            vel_control_obj.lin_vel_is_local = False
            self.vel_control_objs.append(vel_control_obj)
            # linear_vel_in_map_frame = np.array([0.1,0.0,0.0])
            # angular_velocity_in_map_frame = np.array([0.0,0.0,0.0])
            
            self.vel_control_objs[i].linear_velocity = np.array([0.0,0.0,0.0])
            self.vel_control_objs[i].angular_velocity = np.array([0.0,0.0,0.0])
        # orientation_x = 0   # @param {type:"slider", min:-180, max:180, step:1}
        # orientation_y = 0  # @param {type:"slider", min:-180, max:180, step:1}
        # orientation_z = 0  # @param {type:"slider", min:-180, max:180, step:1}
        # rotation_x = mn.Quaternion.rotation(mn.Deg(orientation_x), mn.Vector3(1.0, 0, 0))
        # rotation_y = mn.Quaternion.rotation(mn.Deg(orientation_y), mn.Vector3(0.0, 1.0, 0))
        # rotation_z = mn.Quaternion.rotation(mn.Deg(orientation_z), mn.Vector3(0.0, 0, 1.0))
        # self.object_orientation = rotation_z * rotation_y * rotation_x
        
        print(self.initial_state)
        # self.initial_state.append(robot_pos_in_2d+humans_initial_velocity[0]+humans_goal_pos_2d[2])
        # self.groups.append([self.N])
        agent_state = self.env.sim.get_agent_state(0)
        
        computed_velocity = self.sfm.get_velocity(np.array(self.initial_state), groups = self.groups, filename = "result_counter__"+str(self.update_counter), save_anim = True)
        print("changing velocity", self.human_update_counter)
        for i in range(self.N):
            human_state = self.objs[i].rigid_state
            next_vel_control = mn.Vector3(computed_velocity[i,0], computed_velocity[i,1], 0.0)
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
            if(not np.isnan(angle).any()):
                set_object_state_from_agent(self.env._sim, self.objs[i], offset= human_state.translation - agent_state.position, orientation = object_orientation2)
                self.vel_control_objs[i].linear_velocity = [computed_velocity[i,0], 0.0,  computed_velocity[i,1]]
            else:
                self.vel_control_objs[i].linear_velocity = [0.0,0.0,0.0]
                self.vel_control_objs[i].angular_velocity = [0.0,0.0,0.0]

            self.initial_state[i][2:4] = computed_velocity[i]
            self.goal_dist[i] = np.linalg.norm((np.array(self.initial_state[i][0:2])-np.array(self.initial_state[i][4:6])))
        print(self.goal_dist)

        # self.vel_control_objs.angular_velocity = np.array([0.0,0.0,0.0])
        
        # set_object_state_from_agent(self.env._sim, self.file_obj3, offset=offset3, orientation = object_orientation3)
        # ao_mgr = self.env._sim.get_articulated_object_manager()
        # motion_type = habitat_sim.physics.MotionType.KINEMATIC
        # self.ao = ao_mgr.add_articulated_object_from_urdf("./scripts/model.urdf", fixed_base=True)
        # self.ao.motion_type = motion_type
        # set_object_state_from_agent(self.env._sim, self.ao, offset=offset3, orientation = object_orientation2)
        
        # print("Robot root linear and angular velocity is", self.env._sim.robot.root_linear_velocity, self.env._sim.robot.root_angular_velocity)
        # self.sphere_template_id = obj_template_mgr.load_configs('./scripts/sphere')[0]
        # print(self.sphere_template_id)
        # self.obj_3 = rigid_obj_mgr.add_object_by_template_id(self.sphere_template_id)
        # self.obj_template_handle3 = './scripts/sphere.object_config.json'
        # #self.obj_template_handle2 = './banana.object_config.json'
        # self.obj_template3 = obj_template_mgr.get_template_by_handle(self.obj_template_handle3)
        # self.obj_template3.scale *= 3  
        # self.file_obj3 = rigid_obj_mgr.add_object_by_template_handle(self.obj_template_handle3) 
        # objs3 = [self.file_obj3]
        # offset3= np.array([3,1,-1.5])
        # rotation_x = mn.Quaternion.rotation(mn.Deg(0), mn.Vector3(1.0, 0, 0))
        # rotation_y = mn.Quaternion.rotation(mn.Deg(0), mn.Vector3(0, 1.0, 0))
        # rotation_z = mn.Quaternion.rotation(mn.Deg(0), mn.Vector3(0, 0, 1.0))
        # object_orientation2 = rotation_z * rotation_y * rotation_x
        
        # set_object_state_from_agent(self.env._sim, self.file_obj3, offset=offset3, orientation = object_orientation2)
        print("created habitat_plant succsefully")
        embed()

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
    
    
    def _update_position(self):
        state = self.env.sim.get_agent_state(0)
        heading_vector = quaternion_rotate_vector(
            state.rotation.inverse(), np.array([0, 0, -1])
        )
        phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        top_down_map_angle = phi - np.pi / 2 
        agent_pos = state.position
        agent_quat = quat_to_coeff(state.rotation)
        euler = list(tf.transformations.euler_from_quaternion(agent_quat))
        proj_quat = tf.transformations.quaternion_from_euler(0.0,0.0,top_down_map_angle-np.pi)
        # proj_quat = tf.transformations.quaternion_from_euler(euler[0]+np.pi,euler[2],euler[1])
        agent_pos_in_map_frame = convert_points_to_topdown(self.env.sim.pathfinder, [agent_pos])
        self.poseMsg = PoseStamped()
        # self.poseMsg.header.frame_id = "world"
        self.poseMsg.pose.orientation.x = proj_quat[0]
        self.poseMsg.pose.orientation.y = proj_quat[2]
        self.poseMsg.pose.orientation.z = proj_quat[1]
        self.poseMsg.pose.orientation.w = proj_quat[3]
        self.poseMsg.header.stamp = rospy.Time.now()
        self.poseMsg.pose.position.x = agent_pos_in_map_frame[0][0]
        self.poseMsg.pose.position.y = agent_pos_in_map_frame[0][1]
        self.poseMsg.pose.position.z = 0.0
        # self._pub_pose.publish(self.poseMsg)
        
        # self._render()        
    def update_pos_vel(self):
        # agent_state = self.env.sim.get_agent_state(0)
        # previous_rigid_state = habitat_sim.RigidState(
        #     self.env._sim.robot.sim_obj.rotation, self.env._sim.robot.base_pos
        # )

        # # manually integrate the rigid state
        # target_rigid_state = self.vel_control.integrate_transform(
        #     self.time_step, previous_rigid_state
        # )

        # # snap rigid state to navmesh and set state to object/agent
        # # calls pathfinder.try_step or self.pathfinder.try_step_no_sliding
        # end_pos = self.env._sim.step_filter(
        #     previous_rigid_state.translation, target_rigid_state.translation
        # )

        # # set the computed state
        # agent_state.position = end_pos
        # # robot_angle = tf.transformations.euler_from_quaternion(quat_to_coeff(utils.quat_from_magnum(target_rigid_state.rotation)))[1]
        # # agent_angle_target = robot_angle
        # # agent_state.rotation = utils.quat_from_magnum(mn.Quaternion.rotation(
        # #     mn.Rad(agent_angle_target), mn.Vector3(0, 1, 0)
        # # ))
        # # agent_angle = tf.transformations.euler_from_quaternion(quat_to_coeff(agent_state.rotation))[1]
        # self.env._sim.robot.base_pos = end_pos
        # # agent_angle = tf.transformations.euler_from_quaternion(quat_to_coeff(agent_state.rotation))[1]
        # self.env._sim.robot.sim_obj.rotation = target_rigid_state.rotation
        lin_vel = self.linear_velocity[2]
        ang_vel = self.angular_velocity[1]
        base_vel = [lin_vel, ang_vel]
        self.observations.update(self.env.step({"action":"BASE_VELOCITY", "action_args":{"base_vel":base_vel}}))
        # run any dynamics simulation
        
        for i in range(self.N):
            object_state = self.objs[i].rigid_state.translation
            current_initial_pos_2d = to_grid(self.env._sim.pathfinder, object_state, self.grid_dimensions)
            current_initial_pos_2d = [pos*0.025 for pos in current_initial_pos_2d]
            self.initial_state[i][0:2] = current_initial_pos_2d
        self.update_counter+=1
        agent_state = self.env.sim.get_agent_state(0)
        if(np.mod(self.human_update_counter, self.update_multiple) ==0):
            computed_velocity = self.sfm.get_velocity(np.array(self.initial_state), groups = self.groups, filename = "result_counter"+str(self.update_counter))
            for i in range(self.N):
                human_state = self.objs[i].rigid_state
                next_vel_control = mn.Vector3(computed_velocity[i,0], computed_velocity[i,1], 0.0)
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
                if(not np.isnan(angle).any()):
                    set_object_state_from_agent(self.env._sim, self.objs[i], offset= human_state.translation - agent_state.position, orientation = object_orientation2)
                    self.vel_control_objs[i].linear_velocity = [computed_velocity[i,0], 0.0,  computed_velocity[i,1]]
                else:
                    self.vel_control_objs[i].linear_velocity = [0.0,0.0,0.0]
                    self.vel_control_objs[i].angular_velocity = [0.0,0.0,0.0]
                    term_pos = to_grid(self.env._sim.pathfinder, self.final_goals_3d[i,:], self.grid_dimensions)
                    pos = [pos*0.025 for pos in term_pos]
                    goals_2d_final = np.array(pos)
                    if (np.linalg.norm(self.initial_state[i][4:6] - goals_2d_final) > 1):
                        print(" In new path point same goal for!", i, self.initial_state[i][4:6], goals_2d_final) 
                        path = habitat_path.ShortestPath()
                        path.requested_start = np.array(human_state.translation)
                        path.requested_end = self.final_goals_3d[i,:]
                        if(not self.env._sim.pathfinder.find_path(path)):
                            print("Watch this one Tribhi!!!!",i)
                            continue
                        humans_goal_pos_3d = path.points[1]
                        for k in range(len(path.points)-1):
                            goal_pos = list(to_grid(self.env._sim.pathfinder, path.points[k+1], self.grid_dimensions))
                            if(np.linalg.norm(np.array(goal_pos) - self.initial_state[i][4:6]) >1):
                                humans_goal_pos_3d = path.points[k+1]
                                break
                        goal_pos = list(to_grid(self.env._sim.pathfinder, humans_goal_pos_3d, self.grid_dimensions))
                        goal_pos = [pos*0.025 for pos in goal_pos]
                        print("Old goal, new goal", self.initial_state[i][4:6], goal_pos)
                        self.initial_state[i][4:6] = goal_pos
                        
                    else:
                        print("Setting a completely different goal for", i)
                        j = random.randint(0,self.N-1)
                        self.final_goals_3d[i,:] = from_grid(self.env._sim.pathfinder, self.humans_initial_pos_2d[j], self.grid_dimensions)
                        path = habitat_path.ShortestPath()
                        path.requested_start = np.array(human_state.translation)
                        path.requested_end = self.final_goals_3d[i,:]
                        if(not self.env._sim.pathfinder.find_path(path)):
                            print("Watch this one Tribhi!!!!",i)
                            continue
                        humans_goal_pos_3d = path.points[1]
                        print(path.points)
                        goal_pos = list(to_grid(self.env._sim.pathfinder, humans_goal_pos_3d, self.grid_dimensions))
                        goal_pos = [pos*0.025 for pos in goal_pos]
                        self.initial_state[i][4:6] = goal_pos
                self.initial_state[i][2:4] = computed_velocity[i]
                self.goal_dist[i] = np.linalg.norm((np.array(self.initial_state[i][0:2])-np.array(self.initial_state[i][4:6])))
            print(self.initial_state)
        self.human_update_counter +=1
        self.env.sim.step_physics(self.time_step)
        self.observations.update(self.env._task._sim.get_sensor_observations())
        


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
            self._update_position()
            agent_state = self.env.sim.get_agent_state(0)
            
            # print(agent_state)
            # self.vel_control_obj_2.linear_velocity = np.array([0.0,0.0,0.0])
            # self.vel_control_obj_2.angular_velocity = np.array([0.0,0.0,0.0])
            # orientation_x = 0  # @param {type:"slider", min:-180, max:180, step:1}
            # orientation_y = 90  # @param {type:"slider", min:-180, max:180, step:1}
            # orientation_z = 90  # @param {type:"slider", min:-180, max:180, step:1}
            # rotation_x = mn.Quaternion.rotation(mn.Deg(orientation_x), mn.Vector3(1.0, 0, 0))
            # rotation_y = mn.Quaternion.rotation(mn.Deg(orientation_y), mn.Vector3(0, 1.0, 0))
            # rotation_z = mn.Quaternion.rotation(mn.Deg(orientation_z), mn.Vector3(0, 0, 1.0))
            # object_orientation2 = rotation_z * rotation_y * rotation_x
            # offset2= np.array([1,1,-0.5])
            # set_object_state_from_agent(self.env._sim, self.file_obj2, offset=offset2, orientation = object_orientation2)
            lock.release()
            self._r.sleep()
            

    def update_orientation(self):
        if self.received_vel:
            self.received_vel = False
            # self.vel_control_objs[0].linear_velocity = self.linear_velocity
            # self.vel_control_objs[0].angular_velocity = self.angular_velocity
            # self.vel_control.linear_velocity = self.linear_velocity
            # self.vel_control.angular_velocity = self.angular_velocity
        self.update_pos_vel()
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
        depot = self.rtab_pose
        self.start_time = rospy.get_time()
        self.tour_plan.plan(depot)

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
    my_env.linear_velocity = np.array([(1.0 * vel.linear.y), 0.0, (1.0 * vel.linear.x)])
    my_env.angular_velocity = np.array([0, vel.angular.z, 0])
    # my_env.linear_velocity = np.array([-vel.linear.x*np.sin(0.97), -vel.linear.x*np.cos(0.97),0.0])
    # my_env.angular_velocity = np.array([0, 0, vel.angular.z])
    my_env.received_vel = True
    # my_env.update_orientation()

def main():

    my_env = sim_env(env_config_file="configs/tasks/nav_to_obj_copy.yml")
    # start the thread that publishes sensor readings
    my_env.start()

    rospy.Subscriber("/cmd_vel", Twist, callback, (my_env), queue_size=1)
    # define a list capturing how long it took
    # to update agent orientation for past 3 instances
    # TODO modify dt_list to depend on r1
    dt_list = [0.009, 0.009, 0.009]

    # # Old code
    while not rospy.is_shutdown():
   
        my_env.update_orientation()
        # rospy.spin()
        my_env._r.sleep()

    # while not rospy.is_shutdown():
    #     start_time = time.time()
    #     # cv2.imshow("bc_sensor", my_env.observations['bc_sensor'])
    #     # cv2.waitKey(100)
    #     # time.sleep(0.1)
    #     my_env.update_orientation()

    #     dt_list.insert(0, time.time() - start_time)
    #     dt_list.pop()
    #     my_env.set_dt(sum(dt_list) / len(dt_list))

if __name__ == "__main__":
    main()