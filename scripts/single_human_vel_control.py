 #!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.



from habitat.utils.geometry_utils import quaternion_rotate_vector
from habitat.tasks.utils import cartesian_to_polar
from habitat_sim.utils import common as utils
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
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
sys.path.append("/opt/conda/envs/robostackenv/lib/python3.9/site-packages")
import rospy
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
from std_msgs.msg import Bool
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PointStamped, PoseStamped, PoseWithCovarianceStamped
import threading
import tf
from tour_planner_dropped import tour_planner
import csv
from move_base_msgs.msg import MoveBaseActionResult
from nav_msgs.srv import GetPlan
from habitat_sim.robots import FetchRobot
from IPython import embed
lock = threading.Lock()
rospy.init_node("robot_1", anonymous=False)

def convert_points_to_topdown(pathfinder, points, meters_per_pixel = 0.5):
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
    
    def __init__(self, env_config_file):
        threading.Thread.__init__(self)
        self.env_config_file = env_config_file
        self.env = habitat.Env(config=habitat.get_config(self.env_config_file))
        print("Initializeed environment")
        # always assume height equals width
        
        self.env._sim.agents[0].move_filter_fn = self.env._sim.step_filter
        agent_state = self.env.sim.get_agent_state(0)
        self.observations = self.env.reset()
        agent_state.position = [-2.293175119872487,0.0,-1.2777875958067]
        self.env.sim.set_agent_state(agent_state.position, agent_state.rotation)
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
            self.env._sim.pathfinder, height=floor_y, meters_per_pixel=0.5
        )
        self.grid_dimensions = (top_down_map.shape[0], top_down_map.shape[1])
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
        goal_radius = self.env.episodes[0].goals[0].radius
        if goal_radius is None:
            goal_radius = config.SIMULATOR.FORWARD_STEP_SIZE
        self.follower = ShortestPathFollower(
            self.env.sim, goal_radius, False
        )
        self.env._sim.enable_physics = True
        self.vel_control = habitat_sim.physics.VelocityControl()
        self.vel_control.controlling_lin_vel = True
        self.vel_control.lin_vel_is_local = True
        self.vel_control.controlling_ang_vel = True
        self.vel_control.ang_vel_is_local = True
        self.tour_plan = tour_planner()
        print("before initialized object")
        global rigid_obj_mgr
        rigid_obj_mgr = self.env._sim.get_rigid_object_manager()
        global obj_template_mgr
        obj_template_mgr = self.env._sim.get_object_template_manager()
        rigid_obj_mgr.remove_all_objects()
        self.human_template_id = obj_template_mgr.load_configs('./scripts/humantwo')[0]
        print(self.human_template_id)
        self.obj_1 = rigid_obj_mgr.add_object_by_template_id(self.human_template_id)
        self.obj_template_handle = './scripts/humantwo.object_config.json'
        self.obj_template = obj_template_mgr.get_template_by_handle(self.obj_template_handle)
        # self.file_obj = rigid_obj_mgr.add_object_by_template_handle(self.obj_template_handle) 
        # objs = [self.file_obj]
        offset= np.array([-2,1,0.0])
        
        # self.obj_template.scale *= 3   
        orientation_x = 90  # @param {type:"slider", min:-180, max:180, step:1}
        orientation_y = 0  # @param {type:"slider", min:-180, max:180, step:1}
        orientation_z = 0 # @param {type:"slider", min:-180, max:180, step:1}
        # compose the rotations
        rotation_x = mn.Quaternion.rotation(mn.Deg(orientation_x), mn.Vector3(1.0, 0, 0))
        rotation_y = mn.Quaternion.rotation(mn.Deg(orientation_y), mn.Vector3(0, 1.0, 0))
        rotation_z = mn.Quaternion.rotation(mn.Deg(orientation_z), mn.Vector3(0, 0, 1.0))
        object_orientation = rotation_z * rotation_y * rotation_x
        print(object_orientation)
        # set_object_state_from_agent(self.env._sim, self.file_obj, offset=offset, orientation = object_orientation)

        self.banana_template_id = obj_template_mgr.load_configs('./scripts/human')[0]
        print(self.banana_template_id)
        self.obj_2 = rigid_obj_mgr.add_object_by_template_id(self.banana_template_id)
        self.obj_template_handle2 = './scripts/humantwo.object_config.json'
        self.obj_template2 = obj_template_mgr.get_template_by_handle(self.obj_template_handle2)
        # self.obj_template2.scale *= 3  
        self.file_obj2 = rigid_obj_mgr.add_object_by_template_handle(self.obj_template_handle2) 
        objs2 = [self.file_obj2]
        offset2= np.array([1.5,1.0,-1.5])
        self.obj_template.scale *= 3   
        orientation_x = 90   # @param {type:"slider", min:-180, max:180, step:1}
        orientation_y = (np.pi/2-0.97)*180/np.pi  # @param {type:"slider", min:-180, max:180, step:1}
        orientation_z = 180  # @param {type:"slider", min:-180, max:180, step:1}@param {type:"slider", min:-180, max:180, step:1}
        rotation_x = mn.Quaternion.rotation(mn.Deg(orientation_x), mn.Vector3(1.0, 0, 0))
        rotation_y = mn.Quaternion.rotation(mn.Deg(orientation_y), mn.Vector3(0.0, 1.0, 0))
        rotation_z = mn.Quaternion.rotation(mn.Deg(orientation_z), mn.Vector3(0.0, 0, 1.0))
        object_orientation2 = rotation_z * rotation_y * rotation_x
        self.initial_orientation = object_orientation2
        set_object_state_from_agent(self.env._sim, self.file_obj2, offset=offset2, orientation = object_orientation2)
        # self.env._sim.set_object_motion_type(habitat_sim.physics.MotionType.KINEMATIC, self.banana_template_id)
        self.file_obj2.motion_type = habitat_sim.physics.MotionType.KINEMATIC
        print("here")
        print(self.file_obj2.rigid_state.translation, self.file_obj2.rigid_state.rotation)
        self.file_obj2.override_collision_group(habitat_sim.physics.CollisionGroups.Robot)
        self.file_obj2.creation_attributes.orient_front = mn.Vector3(1,0,0)
        print(self.file_obj2.creation_attributes.orient_front)
        self.vel_control_obj_2 = self.file_obj2.velocity_control
        self.vel_control_obj_2.controlling_lin_vel = True
        self.vel_control_obj_2.controlling_ang_vel = True
        self.vel_control_obj_2.ang_vel_is_local = True
        self.vel_control_obj_2.lin_vel_is_local = True
        # self.vel_control_obj_2.linear_velocity = np.array([0.0,0.0,0.0])
        # self.vel_control_obj_2.angular_velocity = np.array([0.0,0.0,0.0])
        self.obj_3 = rigid_obj_mgr.add_object_by_template_id(self.human_template_id)
        self.obj_template_handle = './scripts/humantwo.object_config.json'
        self.obj_template3 = obj_template_mgr.get_template_by_handle(self.obj_template_handle)
        self.obj_template3.scale *= 3  
        # self.file_obj3 = rigid_obj_mgr.add_object_by_template_handle(self.obj_template_handle) 
        # objs3 = [self.file_obj3]
        self.obj_template.scale *= 3   
        orientation_x = 0  # @param {type:"slider", min:-180, max:180, step:1}
        orientation_y = 90  # @param {type:"slider", min:-180, max:180, step:1}
        orientation_z = 90  # @param {type:"slider", min:-180, max:180, step:1}
        rotation_x = mn.Quaternion.rotation(mn.Deg(orientation_x), mn.Vector3(1.0, 0.0, 0))
        rotation_y = mn.Quaternion.rotation(mn.Deg(orientation_y), mn.Vector3(0.0, 1.0, 0))
        rotation_z = mn.Quaternion.rotation(mn.Deg(orientation_z), mn.Vector3(0.0, 0.0, 1.0))
        object_orientation3 = rotation_z * rotation_y * rotation_x
        
        # set_object_state_from_agent(self.env._sim, self.file_obj3, offset=offset3, orientation = object_orientation3)
        # ao_mgr = self.env._sim.get_articulated_object_manager()
        # motion_type = habitat_sim.physics.MotionType.KINEMATIC
        # self.ao = ao_mgr.add_articulated_object_from_urdf("./scripts/model.urdf", fixed_base=True)
        # self.ao.motion_type = motion_type
        # set_object_state_from_agent(self.env._sim, self.ao, offset=offset3, orientation = object_orientation2)
        # config=habitat.get_config(self.env_config_file)
        # print(config.SIMULATOR.ROBOT_URDF)
        # self.env._sim.robot = FetchRobot(config.SIMULATOR.ROBOT_URDF, self.env._sim)
        # self.env._sim.robot.reconfigure()
        # self.env._sim.robot.base_pos = mn.Vector3([agent_state.position[0]+1.5, agent_state.position[1], agent_state.position[2]-1.0])
        # magnum_quat =  mn.Quaternion(mn.Vector3([agent_state.rotation.x, agent_state.rotation.y, agent_state.rotation.z]),agent_state.rotation.w)
        # self.env._sim.robot.sim_obj.rotation = utils.quat_to_magnum(agent_state.rotation)
        # print("here too")
        # print(self.env._sim.robot.sim_obj.translation,self.env._sim.robot.sim_obj.rotation)

        self.sphere_template_id = obj_template_mgr.load_configs('./scripts/sphere')[0]
        print(self.sphere_template_id)
        self.obj_3 = rigid_obj_mgr.add_object_by_template_id(self.sphere_template_id)
        self.obj_template_handle3 = './scripts/sphere.object_config.json'
        # #self.obj_template_handle2 = './banana.object_config.json'
        self.obj_template3 = obj_template_mgr.get_template_by_handle(self.obj_template_handle3)
        self.obj_template3.scale *= 3  
        self.file_obj3 = rigid_obj_mgr.add_object_by_template_handle(self.obj_template_handle3) 
        objs3 = [self.file_obj3]
        human_state = self.file_obj2.rigid_state
        inverted_quat = object_orientation2.inverted()
        self.inverted_vector = inverted_quat.transform_vector(mn.Vector3(1.0,0.0,0.0))
        extra_offset = human_state.rotation.transform_vector(self.inverted_vector)
        offset3= offset2 + np.array([extra_offset[0], extra_offset[1], extra_offset[2]])
        print(offset3)
        rotation_x = mn.Quaternion.rotation(mn.Deg(0), mn.Vector3(1.0, 0, 0))
        rotation_y = mn.Quaternion.rotation(mn.Deg(0), mn.Vector3(0, 1.0, 0))
        rotation_z = mn.Quaternion.rotation(mn.Deg(0), mn.Vector3(0, 0, 1.0))
        object_orientation2 = rotation_z * rotation_y * rotation_x
        set_object_state_from_agent(self.env._sim, self.file_obj3, offset=offset3, orientation = object_orientation2)
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
        agent_state = self.env.sim.get_agent_state(0)
        previous_rigid_state = habitat_sim.RigidState(
            utils.quat_to_magnum(agent_state.rotation), agent_state.position
        )

        # manually integrate the rigid state
        target_rigid_state = self.vel_control.integrate_transform(
            self.time_step, previous_rigid_state
        )

        # snap rigid state to navmesh and set state to object/agent
        # calls pathfinder.try_step or self.pathfinder.try_step_no_sliding
        end_pos = self.env._sim.step_filter(
            previous_rigid_state.translation, target_rigid_state.translation
        )

        # set the computed state
        agent_state.position = end_pos
        agent_state.rotation = utils.quat_from_magnum(
            target_rigid_state.rotation
        )
        self.env.sim.set_agent_state(agent_state.position, agent_state.rotation)
        # run any dynamics simulation
        
        human_state = self.file_obj2.rigid_state
        previous_human_rigid_state = human_state
        target_human_rigid_state = self.vel_control_obj_2.integrate_transform(
            self.time_step, previous_human_rigid_state
        )
        end_pos_human = self.env._sim.step_filter(
            previous_human_rigid_state.translation, target_human_rigid_state.translation
        )
        offset = end_pos_human - end_pos
        offset[1]+=1.0
        set_object_state_from_agent(self.env._sim, self.file_obj2, offset=offset, orientation = target_human_rigid_state.rotation)
        extra_offset = human_state.rotation.transform_vector(self.inverted_vector)
        offset3= offset + np.array([extra_offset[0], extra_offset[1], extra_offset[2]])
        rotation_x = mn.Quaternion.rotation(mn.Deg(0), mn.Vector3(1.0, 0, 0))
        rotation_y = mn.Quaternion.rotation(mn.Deg(0), mn.Vector3(0, 1.0, 0))
        rotation_z = mn.Quaternion.rotation(mn.Deg(0), mn.Vector3(0, 0, 1.0))
        object_orientation2 = rotation_z * rotation_y * rotation_x
        set_object_state_from_agent(self.env._sim, self.file_obj3, offset=offset3, orientation = object_orientation2)
        self.env.sim.step_physics(self.time_step)
        # render observation
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
                    np.float32(self.observations["rgb"][:,:,0:3].ravel()),
                    np.array(
                        [self._sensor_resolution["RGB"], self._sensor_resolution["RGB"]]
                    ),
                )
            )
            # multiply by 10 to get distance in meters
            depth_with_res = np.concatenate(
                (
                    np.float32(self.observations["depth"].ravel() ),
                    np.array(
                        [
                            self._sensor_resolution["DEPTH"],
                            self._sensor_resolution["DEPTH"],
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
            human_state = self.file_obj2.rigid_state
            extra_offset = human_state.rotation.transform_vector(self.inverted_vector)
            offset3= human_state.translation + np.array([extra_offset[0], extra_offset[1], extra_offset[2]])
            vel = self.linear_velocity
            next_vel_control = mn.Vector3(float(vel[0]), 0.0, float(vel[2]))
            speed = np.linalg.norm(vel)                
            inverted_quat = self.initial_orientation.inverted()
            body_vector = inverted_quat.transform_vector(next_vel_control)
            body_velocity = human_state.rotation.transform_vector(body_vector)
            diff_angle = quat_from_two_vectors(extra_offset, body_velocity)
            diff_list = [diff_angle.x, diff_angle.y, diff_angle.z, diff_angle.w]
            angle_diff = tf.transformations.euler_from_quaternion(diff_list)
            print(extra_offset, body_velocity, angle_diff)
            prev_heading = mn.Vector3(offset3[0], offset3[1], offset3[2])
            quat_rot = quat_from_two_vectors(prev_heading, next_vel_control)
            quat_list = [quat_rot.x, quat_rot.y, quat_rot.z, quat_rot.w]
            # angle = tf.transformations.euler_from_quaternion(quat_list)
            angle = angle_diff
            # if(np.abs(angle_diff[1]-np.pi)<0.1 or angle_diff[1]<0.01 or np.abs(angle_diff[1]+np.pi)<0.1):
            #     self.received_vel = False
            if(not np.isnan(angle).any()):
                if(np.abs(angle[2])>0.01):
                    self.vel_control_obj_2.angular_velocity = [0.0,0.0,angle[2]]
                    self.vel_control_obj_2.linear_velocity = [0.0,0.0,0.0]
                else:
                    print(angle)
                    self.vel_control_obj_2.linear_velocity = [-speed*np.cos(angle[2]+0.97),-speed*np.sin(angle[2]+0.97), 0.0]
                    self.vel_control_obj_2.angular_velocity = [0.0,0.0,0.0]
                    self.received_vel = False
            else:
                self.vel_control_obj_2.linear_velocity = [0.0,0.0,0.0]
                self.vel_control_obj_2.angular_velocity = [0.0,0.0,0.0]
            print("velocities", self.vel_control_obj_2.angular_velocity, self.vel_control_obj_2.linear_velocity)
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
    # human_state = my_env.file_obj2.rigid_state
    # prev_vel_control = mn.Vector3(1.0, 0.0, 0.0)
    # next_vel_control = mn.Vector3((1.0 * vel.linear.x), 0.0, (1.0 * vel.linear.y))
    # quat_rot = utils.quat_from_two_vectors(prev_vel_control, next_vel_control)
    # difference_quat = quat_rot - human_state.rotation
    my_env.linear_velocity = np.array([(1.0 * vel.linear.x), 0.0, (1.0 * vel.linear.y)])
    my_env.angular_velocity = np.array([0, vel.angular.z, 0])
    # my_env.linear_velocity = np.array([-vel.linear.x*np.sin(0.97), -vel.linear.x*np.cos(0.97),0.0])
    # my_env.angular_velocity = np.array([0, 0, vel.angular.z])
    my_env.received_vel = True
    # my_env.update_orientation()

def main():

    my_env = sim_env(env_config_file="configs/tasks/pointnav_rgbd.yaml")
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