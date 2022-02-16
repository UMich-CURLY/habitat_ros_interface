import math
import os
import random
import sys

import git
import magnum as mn
import numpy as np

# %matplotlib inline
from matplotlib import pyplot as plt

import habitat_sim
from habitat_sim.utils import common as utils
from habitat_sim.utils import viz_utils as vut
from habitat.utils.visualizations import maps
import habitat_sim.bindings as hsim

from habitat.utils.geometry_utils import quaternion_rotate_vector
from habitat.tasks.utils import cartesian_to_polar
import sys
sys.path.append("/opt/conda/envs/robostackenv/lib/python3.9/site-packages")
import rospy
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PointStamped, PoseStamped
import threading
import tf

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

def make_configuration():
    # simulator configuration
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = "/home/catkin_ws/src/habitat_ros_interface/data/scene_datasets/mp3d/17DRP5sb8fy/17DRP5sb8fy.glb"
    assert os.path.exists(backend_cfg.scene_id)
    backend_cfg.enable_physics = True

    # sensor configurations
    # Note: all sensors must have the same resolution
    # setup 2 rgb sensors for 1st and 3rd person views
    camera_resolution = [256,256]
    sensor_specs = []

    rgba_camera_1stperson_spec = habitat_sim.CameraSensorSpec()
    rgba_camera_1stperson_spec.uuid = "rgba_camera_1stperson"
    rgba_camera_1stperson_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgba_camera_1stperson_spec.resolution = camera_resolution
    rgba_camera_1stperson_spec.postition = [0.0, 0.6, 0.0]
    rgba_camera_1stperson_spec.orientation = [0.0, 0.0, 0.0]
    rgba_camera_1stperson_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(rgba_camera_1stperson_spec)

    depth_camera_1stperson_spec = habitat_sim.CameraSensorSpec()
    depth_camera_1stperson_spec.uuid = "depth_camera_1stperson"
    depth_camera_1stperson_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_camera_1stperson_spec.resolution = camera_resolution
    depth_camera_1stperson_spec.postition = [0.0, 0.6, 0.0]
    depth_camera_1stperson_spec.orientation = [0.0, 0.0, 0.0]
    depth_camera_1stperson_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(depth_camera_1stperson_spec)

    rgba_camera_3rdperson_spec = habitat_sim.CameraSensorSpec()
    rgba_camera_3rdperson_spec.uuid = "rgba_camera_3rdperson"
    rgba_camera_3rdperson_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgba_camera_3rdperson_spec.resolution = camera_resolution
    rgba_camera_3rdperson_spec.postition = [0.0, 1.0, 0.3]
    rgba_camera_3rdperson_spec.orientation = [-45, 0.0, 0.0]
    rgba_camera_3rdperson_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(rgba_camera_3rdperson_spec)

    # agent configuration
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs

    return habitat_sim.Configuration(backend_cfg, [agent_cfg])

def remove_all_objects(sim):
    for id_ in sim.get_existing_object_ids():
        sim.remove_object(id_)

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
    def __init__(self, env_config_file):
        threading.Thread.__init__(self)
        self.env_config_file = env_config_file
        # self.env = habitat.Env(config=habitat.get_config(self.env_config_file))
        cfg = make_configuration()
        self.sim = habitat_sim.Simulator(config=cfg)
        self.agent = self.sim.initialize_agent(0)

        # Set agent state
        agent_state = habitat_sim.AgentState()
        agent_state.position = np.array([-2.293175119872487,0.0,-1.2777875958067])  # in world space
        self.agent.set_state(agent_state)

        # Get agent state
        agent_state = self.agent.get_state()
        self.observations = self.sim.get_sensor_observations()
        # print("Initializeed environment")
        # always assume height equals width
        
        # self.env._sim.agents[0].move_filter_fn = self.env._sim.step_filter
        # agent_state = self.sim.get_agent_state(0)
        # self.observations = self.sim.get_observations()
        # agent_state.position = [-2.293175119872487,0.0,-1.2777875958067]
        # self.env.sim.set_agent_state(agent_state.position, agent_state.rotation)
        # self.env._sim.agents[0].state.velocity = np.float32([0, 0, 0])
        # self.env._sim.agents[0].state.angular_velocity = np.float32([0, 0, 0])
        # # self._sensor_resolution = {
        # #     "RGB": self.env._sim.config["RGB_SENSOR"]["HEIGHT"],
        # #     "DEPTH": self.env._sim.config["DEPTH_SENSOR"]["HEIGHT"],
        # #     "BC_SENSOR": self.env._sim.config["BC_SENSOR"]["HEIGHT"],
        # # }
        # config = self.env._sim.config
        # print(self.env._sim.active_dataset)
        self._sensor_resolution = {
            "RGB": 256,  
            "DEPTH": 256,
        }
        # print(self.env._sim.pathfinder.get_bounds())
        floor_y = 0.0
        top_down_map = maps.get_topdown_map(
            self.sim.pathfinder, height=floor_y, meters_per_pixel=0.5
        )
        self.grid_dimensions = (top_down_map.shape[0], top_down_map.shape[1])
        self._pub_rgb = rospy.Publisher("~rgb", numpy_msg(Floats), queue_size=1)
        self._pub_depth = rospy.Publisher("~depth", numpy_msg(Floats), queue_size=1)
        self._pub_pose = rospy.Publisher("~pose", PoseStamped, queue_size=1)
        # rospy.Subscriber("~plan_3d", numpy_msg(Floats),self.plan_callback, queue_size=1)
        rospy.Subscriber("/clicked_point", PointStamped,self.point_callback,queue_size=1)    
        # # additional RGB sensor I configured
        # goal_radius = self.env.episodes[0].goals[0].radius
        # if goal_radius is None:
        #     goal_radius = config.SIMULATOR.FORWARD_STEP_SIZE
        # self.follower = ShortestPathFollower(
        #     self.env.sim, goal_radius, False
        # )
        print("created habitat_plant succsefully")
    def run(self):
        """Publish sensor readings through ROS on a different thread.
            This method defines what the thread does when the start() method
            of the threading class is called
        """
        while not rospy.is_shutdown():
            lock.acquire()
            rgb_with_res = np.concatenate(
                (
                    np.float32(self.observations["rgba_camera_1stperson"][:,:,0:3].ravel()),
                    np.array(
                        [self._sensor_resolution["RGB"], self._sensor_resolution["RGB"]]
                    ),
                )
            )
            # multiply by 10 to get distance in meters
            depth_with_res = np.concatenate(
                (
                    np.float32(self.observations["depth_camera_1stperson"].ravel() * 10),
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
            # print("Prints obs" , self.observations["rgba_camera_3rdperson"][:,:,0:3].size)
            self._update_position()
            # agent_state = self.env.sim.get_agent_state(0)
            # print(agent_state.rotation)
            lock.release()
            self._r.sleep()

    def _update_position(self):
        state = self.agent.get_state()
        if(self.new_goal):
            agent_state = habitat_sim.AgentState()
            agent_state.position = self.goal_position
            agent_state.rotation = state.rotation
            agent_state.velocity = np.array([1.0,1.0,0.0], dtype=np.float32)
            self.agent.set_state(agent_state)
            print(self.agent.get_state())
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
        agent_pos_in_map_frame = convert_points_to_topdown(self.sim.pathfinder, [agent_pos])
        self.poseMsg = PoseStamped()
        self.poseMsg.header.frame_id = "world"
        self.poseMsg.pose.orientation.x = proj_quat[0]
        self.poseMsg.pose.orientation.y = proj_quat[2]
        self.poseMsg.pose.orientation.z = proj_quat[1]
        self.poseMsg.pose.orientation.w = proj_quat[3]
        self.poseMsg.header.stamp = rospy.Time.now()
        self.poseMsg.pose.position.x = agent_pos_in_map_frame[0][0]
        self.poseMsg.pose.position.y = agent_pos_in_map_frame[0][1]
        self.poseMsg.pose.position.z = 0.0
        self._pub_pose.publish(self.poseMsg)

    def point_callback(self,point):   
        floor_y = 0.0
        map_points = maps.from_grid(
                            int(float(point.point.y)),
                            int(float(point.point.x)),
                            self.grid_dimensions,
                            pathfinder=self.sim.pathfinder,
                        )
        # my_env.current_goal = [map_points[1], floor_y, map_points[0]]
        agent_state = habitat_sim.AgentState()
        self.goal_position = np.array([map_points[1], floor_y, map_points[0]], dtype=np.float32)
        self.new_goal = True
        # agent_state.position = goal_position
        # my_env._current_episode = my_env._current_episode+1
        # my_env.new_goal = True
        # to be sure that the rotation is the same for the same episode_id
        # since the task is currently using pointnav Dataset.
        # seed = 100
        # rng = np.random.RandomState(seed)
        # angle = rng.uniform(0, 2 * np.pi)
        # source_rotation = [0, np.sin(angle / 2), 0, np.cos(angle / 2)]
        # goal_observation = self.sim.get_observations_at(
        #     position=goal_position.tolist(), rotation=source_rotation
        # )

        



def main():

    my_env = sim_env(env_config_file="configs/tasks/pointnav_rgbd.yaml")
    # start the thread that publishes sensor readings
    my_env.start()
    while not rospy.is_shutdown():
        rospy.spin()
    

if __name__ == "__main__":
    main()