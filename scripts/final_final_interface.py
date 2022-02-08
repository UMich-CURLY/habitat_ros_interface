import rospy
from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg
from geometry_msgs.msg import PointStamped, PoseStamped
from nav_msgs.msg import Path
from visualization_msgs.msg import MarkerArray, Marker
import numpy as np
import habitat
import habitat_sim.bindings as hsim
import magnum as mn
import csv
import random
import os
import os.path as osp
import gzip
import json
import threading
import tf
# %matplotlib inline
from matplotlib import pyplot as plt
from habitat.datasets.pointnav.pointnav_generator import generate_pointnav_episode
from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.nav.nav import NavigationTask
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.config.default import get_config as get_baselines_config
from habitat.tasks.nav.nav import NavigationEpisode, NavigationGoal
from habitat.utils.visualizations import maps
from habitat_sim.utils import common as utils
from habitat_sim.utils import viz_utils as vut
from habitat.utils.geometry_utils import quaternion_rotate_vector
from habitat.tasks.utils import cartesian_to_polar
# function to display the topdown map
from PIL import Image

def convert_points_to_topdown(pathfinder, points, meters_per_pixel = 0.5):
    points_topdown = []
    bounds = pathfinder.get_bounds()
    for point in points:
        # convert 3D x,z to topdown x,y
        px = (point[0] - bounds[0][0]) / meters_per_pixel
        py = (point[2] - bounds[0][2]) / meters_per_pixel
        points_topdown.append(np.array([px, py]))
    return points_topdown

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


lock = threading.Lock()
env_config_file = "configs/tasks/pointnav_rgbd.yaml"


def setup_eval(trainer):
    print("I am atleast called")
    # env = trainer.envs._connection_read_fns[0]()
    read_fn = trainer.envs._connection_read_fns.pop(0)
    write_fn = trainer.envs._connection_write_fns.pop(0)
    # write_fn((CALL_COMMAND, (ACTION_SPACE_NAME, None)))
    # CALL_COMMAND = "call"
    # ACTION_SPACE_NAME = "action_space"
    action_spaces = [
            read_fn() 
        ]
    print(action_spaces)
    



class episodal_nav(threading.Thread):
    rospy.init_node("habitat", anonymous=False)
    _current_episode = 0
    _total_number_of_episodes = 0
    observations = []

    _x_axis = 0
    _y_axis = 1
    _z_axis = 2
    _dt = 0.00478
    _sensor_rate = 50  # hz
    _r = rospy.Rate(_sensor_rate)
    _action_rate = 10
    _a = rospy.Rate(_action_rate)

    def callback(self, msg):
        self.policy = list(map(int,msg.data))
        self.policy.reverse()
        print(self.policy)

    def __init__(self):
        threading.Thread.__init__(self)
        self.config = get_baselines_config(
        "./habitat_baselines/config/pointnav/ddppo_pointnav.yaml"
        )
         # @param {type:"string"}
        # steps_in_thousands = "10"  # @param {type:"string"}

        
        self.env = habitat.Env(config=habitat.get_config(env_config_file))
        # self.env._sim.agents[0].state.velocity = np.float32([0, 0, 0])
        # self.env._sim.agents[0].state.angular_velocity = np.float32([0, 0, 0])
        # self.env._sim.agents[0].state.position = np.float32([-5.793274826959732,0.07,-1.4784819169477985])
        # self.env._sim.agents[0].state.rotation = np.float32([0.0,0.0,0.0,1.0])
        self.env._sim.set_agent_state(np.float32([-5.793274826959732,-0.07,-1.4784819169477985]), np.float32([0.0,0.0,0.0,1.0]))
        self.observations = self.env.reset()
        print(self.env._sim.active_dataset)
        self._sensor_resolution = {
            "RGB": 256,
            "DEPTH": 256,
        }
        self._pub_rgb = rospy.Publisher("~rgb", numpy_msg(Floats), queue_size=1)
        self._pub_depth = rospy.Publisher("~depth", numpy_msg(Floats), queue_size=1)
        self._pub_pose = rospy.Publisher("~pose", PoseStamped, queue_size=1)
        rospy.Subscriber("~policy", numpy_msg(Floats), self.callback,  queue_size=1)
        self.policy = []
        # additional RGB sensor I configured

        print("created habitat_plant succsefully")

    
    def create_episode(self):
        sim = self.env._sim

        dset = habitat.datasets.make_dataset("PointNav-v1")
              
        goal_radius = 0.05
        goal_position = []
        goal = NavigationGoal(position=[-0.2931181443940609,-0.07,0.4782877141779114])
        agent_position = np.float32([-5.793274826959732,-0.07,-1.4784819169477985])
        agent_rotation = np.float32([0.0,0.0,0.0,1.0])

        dummy_episode = NavigationEpisode(
        goals=[goal],
        episode_id=0,
        scene_id="data/scene_datasets/mp3d/17DRP5sb8fy/17DRP5sb8fy.glb",
        start_position=agent_position,
        start_rotation=agent_rotation,
        )
        dset.episodes = [dummy_episode]
        # for ep in dset.episodes:
        #     ep.scene_id = "data/scene_datasets/mp3d/17DRP5sb8fy/17DRP5sb8fy.glb"
        print(dset.episodes)
        # scene_key = osp.basename(osp.dirname(osp.dirname(scene)))
        self.out_file = f"./data/datasets/pointnav/mp3d/v1/test/content/17DRP5sb8fy" + str(self._current_episode)+".json.gz"
        os.makedirs(osp.dirname(self.out_file), exist_ok=True)
        with gzip.open(self.out_file, "wt") as f:
            f.write(dset.to_json())

    def run_episode(self):
        lock.acquire()
        seed = "42"      
        self.config.defrost()
        self.config.TASK_CONFIG.SEED = int(seed)
        self.config.LOG_INTERVAL = 1
        self.config.TASK_CONFIG.DATASET.DATA_PATH = self.out_file
        self.config.freeze()
        print(self.config.TASK_CONFIG.DATASET)
        random.seed(self.config.TASK_CONFIG.SEED)
        np.random.seed(self.config.TASK_CONFIG.SEED)  
        trainer_init = baseline_registry.get_trainer(self.config.TRAINER_NAME)
        print(self.config.TRAINER_NAME)
        trainer = trainer_init(self.config)
        trainer.eval()
        # setup_eval(trainer)
        lock.release()
        
    # def _render(self):
    #     # self.env._update_step_stats()  # think this increments episode count
    #     sim_obs = self.env._sim.get_sensor_observations()
    #     self.observations = self.env._sim._sensor_suite.get_observations(sim_obs)
    #     self.observations.update(
    #         self.env._task.sensor_suite.get_observations(
    #             observations=self.observations, episode=self.env.current_episode
    #         )
    #     )
    
    def execute_policy(self):
        
        print("execute_policy called")
        while not (len(self.policy)==0):
            lock.acquire()
            action = self.policy.pop()
            print(action)
            self.observations.update(self.env.step(action))
            self._update_position()
            lock.release()
            self._a.sleep()
        # else:
        #     print("Last action done, create another episode pls")
        


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
        proj_quat = tf.transformations.quaternion_from_euler(0.0,top_down_map_angle-np.pi,0.0)
        # proj_quat = tf.transformations.quaternion_from_euler(euler[0]+np.pi,euler[2],euler[1])
        agent_pos_in_map_frame = convert_points_to_topdown(self.env.sim.pathfinder, [agent_pos])
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
        
        # self._render()

    def _update_attitude(self):
        """ update agent orientation given angular velocity and delta time"""
        state = self.env.sim.get_agent_state(0)
        yaw = state.angular_velocity[2] / 3.1415926 * 180
        dt = self._dt

        _rotate_local_fns = [
            hsim.SceneNode.rotate_x_local,
            hsim.SceneNode.rotate_y_local,
            hsim.SceneNode.rotate_z_local,
        ]
        _rotate_local_fns[self._y_axis](
            self.env._sim.agents[0].scene_node, mn.Deg(yaw * dt)
        )
        self.env._sim.agents[0].scene_node.rotation = self.env._sim.agents[
            0
        ].scene_node.rotation.normalized()
        # self._render()

    def run(self):
        """Publish sensor readings through ROS on a different thread.
            This method defines what the thread does when the start() method
            of the threading class is called
        """
        print("Run function called")
        while not rospy.is_shutdown():
            lock.acquire()
            rgb_with_res = np.concatenate(
                (
                    np.float32(self.observations["rgb"].ravel()),
                    np.array(
                        [self._sensor_resolution["RGB"], self._sensor_resolution["RGB"]]
                    ),
                )
            )

            # multiply by 10 to get distance in meters
            depth_with_res = np.concatenate(
                (
                    np.float32(self.observations["depth"].ravel() * 10),
                    np.array(
                        [
                            self._sensor_resolution["DEPTH"],
                            self._sensor_resolution["DEPTH"],
                        ]
                    ),
                )
            )
            lock.release()
            self._pub_rgb.publish(np.float32(rgb_with_res))
            self._pub_depth.publish(np.float32(depth_with_res))
            
            self._r.sleep()
            

    # def set_linear_velocity(self, vx, vy):
    #     self.env._sim.agents[0].state.velocity[0] = vx
    #     self.env._sim.agents[0].state.velocity[1] = vy

    # def set_yaw(self, yaw):
    #     self.env._sim.agents[0].state.angular_velocity[2] = yaw

    # def update_orientation(self):
    #     lock.acquire()
    #     self._update_attitude()
    #     self._update_position()
    #     self._render()
    #     lock.release()

    def set_dt(self, dt):
        self._dt = dt





def main():
    
    navigate = episodal_nav()
    navigate.create_episode()
    navigate.run_episode()
    print("back in main")
    # navigate.start()
    # navigate.execute_policy()

    
    # navigate.run_episode()

if __name__ == "__main__":
    main()